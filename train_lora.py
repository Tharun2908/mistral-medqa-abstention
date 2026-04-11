"""
train_lora.py
-------------
Fine-tune Mistral-7B-v0.3 on MedQA USMLE using QLoRA + Early Stopping.

Key design decisions:
- Train/val split from train set only — test set kept untouched for final eval
- Prompt-completion format — loss computed on completion only
- Space moved into completion (" A") to make prompt/completion tokenization stable
- prepare_model_for_kbit_training() called before LoRA
- Mixed precision disabled (fp16=False, bf16=False) for stability on this setup
- Early stopping monitors val loss
- Smoke test flag for verifying setup before full training run

Model   : mistralai/Mistral-7B-v0.3
Dataset : GBaker/MedQA-USMLE-4-options
Method  : QLoRA (4-bit quantization + LoRA adapters)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
from datasets import load_dataset

# ── SMOKE TEST FLAG ────────────────────────────────────────────────────────────
# True  -> tiny run to verify setup
# False -> full training
SMOKE_TEST = False

# ── 1. Load Tokenizer ──────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 2. Load Model with 4-bit Quantization ─────────────────────────────────────
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # V100 → fp16, not bf16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

# ── 3. Prepare for k-bit Training ─────────────────────────────────────────────
print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# ── 4. LoRA Configuration ──────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ── 5. Load & Split Dataset ────────────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

split = dataset["train"].train_test_split(test_size=0.1, seed=42)

if SMOKE_TEST:
    train_dataset = split["train"].select(range(40))
    eval_dataset = split["test"].select(range(10))
    print("SMOKE TEST MODE: 40 train / 10 eval examples")
else:
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"FULL MODE: {len(train_dataset)} train / {len(eval_dataset)} eval examples")

print(f"Test set (untouched): {len(dataset['test'])} examples")

# ── 6. Format as Prompt-Completion ─────────────────────────────────────────────
def format_training_example(example):
    """
    Prompt-completion format for completion-only loss.

    Important boundary choice:
    - prompt ends with 'Answer:'
    - completion starts with leading space, e.g. ' B'

    This avoids tokenizer boundary mismatch warnings that can happen when
    prompt ends with a space and completion is just 'B'.
    """
    options_str = "\n".join(
        [f"{k}: {example['options'][k]}" for k in ["A", "B", "C", "D"]]
    )

    return {
        "prompt": (
            f"Question: {example['question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Answer:"
        ),
        "completion": f" {example['answer_idx']}",   # e.g. " B"
    }

train_dataset = train_dataset.map(format_training_example)
eval_dataset = eval_dataset.map(format_training_example)

# ── 7. Training Arguments ──────────────────────────────────────────────────────
if SMOKE_TEST:
    num_epochs = 2
    eval_steps = 5
    save_steps = 5
    log_steps = 5
else:
    num_epochs = 10
    eval_steps = 200
    save_steps = 200
    log_steps = 50

training_args = TrainingArguments(
    output_dir="./mistral-medqa-lora-v3",
    num_train_epochs=num_epochs,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    # If this errors on your installed transformers version,
    # replace eval_strategy with evaluation_strategy
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    fp16=False,
    bf16=False,

    logging_steps=log_steps,
    logging_dir="./logs",
    report_to="none",

    save_total_limit=2,
    max_grad_norm=1.0,
)

# ── 8. Early Stopping Callback ─────────────────────────────────────────────────
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
)

# ── 9. Trainer ─────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=[early_stopping],
)

# ── 10. Train ──────────────────────────────────────────────────────────────────
print(f"\nStarting {'smoke test' if SMOKE_TEST else 'full training'}...")
print("Early stopping: patience=3, monitoring eval_loss")
trainer.train()

# ── 11. Save Model ─────────────────────────────────────────────────────────────
if not SMOKE_TEST:
    print("\nSaving model to HuggingFace...")
    model.push_to_hub("Primeinvincible/mistral-medqa-lora-v3")
    tokenizer.push_to_hub("Primeinvincible/mistral-medqa-lora-v3")
    print("Done! Model saved as mistral-medqa-lora-v3")
else:
    print("\nSmoke test complete! Check for:")
    print("  No tokenizer mismatch warnings")
    print("  No fp16/bf16 crash")
    print("  eval_loss appears in logs")
    print("  Checkpoints saved")
    print("  Early stopping callback attached")
