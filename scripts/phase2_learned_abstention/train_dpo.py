"""
train_dpo.py
------------
DPO (Direct Preference Optimization) training for learned abstention.

Goal:
- Prefer abstaining over confident wrong answers.
- Prefer answering over abstaining when the model is confident and correct.

Key design decisions:
- Start from SFT LoRA adapter: Primeinvincible/mistral-medqa-lora-v3
- Reference model = frozen copy of the same SFT adapter
  -> KL penalty pulls toward the SFT model, not raw base Mistral
- Policy adapter = trainable
- Reference adapter = frozen
- Low LR: 5e-6
- Beta: 0.1
- 2 epochs for full run to reduce overfitting risk on small DPO dataset
- Frequent eval/save every 25 steps
- No load_best_model_at_end because PEFT saves adapter_model.safetensors,
  not pytorch_model.bin.

Input:
- dpo_pairs.json

Output:
- ./mistral-medqa-dpo-final
"""

# ── Reduce TensorFlow warning noise before importing transformers ─────────────
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig


# ── Global Config ─────────────────────────────────────────────────────────────

# You already passed smoke test, so this is now full training.
SMOKE_TEST = False

BASE_MODEL = "mistralai/Mistral-7B-v0.3"
SFT_ADAPTER = "Primeinvincible/mistral-medqa-lora-v3"

OUTPUT_DIR = "./mistral-medqa-dpo"
FINAL_SAVE_DIR = "./mistral-medqa-dpo-final"

# Keep False unless you are logged into Hugging Face inside the pod.
PUSH_TO_HUB = False
HUB_REPO = "Primeinvincible/mistral-medqa-dpo"


# ── 1. Load Tokenizer ─────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Mistral has no separate pad token by default.
tokenizer.pad_token = tokenizer.eos_token

# Left padding is preferred for causal LM preference training.
tokenizer.padding_side = "left"

print("\nTokenizer sanity check:")
for label in [" A", " B", " C", " D", " E"]:
    ids = tokenizer.encode(label, add_special_tokens=False)
    print(f"  '{label}' -> {ids} -> '{tokenizer.decode(ids)}'")


# ── 2. Load Base Model in 4-bit ───────────────────────────────────────────────

print("\nLoading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

base_model.config.use_cache = False

base_model = prepare_model_for_kbit_training(
    base_model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)


# ── 3. Load SFT Adapter as Policy + Reference ─────────────────────────────────

print("\nLoading SFT adapter as policy (trainable)...")

model = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER,
    adapter_name="policy",
    is_trainable=True,
)

print("Loading SFT adapter as reference (frozen)...")

model.load_adapter(
    SFT_ADAPTER,
    adapter_name="reference",
    is_trainable=False,
)

# Set policy as active trainable adapter.
model.set_adapter("policy")
model.config.use_cache = False
model.print_trainable_parameters()


# ── 4. Load DPO Pairs ─────────────────────────────────────────────────────────

print("\nLoading DPO pairs...")

with open("dpo_pairs.json", "r") as f:
    dpo_data = json.load(f)

train_pairs = dpo_data["train"]
val_pairs = dpo_data["val"]

if SMOKE_TEST:
    train_pairs = train_pairs[:50]
    val_pairs = val_pairs[:20]
    print(f"SMOKE TEST: {len(train_pairs)} train / {len(val_pairs)} val pairs")
else:
    print(f"FULL: {len(train_pairs)} train / {len(val_pairs)} val pairs")


# ── Safe sample printing ──────────────────────────────────────────────────────

abstain_sample = next(
    (
        p for p in train_pairs
        if "cannot answer" in p["chosen"].lower()
        or p["chosen"].strip() == "E"
    ),
    None,
)

answer_sample = next(
    (
        p for p in train_pairs
        if not (
            "cannot answer" in p["chosen"].lower()
            or p["chosen"].strip() == "E"
        )
    ),
    None,
)

if abstain_sample:
    print("\nSample abstain pair:")
    print(f"  Prompt  : {abstain_sample['prompt'][:120]}...")
    print(f"  Chosen  : {abstain_sample['chosen']!r}")
    print(f"  Rejected: {abstain_sample['rejected']!r}")

if answer_sample:
    print("\nSample answer pair:")
    print(f"  Prompt  : {answer_sample['prompt'][:120]}...")
    print(f"  Chosen  : {answer_sample['chosen']!r}")
    print(f"  Rejected: {answer_sample['rejected']!r}")


# Convert to HuggingFace Dataset.
train_dataset = Dataset.from_list(
    [
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in train_pairs
    ]
)

val_dataset = Dataset.from_list(
    [
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in val_pairs
    ]
)


# ── 5. DPO Configuration ──────────────────────────────────────────────────────

if SMOKE_TEST:
    num_epochs = 2
    eval_steps = 10
    save_steps = 10
    log_steps = 5
else:
    # Safer full run:
    # 2 epochs reduces overfitting risk compared with 3 epochs.
    # Eval/save every 25 steps gives enough visibility.
    num_epochs = 2
    eval_steps = 25
    save_steps = 25
    log_steps = 10

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,

    # Training length
    num_train_epochs=num_epochs,

    # Batch settings
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch size = 8

    # DPO is sensitive, so keep LR lower than SFT
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    # Beta controls KL penalty strength.
    # 0.1 = conservative but still allows useful deviation from reference.
    beta=0.1,

    # Policy/reference adapter names
    model_adapter_name="policy",
    ref_adapter_name="reference",

    # Evaluation and checkpointing
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=3,

    # Important:
    # Keep this False for PEFT adapter training.
    # Otherwise Trainer may search for pytorch_model.bin, while PEFT saves
    # adapter_model.safetensors.
    load_best_model_at_end=False,

    # Precision
    fp16=False,
    bf16=False,

    # Sequence lengths
    max_length=512,
    max_prompt_length=480,

    # Logging
    logging_steps=log_steps,
    report_to="none",

    # Reproducibility
    seed=42,
)


# ── 6. DPO Trainer ────────────────────────────────────────────────────────────

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)


# ── 7. Train ──────────────────────────────────────────────────────────────────

print(f"\nStarting {'smoke test' if SMOKE_TEST else 'full DPO training'}...")
print(f"Policy   : {SFT_ADAPTER} (trainable)")
print(f"Reference: {SFT_ADAPTER} (frozen)")
print(f"Beta={dpo_config.beta}, LR={dpo_config.learning_rate}")
print(f"Epochs={num_epochs}, eval_steps={eval_steps}, save_steps={save_steps}")

print("\nMonitor:")
print("  rewards/chosen > rewards/rejected")
print("  rewards/margins positive or increasing")
print("  rewards/accuracies > 0.5")
print("  eval_loss not exploding")
print("  If eval_loss rises badly for multiple evals, stop and inspect checkpoints.")

train_result = trainer.train()


# ── 8. Final Evaluation ───────────────────────────────────────────────────────

print("\nRunning final evaluation...")
eval_metrics = trainer.evaluate()

print("\nFinal eval metrics:")
for key, value in eval_metrics.items():
    print(f"  {key}: {value}")


# ── 9. Save Final Adapter + Tokenizer ─────────────────────────────────────────

print(f"\nSaving final policy adapter locally to: {FINAL_SAVE_DIR}")

Path(FINAL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Make sure policy is active before saving.
model.set_adapter("policy")

# Save only the policy adapter when PEFT supports selected_adapters.
# If not, fall back to normal save_pretrained.
try:
    model.save_pretrained(FINAL_SAVE_DIR, selected_adapters=["policy"])
except TypeError:
    model.save_pretrained(FINAL_SAVE_DIR)

tokenizer.save_pretrained(FINAL_SAVE_DIR)

# Save metrics for later comparison.
train_metrics_path = Path(FINAL_SAVE_DIR) / "train_metrics.json"
eval_metrics_path = Path(FINAL_SAVE_DIR) / "eval_metrics.json"

with open(train_metrics_path, "w") as f:
    json.dump(train_result.metrics, f, indent=2)

with open(eval_metrics_path, "w") as f:
    json.dump(eval_metrics, f, indent=2)

print(f"Saved train metrics to: {train_metrics_path}")
print(f"Saved eval metrics to : {eval_metrics_path}")


# ── 10. Optional Push to HuggingFace Hub ──────────────────────────────────────

if PUSH_TO_HUB:
    print(f"\nPushing model to HuggingFace Hub: {HUB_REPO}")
    model.push_to_hub(HUB_REPO)
    tokenizer.push_to_hub(HUB_REPO)
    print(f"Done! Pushed to: {HUB_REPO}")
else:
    print("\nPUSH_TO_HUB=False, so model was saved locally only.")
    print(f"Final local path: {FINAL_SAVE_DIR}")
    print("Note: because the adapter is named 'policy', PEFT may save it inside")
    print("      ./mistral-medqa-dpo-final/policy depending on PEFT behavior.")


# ── 11. Final Message ─────────────────────────────────────────────────────────

if SMOKE_TEST:
    print("\nSmoke test complete! Check for:")
    print(" No errors")
    print(" eval_loss in logs")
    print(" rewards/chosen > rewards/rejected")
    print(" rewards/margins positive or increasing")
    print(" rewards/accuracies > 0.5")
    print("\nIf all good -> set SMOKE_TEST = False and run full training")
else:
    print("\nFull DPO training complete!")
    print(f"Saved final adapter/tokenizer to: {FINAL_SAVE_DIR}")
