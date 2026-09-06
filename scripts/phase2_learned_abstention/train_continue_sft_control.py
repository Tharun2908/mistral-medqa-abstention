"""
train_continue_sft_control.py
-----------------------------

Matched-compute A-D-only continued-SFT control for the clean MedQA abstention study.

Purpose
-------
The supervised 5-way model improved both:
  1) abstention behavior, and
  2) would-be answer accuracy.

This control asks whether the improvement comes simply from giving the original
MedQA SFT adapter another supervised pass.

Fairness constraints
--------------------
- Same starting adapter as supervised 5-way.
- Same exact 9,160 / 1,018 train/validation examples.
- Same full-sentence completion format.
- Same LR, batch size, grad accumulation, scheduler, warmup, epochs, seed.
- Same eval/save cadence.
- The comparison checkpoint is FIXED at step 1000 because the selected
  supervised 5-way model restored checkpoint-1000.

Only the target differs:
  supervised 5-way:
      OOF-correct -> gold A/B/C/D
      OOF-wrong   -> E
  this control:
      ALL examples -> gold A/B/C/D

Official MedQA dev/test are not used during training.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

BASE_MODEL = "mistralai/Mistral-7B-v0.3"
SFT_ADAPTER = "Primeinvincible/mistral-medqa-lora-v3"

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "data"
    / "supervised_5way_data.json"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "continue_sft_control"
)

LR = 5e-6
EPOCHS = 2
MAX_LEN = 1024
SEED = 42
MATCHED_CHECKPOINT_STEP = 1000

COMPLETIONS = {
    "A": " The answer is A.",
    "B": " The answer is B.",
    "C": " The answer is C.",
    "D": " The answer is D.",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_example(record, tokenizer):
    prompt_ids = tokenizer(
        record["prompt"],
        add_special_tokens=True,
    ).input_ids

    # IMPORTANT: ignore the supervised-5way target_label.
    # Every example receives its gold medical answer.
    gold = record["gold_answer"]

    if gold not in COMPLETIONS:
        raise RuntimeError(f"Invalid gold answer: {gold}")

    completion = COMPLETIONS[gold]

    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
    ).input_ids + [tokenizer.eos_token_id]

    max_prompt_len = MAX_LEN - len(completion_ids)

    if max_prompt_len <= 0:
        raise RuntimeError("Completion exceeds MAX_LEN.")

    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def main():
    args = parse_args()
    set_seed(SEED)

    run_dir = OUTPUT_ROOT / args.run_name
    checkpoint_dir = run_dir / "checkpoints"

    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Run directory already exists and is non-empty:\n{run_dir}"
        )

    print("=" * 78)
    print("MATCHED-COMPUTE CONTINUE-SFT A-D-ONLY CONTROL")
    print("=" * 78)
    print(f"Start adapter            : {SFT_ADAPTER}")
    print(f"Learning rate            : {LR}")
    print(f"Epochs                   : {EPOCHS}")
    print(f"Seed                     : {SEED}")
    print(f"Matched comparison step  : {MATCHED_CHECKPOINT_STEP}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("protocol") != "clean_supervised_5way_v1":
        raise RuntimeError(
            f"Unexpected data protocol: {data.get('protocol')}"
        )

    train_rows = data["train"]
    val_rows = data["val"]

    if len(train_rows) != 9160 or len(val_rows) != 1018:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_rows)}, val={len(val_rows)}"
        )

    print(f"\nTrain examples : {len(train_rows)}")
    print(f"Val examples   : {len(val_rows)}")

    # Check that both OOF-correct and OOF-wrong examples are present,
    # while our control target remains gold A-D for all of them.
    n_oof_correct = sum(bool(x["oof_correct"]) for x in train_rows)
    n_oof_wrong = len(train_rows) - n_oof_correct

    print(f"Train OOF-correct source examples : {n_oof_correct}")
    print(f"Train OOF-wrong source examples   : {n_oof_wrong}")
    print("Control target for BOTH groups    : gold A/B/C/D")

    if args.smoke:
        train_rows = train_rows[:80]
        val_rows = val_rows[:40]
        print("\nSMOKE MODE")
        print(f"  train={len(train_rows)}")
        print(f"  val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\nFull-sentence A-D completions:")
    for label, completion in COMPLETIONS.items():
        ids = tokenizer(
            completion,
            add_special_tokens=False,
        ).input_ids
        print(f"  {label}: {completion!r} -> {ids}")

    print("\nTokenizing with completion-only labels...")

    train_tokenized = [
        tokenize_example(x, tokenizer)
        for x in train_rows
    ]
    val_tokenized = [
        tokenize_example(x, tokenizer)
        for x in val_rows
    ]

    sample = train_tokenized[0]
    n_total = len(sample["labels"])
    n_supervised = sum(x != -100 for x in sample["labels"])

    print("\nCompletion-only sanity check:")
    print(f"  total tokens      : {n_total}")
    print(f"  supervised tokens : {n_supervised}")
    print(f"  masked tokens     : {n_total - n_supervised}")

    if not (0 < n_supervised < n_total):
        raise RuntimeError("Completion-only masking failed.")

    print("Completion-only masking: PASS")

    train_ds = Dataset.from_list(train_tokenized)
    val_ds = Dataset.from_list(val_tokenized)

    print("\nLoading base model in 4-bit...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
    )
    base.config.use_cache = False

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    print("\nLoading original MedQA SFT adapter as trainable...")

    model = PeftModel.from_pretrained(
        base,
        SFT_ADAPTER,
        is_trainable=True,
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    if args.smoke:
        epochs = 1
        eval_steps = 5
        save_steps = 5
        logging_steps = 1
        save_total_limit = 5
    else:
        epochs = EPOCHS
        eval_steps = 100
        save_steps = 100
        logging_steps = 25

        # Keep checkpoint-1000 even if another checkpoint has lower eval loss.
        save_total_limit = 20

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        eval_strategy="steps",
        eval_steps=eval_steps,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,

        # Intentionally do NOT restore the control's own best checkpoint.
        # Our matched comparison is explicitly checkpoint-1000.
        load_best_model_at_end=False,

        fp16=False,
        bf16=False,

        logging_steps=logging_steps,
        report_to="none",
        max_grad_norm=1.0,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("\nStarting continued A-D-only SFT...")
    result = trainer.train()

    print("\nTraining finished.")

    final_eval = trainer.evaluate()

    print("\nFinal-step internal validation:")
    for k, v in final_eval.items():
        print(f"  {k}: {v}")

    metadata = {
        "protocol": "clean_continue_sft_matched_compute_v1",
        "purpose": (
            "Control for extra supervised answer training in the supervised "
            "5-way abstention experiment."
        ),
        "source_data": str(DATA_FILE),
        "start_adapter": SFT_ADAPTER,
        "target_definition": "gold A/B/C/D full-sentence completion for every example",
        "learning_rate": LR,
        "epochs": EPOCHS,
        "seed": SEED,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "matched_comparison_checkpoint_step": MATCHED_CHECKPOINT_STEP,
        "train_metrics": result.metrics,
        "final_step_eval_metrics": final_eval,
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    with open(
        run_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 78)

    if args.smoke:
        print("CONTINUE-SFT SMOKE COMPLETE")
        print("=" * 78)
        print(f"Checkpoints -> {checkpoint_dir}")
        return

    matched_checkpoint = checkpoint_dir / f"checkpoint-{MATCHED_CHECKPOINT_STEP}"

    if not matched_checkpoint.exists():
        raise RuntimeError(
            f"Expected matched checkpoint does not exist:\n{matched_checkpoint}"
        )

    print("CONTINUE-SFT CONTROL TRAINING COMPLETE")
    print("=" * 78)
    print(f"Full run directory       : {run_dir}")
    print(f"Matched comparison model : {matched_checkpoint}")
    print(
        "\nIMPORTANT: evaluate checkpoint-1000, NOT the final-step model "
        "and NOT a dev-selected checkpoint."
    )


if __name__ == "__main__":
    main()
