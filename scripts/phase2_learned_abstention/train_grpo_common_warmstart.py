"""
train_warmstart.py
------------------

Clean warm-start SFT for learned abstention.

Protocol:
    - Starts from the original MedQA SFT adapter.
    - Trains ONLY on clean OOF-derived MedQA-train examples.
    - Prompt tokens are masked: completion-only loss.
    - No MedQA dev/test data is used during training.
    - LR is supplied explicitly so candidate warm-starts can later
      be compared on official MedQA dev only.

Input:
    results/clean_protocol/learned_abstention/data/warmstart_data.json

Output:
    results/clean_protocol/learned_abstention/warmstart/<run_name>/
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


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BASE_MODEL = "mistralai/Mistral-7B-v0.3"
SFT_ADAPTER = str(
    Path(__file__).resolve().parents[2]
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "continue_sft_control"
    / "main"
    / "checkpoints"
    / "checkpoint-1000"
)

REPO_ROOT = Path(__file__).resolve().parents[2]

WARMSTART_DATA = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "data"
    / "warmstart_data.json"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo_common_warmstart"
)

MAX_LEN = 1024
EPOCHS = 1
SEED = 42


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Warm-start learning rate.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Output directory name.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only a tiny training subset.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

def build_example(rec, tokenizer):

    prompt = rec["prompt"]

    completion_ids = tokenizer(
        rec["completion"],
        add_special_tokens=False,
    ).input_ids

    # Explicit EOS.
    completion_ids = (
        completion_ids
        + [tokenizer.eos_token_id]
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    ).input_ids

    # Keep the complete target. If necessary, trim prompt from the left.
    max_prompt_len = MAX_LEN - len(completion_ids)

    if max_prompt_len <= 0:
        raise RuntimeError(
            "Completion itself exceeds MAX_LEN."
        )

    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = (
        prompt_ids
        + completion_ids
    )

    labels = (
        [-100] * len(prompt_ids)
        + completion_ids
    )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    set_seed(SEED)

    run_dir = (
        OUTPUT_ROOT
        / args.run_name
    )

    adapter_dir = (
        run_dir
        / "policy"
    )

    if adapter_dir.exists():
        raise RuntimeError(
            f"Output already exists:\n{adapter_dir}"
        )

    # --------------------------------------------------------------
    # Load clean data
    # --------------------------------------------------------------

    print("=" * 72)
    print("CLEAN WARM-START SFT")
    print("=" * 72)

    print(f"Run name : {args.run_name}")
    print(f"LR       : {args.lr}")
    print(f"Epochs   : {EPOCHS}")

    with open(
        WARMSTART_DATA,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data.get("protocol") != "clean_oof_v1":
        raise RuntimeError(
            f"Unexpected protocol: "
            f"{data.get('protocol')}"
        )

    examples = data["examples"]

    if len(examples) != 2300:
        raise RuntimeError(
            f"Expected 2300 warm-start examples, "
            f"got {len(examples)}"
        )

    n_abstain = sum(
        x["type"] == "abstain"
        for x in examples
    )

    n_answer = sum(
        x["type"] == "answer"
        for x in examples
    )

    print(
        f"\nLoaded {len(examples)} examples"
    )
    print(
        f"  abstain : {n_abstain}"
    )
    print(
        f"  answer  : {n_answer}"
    )

    assert n_abstain == 800
    assert n_answer == 1500

    if args.smoke:
        examples = examples[:80]
        print(
            f"\nSMOKE MODE: using "
            f"{len(examples)} examples"
        )

    # --------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = [
        build_example(x, tokenizer)
        for x in examples
    ]

    # --------------------------------------------------------------
    # Completion-only sanity check
    # --------------------------------------------------------------

    sample = tokenized[0]

    total_tokens = len(
        sample["input_ids"]
    )

    supervised_tokens = sum(
        x != -100
        for x in sample["labels"]
    )

    masked_tokens = (
        total_tokens
        - supervised_tokens
    )

    print(
        "\nCompletion-only sanity check:"
    )

    print(
        f"  total tokens      : "
        f"{total_tokens}"
    )

    print(
        f"  supervised tokens : "
        f"{supervised_tokens}"
    )

    print(
        f"  masked tokens     : "
        f"{masked_tokens}"
    )

    if not (
        0
        < supervised_tokens
        < total_tokens
    ):
        raise RuntimeError(
            "Completion-only masking failed."
        )

    print(
        "Completion-only masking: PASS"
    )

    dataset = Dataset.from_list(
        tokenized
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print(
        "\nLoading Mistral-7B in 4-bit..."
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base.config.use_cache = False

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
    )

    print(
        "\nLoading frozen continue-SFT adapter..."
    )

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

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.smoke:

        train_epochs = 1

    else:

        train_epochs = EPOCHS

    training_args = TrainingArguments(
        output_dir=str(
            run_dir / "checkpoints"
        ),

        num_train_epochs=train_epochs,

        per_device_train_batch_size=4,

        gradient_accumulation_steps=4,

        learning_rate=args.lr,

        lr_scheduler_type="cosine",

        warmup_ratio=0.05,

        logging_steps=10,

        save_strategy="no",

        fp16=False,
        bf16=False,

        report_to="none",

        seed=SEED,

        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(
        f"\nStarting warm-start training..."
    )

    result = trainer.train()

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        run_dir
    )

    metadata = {
        "protocol": "clean_oof_v1",
        "run_name": args.run_name,
        "learning_rate": args.lr,
        "epochs": train_epochs,
        "seed": SEED,
        "n_training_examples": len(
            examples
        ),
        "source": str(
            WARMSTART_DATA
        ),
        "train_metrics": result.metrics,
    }

    with open(
        run_dir / "train_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 72
    )

    print(
        "WARM-START TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Saved adapter ->\n"
        f"{adapter_dir}"
    )


if __name__ == "__main__":
    main()
