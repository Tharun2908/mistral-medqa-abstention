"""
train_clean_dpo.py
------------------

Clean DPO training for learned abstention.

Protocol:
    - Starts policy AND reference from the frozen clean warm-start adapter.
    - Uses only clean OOF-derived DPO pairs.
    - Uses internal MedQA-train-derived preference validation for checkpoint
      selection.
    - Official MedQA dev/test are never touched during training.
    - Fixed DPO recipe across ratio ablations.

Default recipe:
    LR    = 5e-6
    beta  = 0.1
    epochs = 2
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
)
from trl import DPOConfig, DPOTrainer


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "data"
)

WARMSTART_DIR = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "warmstart"
    / "lr5e6"
    / "policy"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "dpo"
)

PAIR_FILES = {
    "ratio_1to1":
        DATA_DIR
        / "dpo_pairs_ratio_1to1.json",

    "ratio_1p5to1":
        DATA_DIR
        / "dpo_pairs_ratio_1p5to1.json",

    "ratio_2to1":
        DATA_DIR
        / "dpo_pairs_ratio_2to1.json",
}

LR = 5e-6
BETA = 0.1
EPOCHS = 2
SEED = 42

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 960


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair-set",
        required=True,
        choices=list(PAIR_FILES),
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
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
# Dataset conversion
# ---------------------------------------------------------------------

def to_dataset(rows):

    return Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for row in rows
        ]
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    set_seed(SEED)

    pair_file = PAIR_FILES[
        args.pair_set
    ]

    run_dir = (
        OUTPUT_ROOT
        / args.run_name
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    final_dir = (
        run_dir
        / "final"
    )

    if final_dir.exists():
        raise RuntimeError(
            f"Final output already exists:\n"
            f"{final_dir}"
        )

    # --------------------------------------------------------------
    # Load pair data
    # --------------------------------------------------------------

    print("=" * 72)
    print("CLEAN DPO TRAINING")
    print("=" * 72)

    print(f"Pair set   : {args.pair_set}")
    print(f"Pair file  : {pair_file}")
    print(f"Warm-start : {WARMSTART_DIR}")
    print(f"LR         : {LR}")
    print(f"Beta       : {BETA}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Seed       : {SEED}")

    with open(
        pair_file,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if (
        data.get("protocol")
        != "clean_dpo_pairs_v1"
    ):
        raise RuntimeError(
            f"Unexpected pair protocol: "
            f"{data.get('protocol')}"
        )

    train_rows = data["train"]
    val_rows = data["val"]

    if len(train_rows) != 2160:
        raise RuntimeError(
            f"Expected 2160 train pairs, "
            f"got {len(train_rows)}"
        )

    if len(val_rows) != 240:
        raise RuntimeError(
            f"Expected 240 val pairs, "
            f"got {len(val_rows)}"
        )

    train_types = {
        "abstain": sum(
            x["type"] == "abstain"
            for x in train_rows
        ),
        "answer": sum(
            x["type"] == "answer"
            for x in train_rows
        ),
    }

    val_types = {
        "abstain": sum(
            x["type"] == "abstain"
            for x in val_rows
        ),
        "answer": sum(
            x["type"] == "answer"
            for x in val_rows
        ),
    }

    print(
        f"\nTrain pairs : {len(train_rows)} "
        f"{train_types}"
    )

    print(
        f"Val pairs   : {len(val_rows)} "
        f"{val_types}"
    )

    # --------------------------------------------------------------
    # Smoke subset
    # --------------------------------------------------------------

    if args.smoke:

        train_rows = train_rows[:80]
        val_rows = val_rows[:40]

        print(
            "\nSMOKE MODE:"
        )

        print(
            f"  train={len(train_rows)}"
        )

        print(
            f"  val={len(val_rows)}"
        )

    train_ds = to_dataset(
        train_rows
    )

    val_ds = to_dataset(
        val_rows
    )

    # --------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = "left"

    print(
        "\nCompletion tokenization:"
    )

    completions = [
        " The answer is A.",
        " The answer is B.",
        " The answer is C.",
        " The answer is D.",
        " I cannot answer confidently.",
    ]

    for completion in completions:

        ids = tokenizer(
            completion,
            add_special_tokens=False,
        ).input_ids

        print(
            f"  {repr(completion)} -> "
            f"{ids}"
        )

        if len(ids) == 0:
            raise RuntimeError(
                "Empty completion tokenization."
            )

    # --------------------------------------------------------------
    # Base model
    # --------------------------------------------------------------

    print(
        "\nLoading Mistral-7B in 4-bit..."
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
        )
    )

    base.config.use_cache = False

    base = (
        prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },
        )
    )

    # --------------------------------------------------------------
    # Policy + reference
    # --------------------------------------------------------------

    print(
        "\nLoading frozen clean warm-start "
        "as policy + reference..."
    )

    model = PeftModel.from_pretrained(
        base,
        str(WARMSTART_DIR),
        adapter_name="policy",
        is_trainable=True,
    )

    model.load_adapter(
        str(WARMSTART_DIR),
        adapter_name="reference",
        is_trainable=False,
    )

    model.set_adapter(
        "policy"
    )

    model.config.use_cache = False

    model.print_trainable_parameters()

    # --------------------------------------------------------------
    # Training config
    # --------------------------------------------------------------

    if args.smoke:

        epochs = 1
        eval_steps = 5
        save_steps = 5
        logging_steps = 1

    else:

        epochs = EPOCHS
        eval_steps = 25
        save_steps = 25
        logging_steps = 10

    config = DPOConfig(
        output_dir=str(
            checkpoint_dir
        ),

        num_train_epochs=epochs,

        per_device_train_batch_size=2,

        per_device_eval_batch_size=2,

        gradient_accumulation_steps=4,

        learning_rate=LR,

        lr_scheduler_type="cosine",

        warmup_ratio=0.05,

        beta=BETA,

        model_adapter_name="policy",

        ref_adapter_name="reference",

        eval_strategy="steps",

        eval_steps=eval_steps,

        save_strategy="steps",

        save_steps=save_steps,

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        max_length=MAX_LENGTH,

        max_prompt_length=MAX_PROMPT_LENGTH,

        logging_steps=logging_steps,

        fp16=False,

        bf16=False,

        report_to="none",

        seed=SEED,

        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    print(
        "\nStarting DPO training..."
    )

    result = trainer.train()

    print(
        "\nTraining finished."
    )

    # --------------------------------------------------------------
    # Final internal validation
    # --------------------------------------------------------------

    print(
        "\nEvaluating restored "
        "best checkpoint..."
    )

    eval_metrics = (
        trainer.evaluate()
    )

    for key, value in (
        eval_metrics.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------------
    # Save ONLY policy adapter
    # --------------------------------------------------------------

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.set_adapter(
        "policy"
    )

    model.save_pretrained(
        str(final_dir),
        selected_adapters=[
            "policy"
        ],
    )

    tokenizer.save_pretrained(
        str(run_dir)
    )

    # PEFT normally creates:
    # final/policy/
    final_adapter = (
        final_dir
        / "policy"
    )

    metadata = {
        "protocol":
            "clean_dpo_train_v1",

        "pair_set":
            args.pair_set,

        "pair_file":
            str(pair_file),

        "warmstart":
            str(WARMSTART_DIR),

        "learning_rate":
            LR,

        "beta":
            BETA,

        "epochs":
            epochs,

        "seed":
            SEED,

        "smoke":
            args.smoke,

        "train_n":
            len(train_rows),

        "val_n":
            len(val_rows),

        "train_metrics":
            result.metrics,

        "eval_metrics":
            eval_metrics,

        "best_checkpoint":
            trainer.state.best_model_checkpoint,

        "best_metric":
            trainer.state.best_metric,
    }

    with open(
        run_dir / "run_metadata.json",
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
        "CLEAN DPO TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Best checkpoint : "
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        f"Best eval loss  : "
        f"{trainer.state.best_metric}"
    )

    print(
        f"\nSaved policy ->\n"
        f"{final_adapter}"
    )


if __name__ == "__main__":
    main()
