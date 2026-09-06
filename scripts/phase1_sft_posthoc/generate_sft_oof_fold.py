"""
generate_sft_oof_fold.py
------------------------

Generate ONE fold of clean out-of-fold (OOF) SFT predictions on MedQA train.

Purpose
-------
Warm-start and DPO pair construction depend on:
    - whether SFT answered correctly
    - which answer SFT predicted
    - SFT confidence

We do not want those signals to come from a model that trained on the
same example.

Protocol
--------
Official MedQA train: 10,178 examples

5-fold OOF:
    outer fold -> completely held out for prediction

Remaining 4 folds:
    90% -> SFT training
    10% -> early-stopping validation

Then:
    train QLoRA SFT
    restore best checkpoint
    predict ONLY outer held-out fold

After running folds 0-4, every MedQA training example will have exactly
one prediction from a model that never trained on that example.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =====================================================================
# Repo import
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.medqa_data import load_medqa


# =====================================================================
# Configuration
# =====================================================================

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

N_FOLDS = 5
SEED = 42

MAX_LEN = 1024

ANSWER_SET = ["A", "B", "C", "D"]

RESULT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "oof_sft"
)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# CLI
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(N_FOLDS),
        help="OOF fold number: 0, 1, 2, 3, or 4",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny training run used only to verify the pipeline.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint for this fold.",
    )

    return parser.parse_args()


# =====================================================================
# Prompt formatting
# =====================================================================

def format_prompt(example):

    options_str = "\n".join(
        f"{option}: {example['options'][option]}"
        for option in ANSWER_SET
    )

    return (
        f"Question: {example['question']}\n\n"
        f"Options:\n"
        f"{options_str}\n\n"
        f"Answer:"
    )


def add_training_fields(example):

    return {
        "prompt": format_prompt(example),
        "completion": f" {example['answer_idx']}",
    }


# =====================================================================
# Completion-only tokenization
# =====================================================================

def tokenize_completion_only(example, tokenizer):
    """
    Tokenize prompt + completion.

    Prompt tokens:
        labels = -100
        -> excluded from loss

    Completion tokens:
        labels = actual token IDs
        -> contribute to training loss

    This keeps the SFT objective completion-only.
    """

    prompt = example["prompt"]

    # Example:
    # " A" + EOS
    completion = (
        example["completion"]
        + tokenizer.eos_token
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    ).input_ids

    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
    ).input_ids

    # Ensure completion itself always survives truncation.
    max_prompt_len = (
        MAX_LEN
        - len(completion_ids)
    )

    if max_prompt_len <= 0:
        raise RuntimeError(
            "Completion is longer than MAX_LEN."
        )

    # If necessary, remove oldest prompt tokens.
    # The end of the prompt contains options + Answer:,
    # which is the most important portion.
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[
            -max_prompt_len:
        ]

    input_ids = (
        prompt_ids
        + completion_ids
    )

    labels = (
        [-100] * len(prompt_ids)
        + completion_ids
    )

    attention_mask = (
        [1] * len(input_ids)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =====================================================================
# Answer-token verification
# =====================================================================

def get_answer_token_ids(
    tokenizer,
    sample_prompt,
):

    base_ids = tokenizer(
        sample_prompt,
        add_special_tokens=False,
    ).input_ids

    answer_token_ids = {}

    print(
        "\nVerifying answer option tokenization..."
    )

    for option in ANSWER_SET:

        full_ids = tokenizer(
            sample_prompt + " " + option,
            add_special_tokens=False,
        ).input_ids

        continuation_ids = (
            full_ids[len(base_ids):]
        )

        print(
            f"  Option {option}: "
            f"continuation IDs = "
            f"{continuation_ids}"
        )

        if len(continuation_ids) != 1:
            raise RuntimeError(
                f"Option {option} is not "
                f"a one-token continuation: "
                f"{continuation_ids}"
            )

        answer_token_ids[option] = (
            continuation_ids[0]
        )

    print(
        "Tokenization check passed."
    )

    return answer_token_ids


# =====================================================================
# Prediction
# =====================================================================

@torch.no_grad()
def predict_one(
    model,
    tokenizer,
    example,
    answer_token_ids,
):

    prompt = format_prompt(example)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model(**inputs)

    last_logits = (
        outputs.logits[0, -1, :]
    )

    option_token_ids = torch.tensor(
        [
            answer_token_ids[x]
            for x in ANSWER_SET
        ],
        device=last_logits.device,
    )

    option_logits = (
        last_logits[
            option_token_ids
        ]
    )

    option_probs = torch.softmax(
        option_logits.float(),
        dim=0,
    )

    best_idx = int(
        torch.argmax(option_probs)
    )

    prediction = (
        ANSWER_SET[best_idx]
    )

    confidence = float(
        option_probs[best_idx]
    )

    all_probs = {
        option: float(
            option_probs[i]
        )
        for i, option
        in enumerate(ANSWER_SET)
    }

    return (
        prediction,
        confidence,
        all_probs,
    )


# =====================================================================
# Main
# =====================================================================

def main():

    args = parse_args()

    fold = args.fold

    run_seed = (
        SEED + fold
    )

    set_seed(run_seed)

    # --------------------------------------------------------------
    # Output location
    # --------------------------------------------------------------

    if args.smoke:
        run_name = (
            f"fold_{fold}_smoke"
        )
    else:
        run_name = (
            f"fold_{fold}"
        )

    fold_dir = (
        RESULT_ROOT
        / run_name
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_file = (
        fold_dir
        / "predictions.json"
    )

    if prediction_file.exists():
        raise RuntimeError(
            "\nPredictions already exist:\n"
            f"{prediction_file}\n\n"
            "Delete the folder explicitly "
            "if you really want to rerun it."
        )

    # --------------------------------------------------------------
    # Load official MedQA TRAIN only
    # --------------------------------------------------------------

    print(
        "=" * 72
    )

    print(
        f"OOF SFT — FOLD {fold}"
    )

    print(
        "=" * 72
    )

    full_train = load_medqa(
        "train"
    )

    n_total = len(
        full_train
    )

    if n_total != 10178:
        raise RuntimeError(
            f"Expected 10178 train examples, "
            f"got {n_total}"
        )

    labels = np.array(
        full_train[
            "answer_idx"
        ]
    )

    all_indices = np.arange(
        n_total
    )

    # --------------------------------------------------------------
    # Outer 5-fold split
    # --------------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_splits = list(
        skf.split(
            all_indices,
            labels,
        )
    )

    (
        outer_train_idx,
        holdout_idx,
    ) = fold_splits[fold]

    # --------------------------------------------------------------
    # Inner early-stopping validation
    #
    # Critically:
    # holdout_idx is NEVER used here.
    # --------------------------------------------------------------

    (
        inner_train_idx,
        inner_val_idx,
    ) = train_test_split(
        outer_train_idx,
        test_size=0.10,
        random_state=run_seed,
        stratify=labels[
            outer_train_idx
        ],
    )

    print()

    print(
        f"Full train        : "
        f"{n_total}"
    )

    print(
        f"Outer holdout     : "
        f"{len(holdout_idx)}"
    )

    print(
        f"Outer train pool  : "
        f"{len(outer_train_idx)}"
    )

    print(
        f"Inner SFT train   : "
        f"{len(inner_train_idx)}"
    )

    print(
        f"Inner early-stop  : "
        f"{len(inner_val_idx)}"
    )

    # --------------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------------

    train_set = set(
        inner_train_idx.tolist()
    )

    val_set = set(
        inner_val_idx.tolist()
    )

    holdout_set = set(
        holdout_idx.tolist()
    )

    if train_set & val_set:
        raise RuntimeError(
            "TRAIN / VAL overlap detected!"
        )

    if train_set & holdout_set:
        raise RuntimeError(
            "TRAIN / HOLDOUT overlap detected!"
        )

    if val_set & holdout_set:
        raise RuntimeError(
            "VAL / HOLDOUT overlap detected!"
        )

    print(
        "\nSplit disjointness: PASS"
    )

    # --------------------------------------------------------------
    # Smoke-test reduction
    # --------------------------------------------------------------

    if args.smoke:

        inner_train_idx = (
            inner_train_idx[:80]
        )

        inner_val_idx = (
            inner_val_idx[:20]
        )

        holdout_idx = (
            holdout_idx[:20]
        )

        print(
            "\nSMOKE MODE:"
        )

        print(
            f"  train   = "
            f"{len(inner_train_idx)}"
        )

        print(
            f"  val     = "
            f"{len(inner_val_idx)}"
        )

        print(
            f"  holdout = "
            f"{len(holdout_idx)}"
        )

    # --------------------------------------------------------------
    # Construct datasets
    # --------------------------------------------------------------

    sft_train = (
        full_train.select(
            inner_train_idx.tolist()
        )
    )

    sft_val = (
        full_train.select(
            inner_val_idx.tolist()
        )
    )

    holdout = (
        full_train.select(
            holdout_idx.tolist()
        )
    )

    # Add prompt/completion strings first.
    sft_train = sft_train.map(
        add_training_fields,
        desc="Formatting SFT train",
    )

    sft_val = sft_val.map(
        add_training_fields,
        desc="Formatting SFT validation",
    )

    # --------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = (
        "right"
    )

    # --------------------------------------------------------------
    # Explicit completion-only tokenization
    # --------------------------------------------------------------

    print(
        "\nTokenizing datasets "
        "with completion-only labels..."
    )

    train_columns = (
        sft_train.column_names
    )

    val_columns = (
        sft_val.column_names
    )

    sft_train = sft_train.map(
        lambda example:
            tokenize_completion_only(
                example,
                tokenizer,
            ),
        remove_columns=train_columns,
        desc="Tokenizing SFT train",
    )

    sft_val = sft_val.map(
        lambda example:
            tokenize_completion_only(
                example,
                tokenizer,
            ),
        remove_columns=val_columns,
        desc="Tokenizing SFT validation",
    )

    # --------------------------------------------------------------
    # Completion-only masking sanity check
    # --------------------------------------------------------------

    first_example = (
        sft_train[0]
    )

    n_tokens = len(
        first_example["labels"]
    )

    n_supervised = sum(
        token_id != -100
        for token_id
        in first_example["labels"]
    )

    print(
        "\nCompletion-only sanity check:"
    )

    print(
        f"  total tokens      : "
        f"{n_tokens}"
    )

    print(
        f"  supervised tokens : "
        f"{n_supervised}"
    )

    print(
        f"  masked tokens     : "
        f"{n_tokens - n_supervised}"
    )

    if n_supervised == 0:
        raise RuntimeError(
            "No completion tokens "
            "are supervised."
        )

    if n_supervised == n_tokens:
        raise RuntimeError(
            "Prompt masking FAILED: "
            "every token is supervised."
        )

    print(
        "Completion-only masking: PASS"
    )

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    print(
        "\nLoading Mistral-7B "
        "in 4-bit..."
    )

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
    )

    base_model.config.use_cache = (
        False
    )

    base_model = (
        prepare_model_for_kbit_training(
            base_model
        )
    )

    # --------------------------------------------------------------
    # Same LoRA recipe as original SFT
    # --------------------------------------------------------------

    lora_config = (
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )

    model = get_peft_model(
        base_model,
        lora_config,
    )

    model.print_trainable_parameters()

    # --------------------------------------------------------------
    # Dynamic padding
    # --------------------------------------------------------------

    collator = (
        DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        )
    )

    # --------------------------------------------------------------
    # Training parameters
    # --------------------------------------------------------------

    checkpoint_dir = (
        fold_dir
        / "checkpoints"
    )

    if args.smoke:

        epochs = 1

        eval_steps = 5
        save_steps = 5
        logging_steps = 1

    else:

        epochs = 10

        eval_steps = 200
        save_steps = 200
        logging_steps = 50

    training_args = (
        TrainingArguments(
            output_dir=str(
                checkpoint_dir
            ),

            num_train_epochs=epochs,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,

            gradient_accumulation_steps=4,

            learning_rate=2e-4,

            lr_scheduler_type="cosine",

            warmup_ratio=0.05,

            eval_strategy="steps",

            eval_steps=eval_steps,

            save_strategy="steps",

            save_steps=save_steps,

            load_best_model_at_end=True,

            metric_for_best_model=(
                "eval_loss"
            ),

            greater_is_better=False,

            # Preserve original SFT recipe.
            fp16=False,
            bf16=False,

            logging_steps=logging_steps,

            report_to="none",

            save_total_limit=2,

            max_grad_norm=1.0,

            seed=run_seed,
        )
    )

    # --------------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------------

    early_stopping = (
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )
    )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=sft_train,
        eval_dataset=sft_val,

        data_collator=collator,

        callbacks=[
            early_stopping,
        ],
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    print(
        f"\nStarting OOF SFT "
        f"fold {fold}..."
    )

    train_result = trainer.train(
        resume_from_checkpoint=args.resume
    )

    print(
        "\nTraining finished."
    )

    # load_best_model_at_end=True means
    # trainer.model now contains the best checkpoint.

    # --------------------------------------------------------------
    # Save best adapter
    # --------------------------------------------------------------

    best_adapter_dir = (
        fold_dir
        / "best_adapter"
    )

    trainer.model.save_pretrained(
        best_adapter_dir
    )

    tokenizer.save_pretrained(
        best_adapter_dir
    )

    # --------------------------------------------------------------
    # Prepare for held-out inference
    # --------------------------------------------------------------

    trainer.model.eval()

    trainer.model.config.use_cache = (
        True
    )

    sample_prompt = format_prompt(
        holdout[0]
    )

    answer_token_ids = (
        get_answer_token_ids(
            tokenizer,
            sample_prompt,
        )
    )

    # --------------------------------------------------------------
    # OOF inference
    # --------------------------------------------------------------

    print(
        f"\nPredicting outer held-out "
        f"fold: {len(holdout)} examples..."
    )

    predictions = []

    correct = 0

    for local_index, example in enumerate(
        tqdm(
            holdout,
            desc=(
                f"OOF predict fold {fold}"
            ),
        )
    ):

        (
            prediction,
            confidence,
            all_probs,
        ) = predict_one(
            trainer.model,
            tokenizer,
            example,
            answer_token_ids,
        )

        is_correct = (
            prediction
            == example["answer_idx"]
        )

        correct += int(
            is_correct
        )

        # Important:
        # retain original position within
        # the 10,178-example MedQA train set.
        original_train_index = int(
            holdout_idx[
                local_index
            ]
        )

        predictions.append(
            {
                "train_index":
                    original_train_index,

                "id":
                    example["id"],

                "question":
                    example["question"],

                "options":
                    example["options"],

                "answer_idx":
                    example["answer_idx"],

                "prediction":
                    prediction,

                "is_correct":
                    is_correct,

                "confidence":
                    confidence,

                "all_probs":
                    all_probs,

                "oof_fold":
                    fold,
            }
        )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    oof_accuracy = (
        correct
        / len(predictions)
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output = {
        "protocol":
            "clean_oof_v1",

        "fold":
            fold,

        "n_folds":
            N_FOLDS,

        "base_seed":
            SEED,

        "run_seed":
            run_seed,

        "smoke":
            args.smoke,

        "n_total_medqa_train":
            n_total,

        "n_outer_holdout":
            len(holdout_idx),

        "n_inner_train":
            len(inner_train_idx),

        "n_inner_val":
            len(inner_val_idx),

        "oof_accuracy":
            oof_accuracy,

        "best_checkpoint":
            trainer.state.best_model_checkpoint,

        "best_eval_loss":
            trainer.state.best_metric,

        "train_metrics":
            train_result.metrics,

        "predictions":
            predictions,
    }

    with open(
        prediction_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        f"OOF FOLD {fold} COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Held-out examples : "
        f"{len(predictions)}"
    )

    print(
        f"Correct           : "
        f"{correct}"
    )

    print(
        f"OOF accuracy      : "
        f"{oof_accuracy:.4f}"
    )

    print(
        f"Best checkpoint   : "
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        f"Best eval loss    : "
        f"{trainer.state.best_metric}"
    )

    print(
        "\nSaved predictions ->"
    )

    print(
        prediction_file
    )


if __name__ == "__main__":
    main()
