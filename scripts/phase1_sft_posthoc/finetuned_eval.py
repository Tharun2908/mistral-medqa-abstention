"""
finetuned_eval.py
-----------------

Clean-protocol evaluation for the fine-tuned Mistral-7B MedQA model.

Protocol:
    dev  -> development / calibration / threshold selection
    test -> final locked evaluation only

The test split requires --final-test explicitly.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------
# Make repo root importable even when this script is executed directly.
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.medqa_data import load_medqa


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_MODEL = "mistralai/Mistral-7B-v0.3"
SFT_ADAPTER = "Primeinvincible/mistral-medqa-lora-v3"

RESULT_DIR = REPO_ROOT / "results" / "clean_protocol" / "phase1_sft"

ANSWER_SET = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Evaluation split. Default is dev.",
    )

    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Required to access the locked MedQA test split.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional DEV-only smoke-test limit.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

def format_prompt(example):
    options_str = "\n".join(
        f"{k}: {example['options'][k]}"
        for k in ANSWER_SET
    )

    return (
        f"Question: {example['question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------
# Answer-token verification
# ---------------------------------------------------------------------

def get_answer_token_ids(tokenizer, sample_prompt):
    """
    Verify A/B/C/D are single-token continuations after the prompt.
    """

    answer_token_ids = {}

    base_ids = tokenizer(
        sample_prompt,
        add_special_tokens=False,
    ).input_ids

    print("\nVerifying answer option tokenization...")

    for option in ANSWER_SET:

        full_ids = tokenizer(
            sample_prompt + " " + option,
            add_special_tokens=False,
        ).input_ids

        continuation_ids = full_ids[len(base_ids):]

        print(
            f"  Option {option}: "
            f"continuation IDs = {continuation_ids}"
        )

        if len(continuation_ids) != 1:
            raise ValueError(
                f"Option {option} is not a single-token continuation: "
                f"{continuation_ids}"
            )

        answer_token_ids[option] = continuation_ids[0]

    print("Tokenization check passed.\n")

    return answer_token_ids


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_answer(
    model,
    tokenizer,
    prompt,
    answer_token_ids,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]

    option_ids = torch.tensor(
        [answer_token_ids[x] for x in ANSWER_SET],
        device=last_logits.device,
    )

    option_logits = last_logits[option_ids]

    option_probs = torch.softmax(
        option_logits.float(),
        dim=0,
    )

    best_idx = int(torch.argmax(option_probs))

    prediction = ANSWER_SET[best_idx]

    confidence = float(option_probs[best_idx])

    all_probs = {
        option: float(option_probs[i])
        for i, option in enumerate(ANSWER_SET)
    }

    return prediction, confidence, all_probs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    # --------------------------------------------------------------
    # Test-set protection
    # --------------------------------------------------------------

    if args.split == "test" and not args.final_test:
        raise RuntimeError(
            "\nTEST ACCESS DENIED.\n"
            "The MedQA test set is for final locked evaluation only.\n\n"
            "For development use:\n"
            "    --split dev\n\n"
            "Final evaluation requires:\n"
            "    --split test --final-test\n"
        )

    if args.split == "test" and args.limit is not None:
        raise RuntimeError(
            "Partial test-set evaluation is disabled. "
            "Final test evaluation must use all examples."
        )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    print("=" * 60)
    print("CLEAN MEDQA EVALUATION")
    print("=" * 60)

    print(f"Requested split : {args.split}")

    dataset = load_medqa(
        args.split,
        allow_test=args.final_test,
    )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError("--limit must be > 0")

        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

        print(f"DEV smoke-test limit: {len(dataset)}")

    print(f"Examples        : {len(dataset)}")

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print(f"Compute dtype: {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Loading SFT LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER,
    )

    model.eval()

    print("Model loaded.")

    # --------------------------------------------------------------
    # Tokenization sanity check
    # --------------------------------------------------------------

    sample_prompt = format_prompt(dataset[0])

    answer_token_ids = get_answer_token_ids(
        tokenizer,
        sample_prompt,
    )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    rows = []

    correct = 0

    for i, example in enumerate(
        tqdm(dataset, desc=f"Evaluating SFT on {args.split}")
    ):

        prompt = format_prompt(example)

        prediction, confidence, all_probs = predict_answer(
            model,
            tokenizer,
            prompt,
            answer_token_ids,
        )

        is_correct = (
            prediction == example["answer_idx"]
        )

        correct += int(is_correct)

        rows.append(
            {
                "idx": i,
                "id": example["id"],
                "prediction": prediction,
                "ground_truth": example["answer_idx"],
                "is_correct": is_correct,
                "confidence": confidence,
                "all_probs": all_probs,
            }
        )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    accuracy = correct / len(rows)

    print("\n" + "=" * 60)

    print(f"Split             : {args.split}")
    print(f"Examples          : {len(rows)}")
    print(f"Correct           : {correct}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Accuracy (%)      : {accuracy * 100:.2f}%")

    print("=" * 60)

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"{args.split}_limit{args.limit}"
        if args.limit is not None
        else args.split
    )

    output_file = (
        RESULT_DIR
        / f"sft_{suffix}_predictions.json"
    )

    output = {
        "protocol": "clean_v1",
        "split": args.split,
        "final_test": args.final_test,
        "model": BASE_MODEL,
        "adapter": SFT_ADAPTER,
        "n_examples": len(rows),
        "accuracy": accuracy,
        "correct": correct,
        "predictions": rows,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(f"\nSaved -> {output_file}")


if __name__ == "__main__":
    main()
