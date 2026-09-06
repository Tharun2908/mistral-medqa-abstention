"""
eval_warmstart_dev.py
---------------------

Clean MedQA DEV evaluator for warm-start candidate selection.

Important:
    - DEV ONLY.
    - Does not touch official MedQA test.
    - Scores full-sentence completions A/B/C/D/E.
    - Same scoring rule will later be used for DPO.
    - Saves expensive per-example scores before computing summary metrics.

E = "I cannot answer confidently."
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.medqa_data import load_medqa


BASE_MODEL = "mistralai/Mistral-7B-v0.3"

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "warmstart_eval"
)

ANSWER_SET = ["A", "B", "C", "D"]

LETTERS = ["A", "B", "C", "D", "E"]

MAX_LEN = 1024

COMPLETIONS = {
    "A": " The answer is A.",
    "B": " The answer is B.",
    "C": " The answer is C.",
    "D": " The answer is D.",
    "E": " I cannot answer confidently.",
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help=(
            "PEFT adapter path or Hugging Face adapter ID."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name used for saved dev results.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

def build_prompt(question, options):

    option_lines = "\n".join(
        f"{k}: {options[k]}"
        for k in ANSWER_SET
    )

    return (
        f"Question: {question}\n\n"
        f"Options:\n"
        f"{option_lines}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model(adapter):

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model in 4-bit...")

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

    print(f"Loading adapter:\n{adapter}")

    model = PeftModel.from_pretrained(
        base,
        adapter,
    )

    model.eval()
    model.config.use_cache = True

    return model, tokenizer


# ---------------------------------------------------------------------
# Completion scoring
# ---------------------------------------------------------------------

@torch.no_grad()
def score_completion_batch(
    model,
    tokenizer,
    prompt,
):

    device = next(
        model.parameters()
    ).device

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    ).input_ids

    sequences = []
    completion_spans = []

    for letter in LETTERS:

        completion = COMPLETIONS[letter]

        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
        ).input_ids

        budget = (
            MAX_LEN
            - len(completion_ids)
        )

        if budget <= 0:
            raise RuntimeError(
                "Completion exceeds MAX_LEN."
            )

        if len(prompt_ids) > budget:
            used_prompt = prompt_ids[-budget:]
        else:
            used_prompt = prompt_ids

        ids = (
            used_prompt
            + completion_ids
        )

        sequences.append(ids)

        completion_spans.append(
            (
                len(used_prompt),
                len(ids),
            )
        )

    max_length = max(
        len(x)
        for x in sequences
    )

    pad_id = tokenizer.pad_token_id

    input_ids = torch.full(
        (
            len(sequences),
            max_length,
        ),
        pad_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (
            len(sequences),
            max_length,
        ),
        dtype=torch.long,
    )

    for i, sequence in enumerate(
        sequences
    ):

        input_ids[
            i,
            :len(sequence)
        ] = torch.tensor(
            sequence,
            dtype=torch.long,
        )

        attention_mask[
            i,
            :len(sequence)
        ] = 1

    input_ids = input_ids.to(device)

    attention_mask = (
        attention_mask.to(device)
    )

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits.float()

    score_map = {}

    for i, letter in enumerate(
        LETTERS
    ):

        start, end = (
            completion_spans[i]
        )

        prediction_logits = logits[
            i,
            start - 1:end - 1,
            :
        ]

        labels = input_ids[
            i,
            start:end
        ]

        log_probs = torch.log_softmax(
            prediction_logits,
            dim=-1,
        )

        token_log_probs = (
            log_probs.gather(
                1,
                labels.unsqueeze(1),
            )
            .squeeze(1)
        )

        # Length-normalized completion score.
        score_map[letter] = float(
            token_log_probs.mean().item()
        )

    return score_map


def score_example(
    model,
    tokenizer,
    prompt,
):

    score_map = score_completion_batch(
        model,
        tokenizer,
        prompt,
    )

    # Natural decision includes abstain E.
    decision = max(
        LETTERS,
        key=lambda x: score_map[x],
    )

    # Counterfactual answer if abstention
    # did not exist.
    would_be = max(
        ANSWER_SET,
        key=lambda x: score_map[x],
    )

    five_scores = torch.tensor(
        [
            score_map[x]
            for x in LETTERS
        ],
        dtype=torch.float32,
    )

    five_probs = torch.softmax(
        five_scores,
        dim=0,
    )

    p_abstain = float(
        five_probs[
            LETTERS.index("E")
        ].item()
    )

    answer_scores = torch.tensor(
        [
            score_map[x]
            for x in ANSWER_SET
        ],
        dtype=torch.float32,
    )

    answer_probs = torch.softmax(
        answer_scores,
        dim=0,
    )

    answer_confidence = float(
        answer_probs.max().item()
    )

    return {
        "decision": decision,
        "would_be": would_be,
        "p_abstain": p_abstain,
        "answer_confidence":
            answer_confidence,
        "scores": score_map,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_ROOT
        / f"{args.run_name}.json"
    )

    if output_file.exists():
        raise RuntimeError(
            f"Result already exists:\n"
            f"{output_file}"
        )

    print("=" * 72)
    print("CLEAN WARM-START DEV EVALUATION")
    print("=" * 72)

    print(f"Run     : {args.run_name}")
    print(f"Adapter : {args.adapter}")

    # --------------------------------------------------------------
    # DEV ONLY
    # --------------------------------------------------------------

    dev = load_medqa("dev")

    if len(dev) != 1272:
        raise RuntimeError(
            f"Expected 1272 dev examples, "
            f"got {len(dev)}"
        )

    print(
        f"\nOfficial MedQA dev examples: "
        f"{len(dev)}"
    )

    model, tokenizer = load_model(
        args.adapter
    )

    # --------------------------------------------------------------
    # Tokenization check
    # --------------------------------------------------------------

    print(
        "\nCompletion tokenization:"
    )

    for letter in LETTERS:

        ids = tokenizer(
            COMPLETIONS[letter],
            add_special_tokens=False,
        ).input_ids

        print(
            f"  {letter}: {ids}"
        )

        if len(ids) == 0:
            raise RuntimeError(
                f"Empty completion for {letter}"
            )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    rows = []

    print(
        "\nEvaluating official dev..."
    )

    for i, example in enumerate(dev):

        prompt = build_prompt(
            example["question"],
            example["options"],
        )

        scored = score_example(
            model,
            tokenizer,
            prompt,
        )

        gold = example["answer_idx"]

        decision = scored["decision"]

        would_be = scored["would_be"]

        row = {
            "dev_index": i,
            "id": example["id"],
            "gold": gold,

            "decision": decision,

            "would_be": would_be,

            "abstain":
                decision == "E",

            "answered":
                decision in ANSWER_SET,

            "decision_correct":
                decision == gold,

            "wouldbe_correct":
                would_be == gold,

            "p_abstain":
                scored["p_abstain"],

            "answer_confidence":
                scored[
                    "answer_confidence"
                ],

            "scores":
                scored["scores"],
        }

        rows.append(row)

        if (i + 1) % 100 == 0:
            print(
                f"  evaluated "
                f"{i + 1}/{len(dev)}"
            )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    n = len(rows)

    answered = [
        x for x in rows
        if x["answered"]
    ]

    n_answered = len(answered)

    coverage = (
        n_answered / n
    )

    abstain_rate = (
        1.0 - coverage
    )

    answered_accuracy = (
        sum(
            x["decision_correct"]
            for x in answered
        )
        / n_answered
        if n_answered
        else float("nan")
    )

    would_be_accuracy = (
        sum(
            x["wouldbe_correct"]
            for x in rows
        )
        / n
    )

    dataset_wrong_rate = (
        sum(
            x["answered"]
            and not x[
                "decision_correct"
            ]
            for x in rows
        )
        / n
    )

    p_e = np.array(
        [
            x["p_abstain"]
            for x in rows
        ],
        dtype=float,
    )

    wrong = np.array(
        [
            int(
                not x[
                    "wouldbe_correct"
                ]
            )
            for x in rows
        ],
        dtype=int,
    )

    finite = np.isfinite(p_e)

    dropped = int(
        (~finite).sum()
    )

    p_e_finite = p_e[finite]

    wrong_finite = wrong[finite]

    if (
        wrong_finite.sum() > 0
        and wrong_finite.sum()
        < len(wrong_finite)
    ):

        pe_auroc = roc_auc_score(
            wrong_finite,
            p_e_finite,
        )

    else:

        pe_auroc = float("nan")

    pe_correct = p_e_finite[
        wrong_finite == 0
    ]

    pe_wrong = p_e_finite[
        wrong_finite == 1
    ]

    mean_pe_correct = float(
        pe_correct.mean()
    )

    mean_pe_wrong = float(
        pe_wrong.mean()
    )

    max_pe = float(
        p_e_finite.max()
    )

    # Natural utility.
    #
    # Same utility semantics as the project's
    # abstention reward:
    #
    # correct = +1
    # abstain = +0.3
    # wrong   = -2
    #
    rewards = []

    for x in rows:

        if x["abstain"]:
            reward = 0.3

        elif x["decision_correct"]:
            reward = 1.0

        else:
            reward = -2.0

        rewards.append(reward)

    mean_utility = float(
        np.mean(rewards)
    )

    summary = {
        "protocol":
            "clean_dev_warmstart_v1",

        "split":
            "dev",

        "n_dev":
            n,

        "run_name":
            args.run_name,

        "adapter":
            args.adapter,

        "natural_operating_point": {
            "coverage":
                coverage,

            "abstain_rate":
                abstain_rate,

            "answered_accuracy":
                answered_accuracy,

            "dataset_wrong_rate":
                dataset_wrong_rate,

            "mean_utility":
                mean_utility,
        },

        "answer_behavior": {
            "would_be_accuracy":
                would_be_accuracy,
        },

        "pe_diagnostic": {
            "mean_pe_wouldbe_correct":
                mean_pe_correct,

            "mean_pe_wouldbe_wrong":
                mean_pe_wrong,

            "max_pe":
                max_pe,

            "auroc_pe_as_wrongness":
                pe_auroc,

            "n_dropped_nonfinite":
                dropped,
        },

        "rows":
            rows,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "DEV EVALUATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Would-be answer accuracy : "
        f"{would_be_accuracy:.4f}"
    )

    print(
        f"Natural coverage         : "
        f"{coverage:.4f}"
    )

    print(
        f"Natural abstain rate     : "
        f"{abstain_rate:.4f}"
    )

    print(
        f"Answered accuracy        : "
        f"{answered_accuracy:.4f}"
    )

    print(
        f"Dataset wrong rate       : "
        f"{dataset_wrong_rate:.4f}"
    )

    print(
        f"Mean utility             : "
        f"{mean_utility:.4f}"
    )

    print(
        "\nP(E) diagnostic:"
    )

    print(
        f"  mean P(E) | correct    : "
        f"{mean_pe_correct:.4f}"
    )

    print(
        f"  mean P(E) | wrong      : "
        f"{mean_pe_wrong:.4f}"
    )

    print(
        f"  max P(E)               : "
        f"{max_pe:.4f}"
    )

    print(
        f"  AUROC P(E) -> wrong    : "
        f"{pe_auroc:.4f}"
    )

    print(
        f"\nSaved ->\n{output_file}"
    )


if __name__ == "__main__":
    main()
