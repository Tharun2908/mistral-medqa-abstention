"""
Select clean GRPO checkpoints using the preregistered DEV-only rule.

Arm A:
    wrongness = - max softmax(A-D completion scores)

Arm B:
    wrongness = score(E) - max(score(A-D))

Selection:
    highest DEV wrongness AUROC
    exact tie -> earlier checkpoint

Common init is shown as step 0 reference only.
It is NOT a GRPO checkpoint candidate.

TEST IS NEVER LOADED.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_ROOT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "warmstart_eval"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "learned_abstention"
    / "grpo"
    / "dev_checkpoint_selection.json"
)

ANSWER_LABELS = ["A", "B", "C", "D"]

CANDIDATES = [
    ("checkpoint-50", 50),
    ("checkpoint-100", 100),
    ("checkpoint-150", 150),
    ("checkpoint-200", 200),
    ("checkpoint-250", 250),
    ("final", 270),
]


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def answer_confidence(row):

    vals = [
        float(row["scores"][x])
        for x in ANSWER_LABELS
    ]

    return float(
        np.max(
            softmax(vals)
        )
    )


def abstention_margin(row):

    scores = row["scores"]

    best_answer = max(
        float(scores[x])
        for x in ANSWER_LABELS
    )

    return (
        float(scores["E"])
        - best_answer
    )


def load_eval(name):

    path = (
        EVAL_ROOT
        / f"{name}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing DEV evaluation:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rows = data["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"{name}: expected 1272 DEV rows, "
            f"got {len(rows)}"
        )

    return rows


def evaluate_rows(rows):

    correct = np.asarray(
        [
            int(row["wouldbe_correct"])
            for row in rows
        ],
        dtype=int,
    )

    wrong = 1 - correct

    confidence = np.asarray(
        [
            answer_confidence(row)
            for row in rows
        ],
        dtype=float,
    )

    margin = np.asarray(
        [
            abstention_margin(row)
            for row in rows
        ],
        dtype=float,
    )

    confidence_auc = float(
        roc_auc_score(
            wrong,
            -confidence,
        )
    )

    margin_auc = float(
        roc_auc_score(
            wrong,
            margin,
        )
    )

    # Natural five-way decision:
    # E wins iff its completion score exceeds
    # every A-D completion score.
    natural_answered = np.asarray(
        [
            float(row["scores"]["E"])
            <= max(
                float(row["scores"][x])
                for x in ANSWER_LABELS
            )
            for row in rows
        ],
        dtype=bool,
    )

    natural_coverage = float(
        natural_answered.mean()
    )

    return {
        "wouldbe_accuracy":
            float(correct.mean()),

        "confidence_wrongness_auroc":
            confidence_auc,

        "margin_wrongness_auroc":
            margin_auc,

        "natural_coverage":
            natural_coverage,

        "mean_margin_correct":
            float(
                margin[
                    correct == 1
                ].mean()
            ),

        "mean_margin_wrong":
            float(
                margin[
                    correct == 0
                ].mean()
            ),
    }


def arm_records(arm):

    records = []

    for label, step in CANDIDATES:

        if label == "final":
            name = (
                f"grpo_arm_{arm.lower()}_final"
            )
        else:
            name = (
                f"grpo_arm_{arm.lower()}_"
                f"checkpoint_{step}"
            )

        rows = load_eval(name)

        metrics = evaluate_rows(
            rows
        )

        records.append(
            {
                "label": label,
                "step": step,
                "eval_name": name,
                **metrics,
            }
        )

    return records


def select(records, arm):

    metric = (
        "confidence_wrongness_auroc"
        if arm == "A"
        else "margin_wrongness_auroc"
    )

    # Highest AUROC.
    # Exact tie -> lower/earlier step.
    ranked = sorted(
        records,
        key=lambda x: (
            -x[metric],
            x["step"],
        ),
    )

    return ranked[0], metric


def print_table(
    title,
    records,
    selection_metric,
    selected,
):

    print("\n" + "=" * 105)
    print(title)
    print("=" * 105)

    print(
        f"{'checkpoint':16}"
        f"{'WB acc':>10}"
        f"{'conf AUC':>12}"
        f"{'margin AUC':>13}"
        f"{'nat cov':>11}"
        f"{'SELECT':>10}"
    )

    print("-" * 105)

    for r in records:

        marker = (
            "<-- BEST"
            if r["label"]
            == selected["label"]
            else ""
        )

        print(
            f"{r['label']:16}"
            f"{r['wouldbe_accuracy']:10.4f}"
            f"{r['confidence_wrongness_auroc']:12.4f}"
            f"{r['margin_wrongness_auroc']:13.4f}"
            f"{r['natural_coverage']:11.4f}"
            f"{marker:>10}"
        )

    print(
        f"\nSelection metric: "
        f"{selection_metric}"
    )

    print(
        f"Selected: "
        f"{selected['label']} "
        f"(step {selected['step']})"
    )

    print(
        f"Selected AUROC: "
        f"{selected[selection_metric]:.6f}"
    )


def main():

    print("=" * 105)
    print("CLEAN GRPO — DEV CHECKPOINT SELECTION")
    print("=" * 105)

    print(
        "\nProtocol: DEV ONLY"
    )

    print(
        "Official TEST touched: NO"
    )

    # ----------------------------------------------------------
    # Common init reference
    # ----------------------------------------------------------

    common_rows = load_eval(
        "grpo_common_init"
    )

    common = evaluate_rows(
        common_rows
    )

    print("\nCOMMON INIT — STEP 0 REFERENCE")

    print(
        f"WB accuracy             : "
        f"{common['wouldbe_accuracy']:.4f}"
    )

    print(
        f"Confidence wrong AUROC  : "
        f"{common['confidence_wrongness_auroc']:.4f}"
    )

    print(
        f"Margin wrong AUROC      : "
        f"{common['margin_wrongness_auroc']:.4f}"
    )

    print(
        f"Natural coverage        : "
        f"{common['natural_coverage']:.4f}"
    )

    # ----------------------------------------------------------
    # Arm A
    # ----------------------------------------------------------

    arm_a = arm_records("A")

    selected_a, metric_a = select(
        arm_a,
        "A",
    )

    print_table(
        "ARM A — ANSWER-ONLY GRPO",
        arm_a,
        metric_a,
        selected_a,
    )

    # ----------------------------------------------------------
    # Arm B
    # ----------------------------------------------------------

    arm_b = arm_records("B")

    selected_b, metric_b = select(
        arm_b,
        "B",
    )

    print_table(
        "ARM B — ABSTENTION GRPO",
        arm_b,
        metric_b,
        selected_b,
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    result = {
        "protocol":
            "clean_grpo_dev_checkpoint_selection_v1",

        "split":
            "dev",

        "n_dev":
            1272,

        "test_used":
            False,

        "common_init_reference":
            common,

        "arm_a": {
            "selection_rule":
                (
                    "highest DEV AUROC of "
                    "-A-D confidence; "
                    "exact tie -> earlier checkpoint"
                ),

            "selection_metric":
                metric_a,

            "trajectory":
                arm_a,

            "selected":
                selected_a,
        },

        "arm_b": {
            "selection_rule":
                (
                    "highest DEV AUROC of "
                    "E margin; "
                    "exact tie -> earlier checkpoint"
                ),

            "selection_metric":
                metric_b,

            "trajectory":
                arm_b,

            "selected":
                selected_b,
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print("\n" + "=" * 105)

    print(
        "SELECTION COMPLETE — TEST STILL LOCKED"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
