"""
check_seed_stability_dev.py

DEV-only stability diagnostic for the additional optimization seed.

Compares:
    DPO 2:1       seed 42 vs seed 43
    Supervised 5-way seed 42 vs seed 43

Final abstention score:
    margin = score(E) - max(score(A), score(B), score(C), score(D))

Interpretation:
    larger margin -> stronger abstention preference / more likely wrong
    answer when margin <= threshold

Reports:
    - would-be answer accuracy
    - natural coverage (margin <= 0)
    - natural answered accuracy
    - margin AUROC for detecting wrong answers
    - matched-coverage answered accuracy at 30/40/50/60%

This is a DEV diagnostic only. Official test is untouched.
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
    / "seed_stability_margin_dev.json"
)

RUNS = {
    "dpo_seed42":
        EVAL_ROOT / "dpo_ratio_2to1.json",

    "dpo_seed43":
        EVAL_ROOT / "dpo_ratio_2to1_seed43.json",

    "supervised_seed42":
        EVAL_ROOT / "supervised_5way.json",

    "supervised_seed43":
        EVAL_ROOT / "supervised_5way_seed43.json",
}

TARGET_COVERAGES = [
    0.30,
    0.40,
    0.50,
    0.60,
]

ANSWER_LABELS = [
    "A",
    "B",
    "C",
    "D",
]


def load_rows(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data.get("split") != "dev":
        raise RuntimeError(
            f"{path}: expected split=dev, "
            f"got {data.get('split')}"
        )

    rows = data["rows"]

    if len(rows) != 1272:
        raise RuntimeError(
            f"{path}: expected 1272 rows, "
            f"got {len(rows)}"
        )

    return rows


def get_margin(row):

    scores = row["scores"]

    best_answer = max(
        float(scores[label])
        for label in ANSWER_LABELS
    )

    e_score = float(
        scores["E"]
    )

    return (
        e_score
        - best_answer
    )


def get_correct(row):

    return int(
        bool(
            row["wouldbe_correct"]
        )
    )


def select_margin_threshold(
    margins,
    target_coverage,
):

    n = len(margins)

    k = int(
        round(
            target_coverage * n
        )
    )

    k = max(
        1,
        min(k, n),
    )

    # Lower margin = stronger tendency to answer.
    ranked = np.sort(
        margins
    )

    return float(
        ranked[k - 1]
    )


def metrics_at_threshold(
    correct,
    margins,
    threshold,
):

    answered = (
        margins <= threshold
    )

    n = len(correct)

    n_answered = int(
        answered.sum()
    )

    n_correct = int(
        correct[answered].sum()
    )

    n_wrong = (
        n_answered
        - n_correct
    )

    n_abstained = (
        n
        - n_answered
    )

    coverage = (
        n_answered / n
    )

    answered_accuracy = (
        n_correct / n_answered
        if n_answered > 0
        else float("nan")
    )

    wrong_rate = (
        n_wrong / n
    )

    utility = (
        (
            n_correct * 1.0
            + n_abstained * 0.3
            + n_wrong * -2.0
        )
        / n
    )

    return {
        "threshold":
            float(threshold),

        "coverage":
            float(coverage),

        "answered_accuracy":
            float(answered_accuracy),

        "dataset_wrong_rate":
            float(wrong_rate),

        "utility":
            float(utility),
    }


def main():

    print("=" * 88)
    print("DEV SEED STABILITY — FINAL ABSTENTION MARGIN")
    print("=" * 88)

    rows_by_run = {
        name: load_rows(path)
        for name, path in RUNS.items()
    }

    # ----------------------------------------------------------
    # Verify all evaluations refer to exactly the same questions.
    # ----------------------------------------------------------

    reference_ids = [
        row["id"]
        for row in rows_by_run["dpo_seed42"]
    ]

    for name, rows in rows_by_run.items():

        ids = [
            row["id"]
            for row in rows
        ]

        if ids != reference_ids:
            raise RuntimeError(
                f"Row alignment failed for {name}"
            )

    print(
        "\nAlignment check: PASS using id"
    )

    results = {}

    # ----------------------------------------------------------
    # Build diagnostics.
    # ----------------------------------------------------------

    for name, rows in rows_by_run.items():

        correct = np.asarray(
            [
                get_correct(row)
                for row in rows
            ],
            dtype=int,
        )

        margins = np.asarray(
            [
                get_margin(row)
                for row in rows
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(margins)
        ):
            raise RuntimeError(
                f"{name}: non-finite margins"
            )

        wrong = (
            1 - correct
        )

        margin_auroc = float(
            roc_auc_score(
                wrong,
                margins,
            )
        )

        # Natural policy:
        # E wins exactly when margin > 0.
        natural_answered = (
            margins <= 0.0
        )

        natural_n = int(
            natural_answered.sum()
        )

        natural_correct = int(
            correct[
                natural_answered
            ].sum()
        )

        natural_coverage = (
            natural_n / len(rows)
        )

        natural_answered_accuracy = (
            natural_correct / natural_n
            if natural_n > 0
            else float("nan")
        )

        matched = {}

        for target in TARGET_COVERAGES:

            threshold = (
                select_margin_threshold(
                    margins,
                    target,
                )
            )

            matched[
                str(target)
            ] = metrics_at_threshold(
                correct,
                margins,
                threshold,
            )

        results[name] = {
            "would_be_accuracy":
                float(correct.mean()),

            "margin_auroc_wrongness":
                margin_auroc,

            "natural_coverage":
                float(natural_coverage),

            "natural_answered_accuracy":
                float(
                    natural_answered_accuracy
                ),

            "mean_margin_correct":
                float(
                    margins[
                        correct == 1
                    ].mean()
                ),

            "mean_margin_wrong":
                float(
                    margins[
                        correct == 0
                    ].mean()
                ),

            "matched_coverage":
                matched,
        }

    # ----------------------------------------------------------
    # Summary table.
    # ----------------------------------------------------------

    print("\n" + "=" * 88)
    print("SEED SUMMARY")
    print("=" * 88)

    print(
        f"{'run':22}"
        f"{'wb_acc':>10}"
        f"{'nat_cov':>10}"
        f"{'nat_acc':>10}"
        f"{'margin_auc':>12}"
    )

    print("-" * 64)

    for name, r in results.items():

        print(
            f"{name:22}"
            f"{r['would_be_accuracy']:10.4f}"
            f"{r['natural_coverage']:10.4f}"
            f"{r['natural_answered_accuracy']:10.4f}"
            f"{r['margin_auroc_wrongness']:12.4f}"
        )

    # ----------------------------------------------------------
    # Matched-coverage tables.
    # ----------------------------------------------------------

    print("\n" + "=" * 88)
    print("MATCHED-COVERAGE ANSWERED ACCURACY")
    print("=" * 88)

    for target in TARGET_COVERAGES:

        print(
            f"\nTarget coverage: "
            f"{target:.0%}"
        )

        print(
            f"{'run':22}"
            f"{'threshold':>12}"
            f"{'coverage':>11}"
            f"{'ans_acc':>11}"
            f"{'wrong':>11}"
            f"{'utility':>11}"
        )

        print("-" * 78)

        for name, r in results.items():

            m = r[
                "matched_coverage"
            ][str(target)]

            print(
                f"{name:22}"
                f"{m['threshold']:12.6f}"
                f"{m['coverage']:11.4f}"
                f"{m['answered_accuracy']:11.4f}"
                f"{m['dataset_wrong_rate']:11.4f}"
                f"{m['utility']:11.4f}"
            )

    # ----------------------------------------------------------
    # Explicit seed deltas.
    # ----------------------------------------------------------

    print("\n" + "=" * 88)
    print("SEED 43 - SEED 42 DELTAS")
    print("=" * 88)

    pairs = [
        (
            "DPO",
            "dpo_seed42",
            "dpo_seed43",
        ),
        (
            "Supervised",
            "supervised_seed42",
            "supervised_seed43",
        ),
    ]

    seed_deltas = {}

    for label, seed42, seed43 in pairs:

        r42 = results[seed42]
        r43 = results[seed43]

        print(f"\n{label}")

        print(
            "  would-be accuracy: "
            f"{r43['would_be_accuracy'] - r42['would_be_accuracy']:+.4f}"
        )

        print(
            "  natural coverage:  "
            f"{r43['natural_coverage'] - r42['natural_coverage']:+.4f}"
        )

        print(
            "  margin AUROC:       "
            f"{r43['margin_auroc_wrongness'] - r42['margin_auroc_wrongness']:+.4f}"
        )

        coverage_deltas = {}

        for target in TARGET_COVERAGES:

            key = str(target)

            delta = (
                r43[
                    "matched_coverage"
                ][key][
                    "answered_accuracy"
                ]
                -
                r42[
                    "matched_coverage"
                ][key][
                    "answered_accuracy"
                ]
            )

            coverage_deltas[key] = (
                float(delta)
            )

            print(
                f"  answered acc @ "
                f"{target:.0%}: "
                f"{delta:+.4f}"
            )

        seed_deltas[label] = {
            "would_be_accuracy_delta":
                float(
                    r43["would_be_accuracy"]
                    - r42["would_be_accuracy"]
                ),

            "natural_coverage_delta":
                float(
                    r43["natural_coverage"]
                    - r42["natural_coverage"]
                ),

            "margin_auroc_delta":
                float(
                    r43[
                        "margin_auroc_wrongness"
                    ]
                    -
                    r42[
                        "margin_auroc_wrongness"
                    ]
                ),

            "matched_answered_accuracy_deltas":
                coverage_deltas,
        }

    output = {
        "protocol":
            "clean_dev_seed_stability_margin_v1",

        "split":
            "dev",

        "n_dev":
            1272,

        "score_definition":
            (
                "margin = score(E) - "
                "max(score(A),score(B),score(C),score(D))"
            ),

        "auroc_target":
            "would_be_wrong",

        "test_used":
            False,

        "runs":
            results,

        "seed43_minus_seed42":
            seed_deltas,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(OUTPUT_FILE)

    print(
        "\nSEED STABILITY CHECK: PASS"
    )


if __name__ == "__main__":
    main()
