"""
Final paired bootstrap for MedQA TEST.

Uses ONLY previously saved per-example TEST predictions.

NO:
- model inference
- checkpoint selection
- threshold reselection
- TEST-derived operating-point tuning

Original preregistered/locked comparison and post-hoc controls
are reported separately.

Bootstrap:
    10,000 paired resamples of the 1,273 TEST examples
    percentile 95% confidence intervals
    seed = 20260906

Important:
    CIs are nominal 95% intervals.
    No multiplicity-adjusted significance claims are made.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]

N_BOOT = 10_000
SEED = 20260906

TARGETS = ["0.3", "0.4", "0.5", "0.6"]


# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------

FILES = {

    # Original locked comparison
    "original_sft":
        "results/clean_protocol/locked_test/final_comparison/"
        "original_sft.json",

    "continue_sft":
        "results/clean_protocol/locked_test/final_comparison/"
        "continue_sft.json",

    "correct_only_sft":
        "results/clean_protocol/locked_test/final_comparison/"
        "correct_only_sft.json",

    "supervised_5way":
        "results/clean_protocol/locked_test/final_comparison/"
        "supervised_5way.json",

    "dpo_2to1":
        "results/clean_protocol/locked_test/final_comparison/"
        "dpo_2to1.json",

    "grpo_common_init":
        "results/clean_protocol/locked_test/final_comparison/"
        "grpo_common_init.json",

    "grpo_arm_a":
        "results/clean_protocol/locked_test/grpo/"
        "grpo_arm_a_checkpoint_50.json",

    "grpo_arm_b":
        "results/clean_protocol/locked_test/grpo/"
        "grpo_arm_b_checkpoint_50.json",

    # POST-HOC robustness control
    "dpo_common_init":
        "results/clean_protocol/posthoc_test/"
        "dpo_common_init_ratio_2to1/"
        "dpo_common_init_ratio_2to1_test.json",
}


SCORE_TYPES = {
    "original_sft": "confidence",
    "continue_sft": "confidence",
    "correct_only_sft": "confidence",
    "supervised_5way": "margin",
    "dpo_2to1": "margin",
    "grpo_common_init": "confidence",
    "grpo_arm_a": "confidence",
    "grpo_arm_b": "margin",
    "dpo_common_init": "margin",
}


ORIGINAL_METHODS = [
    "original_sft",
    "continue_sft",
    "correct_only_sft",
    "supervised_5way",
    "dpo_2to1",
    "grpo_common_init",
    "grpo_arm_a",
    "grpo_arm_b",
]


ORIGINAL_PAIRS = [
    ("continue_sft", "original_sft"),
    ("continue_sft", "supervised_5way"),
    ("continue_sft", "correct_only_sft"),
    ("correct_only_sft", "supervised_5way"),
    ("grpo_arm_a", "grpo_common_init"),
    ("grpo_arm_a", "grpo_arm_b"),
]


POSTHOC_PAIRS = [
    ("dpo_common_init", "dpo_2to1"),
    ("dpo_common_init", "continue_sft"),
]


OUTPUT = (
    REPO_ROOT
    / "results"
    / "clean_protocol"
    / "locked_test"
    / "final_comparison"
    / "final_test_paired_bootstrap.json"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def unwrap_result(data):

    if (
        "result" in data
        and isinstance(data["result"], dict)
    ):
        return data["result"]

    if (
        "results" in data
        and isinstance(data["results"], dict)
    ):
        return data["results"]

    return data


def get_threshold(meta, target):

    ops = meta.get(
        "operating_points",
        {}
    )

    if target not in ops:
        raise RuntimeError(
            f"Missing operating point {target}"
        )

    item = ops[target]

    for key in [
        "frozen_dev_threshold",
        "threshold",
    ]:
        if key in item:
            return float(item[key])

    raise RuntimeError(
        f"No threshold found for {target}: {item.keys()}"
    )


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def safe_auc(y, score):

    if np.unique(y).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def operating_metrics(
    correct,
    answered,
):

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

    acc = (
        n_correct / n_answered
        if n_answered
        else np.nan
    )

    utility = (
        n_correct
        + 0.3 * n_abstained
        - 2.0 * n_wrong
    ) / n

    return {
        "coverage":
            n_answered / n,

        "answered_accuracy":
            acc,

        "dataset_wrong_rate":
            n_wrong / n,

        "utility":
            utility,
    }


# ---------------------------------------------------------------------
# Load + verify
# ---------------------------------------------------------------------

methods = {}

master_ids = None
master_gold = None


for name, relpath in FILES.items():

    path = REPO_ROOT / relpath

    if not path.exists():
        raise FileNotFoundError(path)

    with open(
        path,
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rows = data["rows"]
    meta = unwrap_result(data)

    if len(rows) != 1273:
        raise RuntimeError(
            f"{name}: expected 1273 rows, got {len(rows)}"
        )

    ids = [
        r["id"]
        for r in rows
    ]

    gold = [
        r["gold"]
        for r in rows
    ]

    if master_ids is None:
        master_ids = ids
        master_gold = gold
    else:
        if ids != master_ids:
            raise RuntimeError(
                f"ID alignment failed: {name}"
            )

        if gold != master_gold:
            raise RuntimeError(
                f"Gold alignment failed: {name}"
            )

    correct = np.asarray(
        [
            int(r["wouldbe_correct"])
            for r in rows
        ],
        dtype=np.int8,
    )

    confidence = np.asarray(
        [
            float(r["answer_confidence"])
            for r in rows
        ],
        dtype=float,
    )

    margin = np.asarray(
        [
            float(r["abstention_margin"])
            for r in rows
        ],
        dtype=float,
    )

    score_type = SCORE_TYPES[name]

    if score_type == "confidence":
        # Higher wrongness score = lower confidence.
        deployment_score = -confidence

    elif score_type == "margin":
        # Higher margin = more abstention-like / wrong-like.
        deployment_score = margin

    else:
        raise ValueError(score_type)

    thresholds = {
        target:
            get_threshold(
                meta,
                target,
            )
        for target in TARGETS
    }

    answered_masks = {}

    for target in TARGETS:

        threshold = (
            thresholds[target]
        )

        if score_type == "confidence":

            answered_masks[target] = (
                confidence
                >= threshold
            )

        else:

            answered_masks[target] = (
                margin
                <= threshold
            )

    methods[name] = {
        "path":
            str(path),

        "score_type":
            score_type,

        "correct":
            correct,

        "deployment_score":
            deployment_score,

        "confidence":
            confidence,

        "margin":
            margin,

        "thresholds":
            thresholds,

        "answered_masks":
            answered_masks,
    }


n = len(master_ids)

print("=" * 100)
print("FINAL TEST PAIRED BOOTSTRAP")
print("=" * 100)

print(f"Examples   : {n}")
print(f"Replicates : {N_BOOT}")
print(f"Seed       : {SEED}")
print("Alignment  : PASS")


# ---------------------------------------------------------------------
# Full-sample point estimates
# ---------------------------------------------------------------------

point = {}

for name, m in methods.items():

    correct = m["correct"]
    wrong = 1 - correct

    point[name] = {
        "wouldbe_accuracy":
            float(correct.mean()),

        "deployment_wrongness_auroc":
            safe_auc(
                wrong,
                m["deployment_score"],
            ),

        "operating_points":
            {},
    }

    for target in TARGETS:

        point[name][
            "operating_points"
        ][target] = operating_metrics(
            correct,
            m["answered_masks"][target],
        )


# ---------------------------------------------------------------------
# Allocate bootstrap arrays
# ---------------------------------------------------------------------

boot = {}

for name in methods:

    boot[name] = {
        "wouldbe_accuracy":
            np.empty(
                N_BOOT,
                dtype=float,
            ),

        "deployment_wrongness_auroc":
            np.empty(
                N_BOOT,
                dtype=float,
            ),

        "operating_points": {
            target: {
                "coverage":
                    np.empty(N_BOOT),

                "answered_accuracy":
                    np.empty(N_BOOT),

                "dataset_wrong_rate":
                    np.empty(N_BOOT),

                "utility":
                    np.empty(N_BOOT),
            }
            for target in TARGETS
        },
    }


rng = np.random.default_rng(
    SEED
)


# ---------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------

for b in range(N_BOOT):

    idx = rng.integers(
        0,
        n,
        size=n,
    )

    for name, m in methods.items():

        correct = (
            m["correct"][idx]
        )

        wrong = (
            1
            - correct
        )

        score = (
            m["deployment_score"][idx]
        )

        boot[name][
            "wouldbe_accuracy"
        ][b] = correct.mean()

        boot[name][
            "deployment_wrongness_auroc"
        ][b] = safe_auc(
            wrong,
            score,
        )

        for target in TARGETS:

            answered = (
                m[
                    "answered_masks"
                ][target][idx]
            )

            metrics = operating_metrics(
                correct,
                answered,
            )

            for metric, value in (
                metrics.items()
            ):

                boot[name][
                    "operating_points"
                ][target][metric][b] = value

    if (
        (b + 1) % 1000 == 0
        or
        b == 0
    ):

        print(
            f"  bootstrap "
            f"{b + 1}/{N_BOOT}"
        )


# ---------------------------------------------------------------------
# Method-level confidence intervals
# ---------------------------------------------------------------------

method_summary = {}

for name in methods:

    method_summary[name] = {
        "status":
            (
                "posthoc_control"
                if name == "dpo_common_init"
                else "original_locked_comparison"
            ),

        "score_type":
            methods[name]["score_type"],

        "wouldbe_accuracy": {
            "point":
                point[name][
                    "wouldbe_accuracy"
                ],

            "ci95":
                percentile_ci(
                    boot[name][
                        "wouldbe_accuracy"
                    ]
                ),
        },

        "deployment_wrongness_auroc": {
            "point":
                point[name][
                    "deployment_wrongness_auroc"
                ],

            "ci95":
                percentile_ci(
                    boot[name][
                        "deployment_wrongness_auroc"
                    ][
                        np.isfinite(
                            boot[name][
                                "deployment_wrongness_auroc"
                            ]
                        )
                    ]
                ),
        },

        "operating_points": {},
    }

    for target in TARGETS:

        method_summary[name][
            "operating_points"
        ][target] = {}

        for metric in [
            "coverage",
            "answered_accuracy",
            "dataset_wrong_rate",
            "utility",
        ]:

            vals = boot[name][
                "operating_points"
            ][target][metric]

            vals = vals[
                np.isfinite(vals)
            ]

            method_summary[name][
                "operating_points"
            ][target][metric] = {
                "point":
                    point[name][
                        "operating_points"
                    ][target][metric],

                "ci95":
                    percentile_ci(vals),
            }


# ---------------------------------------------------------------------
# Pairwise summaries
# ---------------------------------------------------------------------

def paired_summary(
    a,
    b,
):

    result = {
        "a":
            a,

        "b":
            b,

        "interpretation":
            f"{a} minus {b}",

        "wouldbe_accuracy":
            {},

        "deployment_wrongness_auroc":
            {},

        "operating_points":
            {},
    }

    # WB accuracy
    d = (
        boot[a]["wouldbe_accuracy"]
        - boot[b]["wouldbe_accuracy"]
    )

    result[
        "wouldbe_accuracy"
    ] = {
        "point_difference":
            (
                point[a]["wouldbe_accuracy"]
                - point[b]["wouldbe_accuracy"]
            ),

        "ci95":
            percentile_ci(d),

        "bootstrap_probability_positive":
            float(
                np.mean(d > 0)
            ),
    }

    # AUROC
    d = (
        boot[a]["deployment_wrongness_auroc"]
        - boot[b]["deployment_wrongness_auroc"]
    )

    d = d[
        np.isfinite(d)
    ]

    result[
        "deployment_wrongness_auroc"
    ] = {
        "point_difference":
            (
                point[a][
                    "deployment_wrongness_auroc"
                ]
                - point[b][
                    "deployment_wrongness_auroc"
                ]
            ),

        "ci95":
            percentile_ci(d),

        "bootstrap_probability_positive":
            float(
                np.mean(d > 0)
            ),
    }

    # Frozen-threshold metrics
    for target in TARGETS:

        result[
            "operating_points"
        ][target] = {}

        for metric in [
            "coverage",
            "answered_accuracy",
            "dataset_wrong_rate",
            "utility",
        ]:

            da = boot[a][
                "operating_points"
            ][target][metric]

            db = boot[b][
                "operating_points"
            ][target][metric]

            diff = da - db

            diff = diff[
                np.isfinite(diff)
            ]

            result[
                "operating_points"
            ][target][metric] = {
                "point_difference":
                    (
                        point[a][
                            "operating_points"
                        ][target][metric]
                        - point[b][
                            "operating_points"
                        ][target][metric]
                    ),

                "ci95":
                    percentile_ci(diff),

                "bootstrap_probability_positive":
                    float(
                        np.mean(
                            diff > 0
                        )
                    ),
            }

    return result


original_pairwise = {
    f"{a}_minus_{b}":
        paired_summary(a, b)
    for a, b
    in ORIGINAL_PAIRS
}


posthoc_pairwise = {
    f"{a}_minus_{b}":
        paired_summary(a, b)
    for a, b
    in POSTHOC_PAIRS
}


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

output = {
    "protocol":
        "final_test_paired_bootstrap_v1",

    "split":
        "test",

    "n_test":
        n,

    "bootstrap_replicates":
        N_BOOT,

    "bootstrap_seed":
        SEED,

    "interval":
        "percentile_95",

    "multiplicity_note":
        (
            "Intervals are nominal 95% bootstrap confidence intervals. "
            "No multiplicity-adjusted significance claims should be "
            "made across operating points."
        ),

    "test_policy_note":
        (
            "All operating-point statistics use exact DEV-frozen "
            "numeric thresholds. No TEST threshold reselection."
        ),

    "original_locked_comparison": {
        "methods":
            {
                name:
                    method_summary[name]
                for name in ORIGINAL_METHODS
            },

        "pairwise":
            original_pairwise,
    },

    "posthoc_controls": {
        "warning":
            (
                "Common-init DPO was motivated after the original "
                "locked TEST had already been opened. It is a "
                "post-hoc robustness/control follow-up and must not "
                "be presented as part of the preregistered one-look "
                "comparison."
            ),

        "methods": {
            "dpo_common_init":
                method_summary[
                    "dpo_common_init"
                ]
        },

        "pairwise":
            posthoc_pairwise,
    },
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


# ---------------------------------------------------------------------
# Concise printout
# ---------------------------------------------------------------------

print("\n" + "=" * 100)
print("ORIGINAL LOCKED TEST — METHOD CIs")
print("=" * 100)

print(
    f"{'method':22}"
    f"{'WB accuracy [95% CI]':>29}"
    f"{'deploy AUROC [95% CI]':>31}"
)

print("-" * 100)

for name in ORIGINAL_METHODS:

    s = method_summary[name]

    wb = s["wouldbe_accuracy"]
    auc = s["deployment_wrongness_auroc"]

    print(
        f"{name:22}"
        f"{wb['point']:.4f} "
        f"[{wb['ci95'][0]:.4f}, "
        f"{wb['ci95'][1]:.4f}]"
        f"{auc['point']:12.4f} "
        f"[{auc['ci95'][0]:.4f}, "
        f"{auc['ci95'][1]:.4f}]"
    )


print("\n" + "=" * 100)
print("ORIGINAL LOCKED TEST — KEY PAIRED DIFFERENCES")
print("=" * 100)

for key, r in original_pairwise.items():

    print(f"\n{key}")

    wb = r["wouldbe_accuracy"]
    auc = r["deployment_wrongness_auroc"]

    print(
        "  WB accuracy diff : "
        f"{wb['point_difference']:+.4f} "
        f"[{wb['ci95'][0]:+.4f}, "
        f"{wb['ci95'][1]:+.4f}]"
    )

    print(
        "  deploy AUC diff  : "
        f"{auc['point_difference']:+.4f} "
        f"[{auc['ci95'][0]:+.4f}, "
        f"{auc['ci95'][1]:+.4f}]"
    )

    for target in TARGETS:

        acc = r[
            "operating_points"
        ][target][
            "answered_accuracy"
        ]

        util = r[
            "operating_points"
        ][target][
            "utility"
        ]

        print(
            f"  {float(target):.0%}: "
            f"answered-acc diff="
            f"{acc['point_difference']:+.4f} "
            f"[{acc['ci95'][0]:+.4f}, "
            f"{acc['ci95'][1]:+.4f}]  "
            f"utility diff="
            f"{util['point_difference']:+.4f} "
            f"[{util['ci95'][0]:+.4f}, "
            f"{util['ci95'][1]:+.4f}]"
        )


print("\n" + "=" * 100)
print("POST-HOC DPO CONTROL")
print("=" * 100)

s = method_summary[
    "dpo_common_init"
]

print(
    "Common-init DPO WB accuracy: "
    f"{s['wouldbe_accuracy']['point']:.4f} "
    f"[{s['wouldbe_accuracy']['ci95'][0]:.4f}, "
    f"{s['wouldbe_accuracy']['ci95'][1]:.4f}]"
)

print(
    "Common-init DPO margin AUROC: "
    f"{s['deployment_wrongness_auroc']['point']:.4f} "
    f"[{s['deployment_wrongness_auroc']['ci95'][0]:.4f}, "
    f"{s['deployment_wrongness_auroc']['ci95'][1]:.4f}]"
)

for key, r in posthoc_pairwise.items():

    print(f"\n{key}")

    wb = r["wouldbe_accuracy"]
    auc = r["deployment_wrongness_auroc"]

    print(
        "  WB accuracy diff : "
        f"{wb['point_difference']:+.4f} "
        f"[{wb['ci95'][0]:+.4f}, "
        f"{wb['ci95'][1]:+.4f}]"
    )

    print(
        "  deploy AUC diff  : "
        f"{auc['point_difference']:+.4f} "
        f"[{auc['ci95'][0]:+.4f}, "
        f"{auc['ci95'][1]:+.4f}]"
    )

    for target in TARGETS:

        acc = r[
            "operating_points"
        ][target][
            "answered_accuracy"
        ]

        util = r[
            "operating_points"
        ][target][
            "utility"
        ]

        print(
            f"  {float(target):.0%}: "
            f"answered-acc diff="
            f"{acc['point_difference']:+.4f} "
            f"[{acc['ci95'][0]:+.4f}, "
            f"{acc['ci95'][1]:+.4f}]  "
            f"utility diff="
            f"{util['point_difference']:+.4f} "
            f"[{util['ci95'][0]:+.4f}, "
            f"{util['ci95'][1]:+.4f}]"
        )


print("\n" + "=" * 100)
print("FINAL TEST BOOTSTRAP COMPLETE")
print("=" * 100)

print(f"\nSaved:\n{OUTPUT}")
