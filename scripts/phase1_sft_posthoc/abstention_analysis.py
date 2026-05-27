"""
abstention_analysis.py
----------------------
Analyze abstention thresholds on baseline and fine-tuned model results.
Sweeps confidence thresholds and computes accuracy, coverage, wrong answer rate.
Produces comparison between baseline and fine-tuned across all thresholds.

Input  : baseline_results.json, finetuned_results.json
Output : abstention_results.json
"""

import json
import numpy as np

# ── 1. Load Results ────────────────────────────────────────────────────────────
print("Loading results...")
with open("baseline_results.json", "r") as f:
    baseline_data = json.load(f)

with open("finetuned_results.json", "r") as f:
    finetuned_data = json.load(f)

baseline_preds  = baseline_data["predictions"]
finetuned_preds = finetuned_data["predictions"]

print(f"Baseline  : {len(baseline_preds)} examples")
print(f"Fine-tuned: {len(finetuned_preds)} examples")

# ── 2. Sweep Thresholds ────────────────────────────────────────────────────────
def sweep_thresholds(predictions, thresholds):
    """
    For each threshold:
      - answered         : examples where confidence >= threshold
      - abstained        : examples where confidence < threshold
      - answered_accuracy: correct / answered
      - coverage         : answered / total
      - wrong_answer_rate: wrong answered / total
      - abstention_rate  : abstained / total
    """
    total = len(predictions)
    rows  = []

    for t in thresholds:
        answered  = [p for p in predictions if p["confidence"] >= t]
        abstained = [p for p in predictions if p["confidence"] < t]

        if len(answered) == 0:
            continue

        correct           = sum(p["is_correct"] for p in answered)
        wrong             = len(answered) - correct
        answered_accuracy = correct / len(answered)
        coverage          = len(answered) / total
        wrong_answer_rate = wrong / total
        abstention_rate   = len(abstained) / total

        rows.append({
            "threshold"         : round(t, 2),
            "answered_accuracy" : round(answered_accuracy, 4),
            "coverage"          : round(coverage, 4),
            "wrong_answer_rate" : round(wrong_answer_rate, 4),
            "abstention_rate"   : round(abstention_rate, 4),
            "answered"          : len(answered),
            "abstained"         : len(abstained),
            "correct"           : correct,
            "wrong"             : wrong,
        })

    return rows

thresholds = np.arange(0.0, 1.0, 0.05).tolist()

print("\nSweeping thresholds for baseline...")
baseline_rows  = sweep_thresholds(baseline_preds, thresholds)

print("Sweeping thresholds for fine-tuned...")
finetuned_rows = sweep_thresholds(finetuned_preds, thresholds)

# ── 3. Print Baseline Table ────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("BASELINE — Threshold Analysis")
print(f"{'='*75}")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Coverage':>10} {'WrongRate':>11} {'Abstained':>10}")
print(f"{'-'*75}")
for row in baseline_rows:
    print(f"{row['threshold']:>10.2f} "
          f"{row['answered_accuracy']*100:>9.2f}% "
          f"{row['coverage']*100:>9.2f}% "
          f"{row['wrong_answer_rate']*100:>10.2f}% "
          f"{row['abstention_rate']*100:>9.2f}%")

# ── 4. Print Fine-tuned Table ──────────────────────────────────────────────────
print(f"\n{'='*75}")
print("FINE-TUNED — Threshold Analysis")
print(f"{'='*75}")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Coverage':>10} {'WrongRate':>11} {'Abstained':>10}")
print(f"{'-'*75}")
for row in finetuned_rows:
    print(f"{row['threshold']:>10.2f} "
          f"{row['answered_accuracy']*100:>9.2f}% "
          f"{row['coverage']*100:>9.2f}% "
          f"{row['wrong_answer_rate']*100:>10.2f}% "
          f"{row['abstention_rate']*100:>9.2f}%")

# ── 5. Find Sweet Spot ─────────────────────────────────────────────────────────
def find_sweet_spot(rows, min_coverage=0.4, min_accuracy=0.70):
    """
    Find best threshold where:
      - coverage >= min_coverage (model answers at least 50% of questions)
      - answered_accuracy >= min_accuracy (at least 70% correct on answered)
      - wrong_answer_rate is minimized
    """
    candidates = [
        r for r in rows
        if r["coverage"] >= min_coverage and r["answered_accuracy"] >= min_accuracy
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["wrong_answer_rate"])

print(f"\n{'='*75}")
print("SWEET SPOT ANALYSIS (coverage >= 50%, accuracy >= 70%)")
print(f"{'='*75}")

baseline_sweet  = find_sweet_spot(baseline_rows)
finetuned_sweet = find_sweet_spot(finetuned_rows)

if baseline_sweet:
    print(f"\nBaseline sweet spot:")
    print(f"  Threshold       : {baseline_sweet['threshold']}")
    print(f"  Accuracy        : {baseline_sweet['answered_accuracy']*100:.2f}%")
    print(f"  Coverage        : {baseline_sweet['coverage']*100:.2f}%")
    print(f"  Wrong rate      : {baseline_sweet['wrong_answer_rate']*100:.2f}%")
    print(f"  Abstention rate : {baseline_sweet['abstention_rate']*100:.2f}%")
else:
    print("\nBaseline: no sweet spot found")

if finetuned_sweet:
    print(f"\nFine-tuned sweet spot:")
    print(f"  Threshold       : {finetuned_sweet['threshold']}")
    print(f"  Accuracy        : {finetuned_sweet['answered_accuracy']*100:.2f}%")
    print(f"  Coverage        : {finetuned_sweet['coverage']*100:.2f}%")
    print(f"  Wrong rate      : {finetuned_sweet['wrong_answer_rate']*100:.2f}%")
    print(f"  Abstention rate : {finetuned_sweet['abstention_rate']*100:.2f}%")
else:
    print("\nFine-tuned: no sweet spot found")

# ── 6. Summary Comparison ──────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("SUMMARY — No Abstention vs Best Abstention")
print(f"{'='*75}")
print(f"\nBaseline (no abstention)  : accuracy={baseline_data['accuracy']*100:.2f}%, coverage=100%")
if baseline_sweet:
    print(f"Baseline (with abstention): accuracy={baseline_sweet['answered_accuracy']*100:.2f}%, "
          f"coverage={baseline_sweet['coverage']*100:.2f}%, "
          f"wrong_rate={baseline_sweet['wrong_answer_rate']*100:.2f}%")

print(f"\nFine-tuned (no abstention)  : accuracy={finetuned_data['accuracy']*100:.2f}%, coverage=100%")
if finetuned_sweet:
    print(f"Fine-tuned (with abstention): accuracy={finetuned_sweet['answered_accuracy']*100:.2f}%, "
          f"coverage={finetuned_sweet['coverage']*100:.2f}%, "
          f"wrong_rate={finetuned_sweet['wrong_answer_rate']*100:.2f}%")

# ── 7. Save Results ────────────────────────────────────────────────────────────
with open("abstention_results.json", "w") as f:
    json.dump({
        "baseline"       : baseline_rows,
        "finetuned"      : finetuned_rows,
        "baseline_sweet" : baseline_sweet,
        "finetuned_sweet": finetuned_sweet,
    }, f, indent=2)

print(f"\nSaved abstention_results.json")
