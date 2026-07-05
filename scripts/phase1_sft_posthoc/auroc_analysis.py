"""
auroc_analysis.py
-----------------
Compute AUROC for error detection using confidence scores.

Question: Can confidence scores predict whether the model is correct or wrong?

Setup:
  - Label each prediction: 1 = correct, 0 = wrong
  - Use confidence score as the classifier score
  - Compute AUROC — how well confidence separates correct from wrong

AUROC = 1.0 -> perfect error detector
AUROC = 0.5 -> no better than random
AUROC > 0.7 -> useful signal for abstention

Input  : baseline_results.json, finetuned_results.json
Output : auroc_results.json
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

# ── 2. Compute AUROC Manually ──────────────────────────────────────────────────
def compute_auroc(predictions):
    """
    Compute AUROC for error detection.

    Labels : 1 = correct, 0 = wrong
    Scores : confidence (higher = more likely correct)

    AUROC = probability that a randomly chosen correct prediction
            has higher confidence than a randomly chosen wrong prediction.

    Computed via trapezoidal rule on ROC curve.
    """
    labels  = [1 if p["is_correct"] else 0 for p in predictions]
    scores  = [p["confidence"] for p in predictions]

    # Sort by score descending
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    # Compute ROC curve points
    total_pos = sum(labels)           # correct predictions
    total_neg = len(labels) - total_pos  # wrong predictions

    tps, fps = 0, 0
    roc_points = [(0.0, 0.0)]

    for score, label in paired:
        if label == 1:
            tps += 1
        else:
            fps += 1
        tpr = tps / total_pos  # true positive rate  = recall
        fpr = fps / total_neg  # false positive rate
        roc_points.append((fpr, tpr))

    # Trapezoidal rule for area under curve
    auroc = 0.0
    for i in range(1, len(roc_points)):
        x1, y1 = roc_points[i - 1]
        x2, y2 = roc_points[i]
        auroc += (x2 - x1) * (y1 + y2) / 2

    return round(auroc, 4), roc_points

# ── 3. Compute for Both Models ─────────────────────────────────────────────────
print("\nComputing AUROC...")
baseline_auroc,  baseline_roc  = compute_auroc(baseline_preds)
finetuned_auroc, finetuned_roc = compute_auroc(finetuned_preds)

# ── 4. Confidence Distribution Analysis ───────────────────────────────────────
def confidence_stats(predictions):
    """
    Compute confidence statistics split by correct vs wrong predictions.
    """
    correct = [p["confidence"] for p in predictions if p["is_correct"]]
    wrong   = [p["confidence"] for p in predictions if not p["is_correct"]]

    return {
        "correct_mean" : round(np.mean(correct), 4),
        "correct_std"  : round(np.std(correct), 4),
        "wrong_mean"   : round(np.mean(wrong), 4),
        "wrong_std"    : round(np.std(wrong), 4),
        "gap"          : round(np.mean(correct) - np.mean(wrong), 4),
        "n_correct"    : len(correct),
        "n_wrong"      : len(wrong),
    }

baseline_stats  = confidence_stats(baseline_preds)
finetuned_stats = confidence_stats(finetuned_preds)

# ── 5. Print Results ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("AUROC FOR ERROR DETECTION")
print(f"{'='*70}")
print(f"\n  {'Model':<20} {'AUROC':>8} {'Interpretation':>20}")
print(f"  {'-'*55}")

def interpret_auroc(auroc):
    if auroc >= 0.90: return "Excellent"
    if auroc >= 0.80: return "Good"
    if auroc >= 0.70: return "Useful"
    if auroc >= 0.60: return "Weak"
    return "Poor (near random)"

print(f"  {'Baseline':<20} {baseline_auroc:>8.4f} {interpret_auroc(baseline_auroc):>20}")
print(f"  {'Fine-tuned':<20} {finetuned_auroc:>8.4f} {interpret_auroc(finetuned_auroc):>20}")

auroc_diff = finetuned_auroc - baseline_auroc
if auroc_diff > 0:
    print(f"\n Fine-tuning improved AUROC by {auroc_diff:.4f} points")
else:
    print(f"\n Fine-tuning did not improve AUROC ({auroc_diff:+.4f} points)")

print(f"\n{'='*70}")
print("CONFIDENCE GAP ANALYSIS")
print("(Correct predictions should have higher confidence than wrong ones)")
print(f"{'='*70}")
print(f"\n  {'Model':<20} {'Correct Conf':>14} {'Wrong Conf':>12} {'Gap':>8}")
print(f"  {'-'*60}")
print(f"  {'Baseline':<20} "
      f"{baseline_stats['correct_mean']*100:>12.2f}% "
      f"{baseline_stats['wrong_mean']*100:>11.2f}% "
      f"{baseline_stats['gap']*100:>+7.2f}%")
print(f"  {'Fine-tuned':<20} "
      f"{finetuned_stats['correct_mean']*100:>12.2f}% "
      f"{finetuned_stats['wrong_mean']*100:>11.2f}% "
      f"{finetuned_stats['gap']*100:>+7.2f}%")

print(f"\n  Baseline  : correct predictions avg {baseline_stats['correct_mean']*100:.2f}% confident, "
      f"wrong predictions avg {baseline_stats['wrong_mean']*100:.2f}% confident")
print(f"  Fine-tuned: correct predictions avg {finetuned_stats['correct_mean']*100:.2f}% confident, "
      f"wrong predictions avg {finetuned_stats['wrong_mean']*100:.2f}% confident")

# ── 6. Save Results ────────────────────────────────────────────────────────────
# Sample ROC points (every 50th) to keep JSON small
baseline_roc_sampled  = baseline_roc[::50]
finetuned_roc_sampled = finetuned_roc[::50]

with open("auroc_results.json", "w") as f:
    json.dump({
        "baseline": {
            "auroc"           : baseline_auroc,
            "interpretation"  : interpret_auroc(baseline_auroc),
            "confidence_stats": baseline_stats,
            "roc_curve"       : baseline_roc_sampled,
        },
        "finetuned": {
            "auroc"           : finetuned_auroc,
            "interpretation"  : interpret_auroc(finetuned_auroc),
            "confidence_stats": finetuned_stats,
            "roc_curve"       : finetuned_roc_sampled,
        },
        "auroc_diff": auroc_diff,
    }, f, indent=2)

print(f"\nSaved auroc_results.json")