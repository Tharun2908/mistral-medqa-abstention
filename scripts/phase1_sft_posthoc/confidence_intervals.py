"""
confidence_intervals.py
-----------------------
Compute bootstrap confidence intervals for key metrics.

Bootstrap method:
  - Resample test set 1000 times with replacement
  - Compute metric each time
  - Take 2.5th and 97.5th percentile -> 95% CI

Metrics:
  - Overall accuracy (baseline vs fine-tuned)
  - Answered accuracy at threshold=0.50
  - Coverage at threshold=0.50
  - Dataset-level wrong rate at threshold=0.50
  - AUROC for error detection

Input  : baseline_results.json, finetuned_results.json
Output : confidence_intervals_results.json
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

# ── 2. Bootstrap Function ──────────────────────────────────────────────────────
def bootstrap_ci(predictions, metric_fn, n_bootstrap=1000, ci=95, seed=42):
    """
    Compute bootstrap confidence interval for a metric.

    predictions : list of prediction dicts
    metric_fn   : function that takes predictions -> scalar metric
    n_bootstrap : number of bootstrap samples
    ci          : confidence level (95 = 95% CI)
    """
    rng = np.random.RandomState(seed)
    n   = len(predictions)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.randint(0, n, size=n)
        sample  = [predictions[i] for i in indices]
        score   = metric_fn(sample)
        bootstrap_scores.append(score)

    lower = np.percentile(bootstrap_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrap_scores, 100 - (100 - ci) / 2)
    mean  = np.mean(bootstrap_scores)

    return round(mean, 4), round(lower, 4), round(upper, 4)

# ── 3. Define Metrics ──────────────────────────────────────────────────────────
def accuracy(preds):
    return sum(p["is_correct"] for p in preds) / len(preds)

def answered_accuracy_at_50(preds):
    answered = [p for p in preds if p["confidence"] >= 0.50]
    if not answered:
        return 0.0
    return sum(p["is_correct"] for p in answered) / len(answered)

def coverage_at_50(preds):
    answered = [p for p in preds if p["confidence"] >= 0.50]
    return len(answered) / len(preds)

def wrong_rate_at_50(preds):
    answered = [p for p in preds if p["confidence"] >= 0.50]
    wrong    = sum(not p["is_correct"] for p in answered)
    return wrong / len(preds)

def auroc(preds):
    """Bootstrap-friendly AUROC computation."""
    labels = [1 if p["is_correct"] else 0 for p in preds]
    scores = [p["confidence"] for p in preds]
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    tps, fps  = 0, 0
    auc       = 0.0
    prev_fpr  = 0.0
    prev_tpr  = 0.0
    for score, label in paired:
        if label == 1:
            tps += 1
        else:
            fps += 1
        fpr = fps / total_neg
        tpr = tps / total_pos
        auc += (fpr - prev_fpr) * (prev_tpr + tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr
    return auc

# ── 4. Compute CIs ─────────────────────────────────────────────────────────────
print("\nComputing bootstrap confidence intervals (1000 samples each)...")
print("This may take a minute...\n")

metrics = [
    ("Overall Accuracy",          accuracy),
    ("Answered Accuracy @0.50",   answered_accuracy_at_50),
    ("Coverage @0.50",            coverage_at_50),
    ("Wrong Rate @0.50",          wrong_rate_at_50),
    ("AUROC",                     auroc),
]

results = {}

print(f"{'='*85}")
print("95% BOOTSTRAP CONFIDENCE INTERVALS")
print(f"{'='*85}")
print(f"\n{'Metric':<28} {'Baseline':>20} {'Fine-Tuned':>20} {'Overlap?':>10}")
print(f"{'-'*85}")

for name, fn in metrics:
    print(f"  Computing {name}...", end=" ", flush=True)

    bl_mean,  bl_low,  bl_high  = bootstrap_ci(baseline_preds,  fn)
    ft_mean,  ft_low,  ft_high  = bootstrap_ci(finetuned_preds, fn)

    # Check overlap
    overlap = bl_low <= ft_high and ft_low <= bl_high
    overlap_str = "Yes " if overlap else "No "

    print(f"\r  {name:<28} "
          f"{bl_mean*100:>6.2f}% [{bl_low*100:.2f}%, {bl_high*100:.2f}%] "
          f"{ft_mean*100:>6.2f}% [{ft_low*100:.2f}%, {ft_high*100:.2f}%] "
          f"{overlap_str:>10}")

    results[name] = {
        "baseline"  : {"mean": bl_mean, "ci_low": bl_low,  "ci_high": bl_high},
        "finetuned" : {"mean": ft_mean, "ci_low": ft_low,  "ci_high": ft_high},
        "overlap"   : bool(overlap),
    }

# ── 5. Interpretation ──────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print("INTERPRETATION")
print(f"{'='*85}")

for name, data in results.items():
    bl  = data["baseline"]
    ft  = data["finetuned"]
    diff = ft["mean"] - bl["mean"]

    if data["overlap"]:
        print(f"\n  {name}:")
        print(f"    Difference: {diff*100:+.2f}% — CIs overlap, not statistically conclusive")
    else:
        print(f"\n  {name}:")
        print(f" Difference: {diff*100:+.2f}% — CIs do NOT overlap, statistically meaningful ")

# ── 6. Save Results ────────────────────────────────────────────────────────────
with open("confidence_intervals_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved confidence_intervals_results.json")
