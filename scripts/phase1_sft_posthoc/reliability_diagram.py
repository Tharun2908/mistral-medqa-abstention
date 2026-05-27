"""
reliability_diagram.py
----------------------
Compute and visualize reliability diagrams for baseline and fine-tuned models.
Measures confidence calibration — how well model confidence aligns with accuracy.

ECE (Expected Calibration Error):
  Weighted average gap between confidence and accuracy across bins.
  ECE = 0   → perfectly calibrated
  ECE high  → poorly calibrated (overconfident or underconfident)

A well-calibrated model follows y=x diagonal:
  "When model says 70% confident → correct ~70% of the time"

Input  : baseline_results.json, finetuned_results.json
Output : reliability_results.json
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

# ── 2. Compute Reliability Data ────────────────────────────────────────────────
def compute_reliability(predictions, n_bins=10):
    """
    Bin predictions by confidence, compute accuracy per bin.

    Fix: last bin uses inclusive upper bound (<=) to capture confidence == 1.0.
    All other bins use exclusive upper bound (<) to avoid double-counting.

    Returns:
      - bins      : list of bin data
      - ece       : Expected Calibration Error
      - mce       : Maximum Calibration Error (worst bin gap)
      - avg_conf  : average confidence across all predictions
      - avg_acc   : average accuracy across all predictions
      - total     : total number of predictions
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins  = []
    total = len(predictions)

    for i in range(n_bins):
        low  = bin_edges[i]
        high = bin_edges[i + 1]

        # Last bin inclusive to capture confidence == 1.0
        if i == n_bins - 1:
            in_bin = [p for p in predictions if low <= p["confidence"] <= high]
        else:
            in_bin = [p for p in predictions if low <= p["confidence"] < high]

        if len(in_bin) == 0:
            bins.append({
                "bin_low"       : round(low, 2),
                "bin_high"      : round(high, 2),
                "bin_mid"       : round((low + high) / 2, 2),
                "avg_confidence": None,
                "accuracy"      : None,
                "count"         : 0,
                "gap"           : None,
                "direction"     : None,
            })
            continue

        avg_conf = sum(p["confidence"] for p in in_bin) / len(in_bin)
        accuracy = sum(p["is_correct"] for p in in_bin) / len(in_bin)
        gap      = abs(avg_conf - accuracy)
        direction = "overconfident" if avg_conf > accuracy else "underconfident"

        bins.append({
            "bin_low"       : round(low, 2),
            "bin_high"      : round(high, 2),
            "bin_mid"       : round((low + high) / 2, 2),
            "avg_confidence": round(avg_conf, 4),
            "accuracy"      : round(accuracy, 4),
            "count"         : len(in_bin),
            "gap"           : round(gap, 4),
            "direction"     : direction,
        })

    # ECE = weighted average of gaps across bins
    ece = sum(
        (b["count"] / total) * b["gap"]
        for b in bins if b["count"] > 0
    )

    # MCE = worst single bin gap
    mce = max(
        (b["gap"] for b in bins if b["count"] > 0),
        default=0
    )

    avg_conf = sum(p["confidence"] for p in predictions) / total
    avg_acc  = sum(p["is_correct"] for p in predictions) / total

    return bins, round(ece, 4), round(mce, 4), round(avg_conf, 4), round(avg_acc, 4), total

baseline_bins, baseline_ece, baseline_mce, baseline_avg_conf, baseline_avg_acc, baseline_total = \
    compute_reliability(baseline_preds)

finetuned_bins, finetuned_ece, finetuned_mce, finetuned_avg_conf, finetuned_avg_acc, finetuned_total = \
    compute_reliability(finetuned_preds)

# ── 3. Print Reliability Tables ────────────────────────────────────────────────
def print_reliability_table(bins, ece, mce, avg_conf, avg_acc, total, title):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"  Total examples                   : {total}")
    print(f"  ECE (Expected Calibration Error) : {ece:.4f}")
    print(f"  MCE (Maximum Calibration Error)  : {mce:.4f}")
    print(f"  Average Confidence               : {avg_conf*100:.2f}%")
    print(f"  Average Accuracy                 : {avg_acc*100:.2f}%")
    print(f"\n  {'Bin Range':>12} {'Avg Conf':>10} {'Accuracy':>10} "
          f"{'Gap':>8} {'Count':>7} {'Direction':>15}")
    print(f"  {'-'*70}")
    for b in bins:
        if b["count"] == 0:
            continue
        print(f"  {b['bin_low']:.2f}-{b['bin_high']:.2f}    "
              f"{b['avg_confidence']*100:>8.2f}% "
              f"{b['accuracy']*100:>9.2f}% "
              f"{b['gap']*100:>7.2f}% "
              f"{b['count']:>7} "
              f"  {b['direction']:>15}")

print_reliability_table(
    baseline_bins, baseline_ece, baseline_mce,
    baseline_avg_conf, baseline_avg_acc, baseline_total,
    "BASELINE — Reliability Diagram"
)
print_reliability_table(
    finetuned_bins, finetuned_ece, finetuned_mce,
    finetuned_avg_conf, finetuned_avg_acc, finetuned_total,
    "FINE-TUNED — Reliability Diagram"
)

# ── 4. ECE Comparison ──────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("ECE COMPARISON")
print(f"{'='*80}")
print(f"\n  {'Model':<20} {'ECE':>8} {'MCE':>8} {'Avg Conf':>10} {'Avg Acc':>10}")
print(f"  {'-'*60}")
print(f"  {'Baseline':<20} {baseline_ece:>8.4f} {baseline_mce:>8.4f} "
      f"{baseline_avg_conf*100:>9.2f}% {baseline_avg_acc*100:>9.2f}%")
print(f"  {'Fine-tuned':<20} {finetuned_ece:>8.4f} {finetuned_mce:>8.4f} "
      f"{finetuned_avg_conf*100:>9.2f}% {finetuned_avg_acc*100:>9.2f}%")

ece_diff = baseline_ece - finetuned_ece
if ece_diff > 0:
    print(f"\n  ✅ Fine-tuning reduced ECE by {ece_diff:.4f} points "
          f"({baseline_ece:.4f} → {finetuned_ece:.4f})")
else:
    print(f"\n  ⚠️  Fine-tuning increased ECE by {abs(ece_diff):.4f} points "
          f"({baseline_ece:.4f} → {finetuned_ece:.4f})")

# ── 5. ASCII Reliability Diagram ───────────────────────────────────────────────
def ascii_reliability_diagram(bins, title):
    """
    ASCII bar chart — bar height = accuracy, marker = avg confidence.
    Perfect calibration = bar top matches confidence marker.
    """
    print(f"\n{title}")
    print("  (█ = accuracy, │ = avg confidence — aligned = well calibrated)")
    print()
    print(f"  {'Bin':>10} | {'Accuracy':>8} | {'Conf':>6} | Chart")
    print(f"  {'-'*70}")
    for b in bins:
        if b["count"] == 0:
            continue
        bar_len   = int(b["accuracy"] * 40)
        ideal_len = int(b["avg_confidence"] * 40)
        bar       = "█" * bar_len
        # Add confidence marker if it falls beyond bar
        suffix = "│" if ideal_len > bar_len else ""
        print(f"  {b['bin_low']:.2f}-{b['bin_high']:.2f} | "
              f"{b['accuracy']*100:>7.1f}% | "
              f"{b['avg_confidence']*100:>5.1f}% | "
              f"{bar}{suffix}")

ascii_reliability_diagram(baseline_bins,  "\nBASELINE — ASCII Reliability Diagram")
ascii_reliability_diagram(finetuned_bins, "\nFINE-TUNED — ASCII Reliability Diagram")

# ── 6. Save Results ────────────────────────────────────────────────────────────
with open("reliability_results.json", "w") as f:
    json.dump({
        "baseline": {
            "bins"    : baseline_bins,
            "ece"     : baseline_ece,
            "mce"     : baseline_mce,
            "avg_conf": baseline_avg_conf,
            "avg_acc" : baseline_avg_acc,
            "total"   : baseline_total,
        },
        "finetuned": {
            "bins"    : finetuned_bins,
            "ece"     : finetuned_ece,
            "mce"     : finetuned_mce,
            "avg_conf": finetuned_avg_conf,
            "avg_acc" : finetuned_avg_acc,
            "total"   : finetuned_total,
        },
    }, f, indent=2)

print(f"\nSaved reliability_results.json")
