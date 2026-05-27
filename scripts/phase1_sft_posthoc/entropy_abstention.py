"""
entropy_abstention.py
---------------------
Compare two abstention strategies on baseline and fine-tuned model results:
  1. Max-prob thresholding : abstain if max(probs) < threshold
  2. Entropy thresholding  : abstain if entropy(probs) > threshold

Entropy measures how concentrated the model's probability mass is over
the four answer options (A/B/C/D) — not over the full vocabulary.
High entropy = model is confused = should abstain.
Low entropy  = model is confident = should answer.

Formula: entropy = -sum(p * log(p) for each option)
Range   : 0 (perfectly certain) to log(4) ≈ 1.386 (completely uncertain)

Input  : baseline_results.json, finetuned_results.json
Output : entropy_abstention_results.json
"""

import json
import numpy as np
import math

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

# ── 2. Compute Entropy ─────────────────────────────────────────────────────────
def compute_entropy(all_probs):
    """
    Compute entropy of probability distribution over 4 options.
    entropy = -sum(p * log(p))
    Small epsilon added to avoid log(0).
    Note: probs are already softmax-normalized over A/B/C/D only,
    so this measures uncertainty among the four answer choices.
    """
    probs = [all_probs["A"], all_probs["B"], all_probs["C"], all_probs["D"]]
    return -sum(p * math.log(p + 1e-10) for p in probs)

# Add entropy to each prediction
for p in baseline_preds:
    p["entropy"] = compute_entropy(p["all_probs"])

for p in finetuned_preds:
    p["entropy"] = compute_entropy(p["all_probs"])

# ── 3. Sweep Functions ─────────────────────────────────────────────────────────
def sweep_maxprob(predictions, thresholds):
    """Abstain if max_prob < threshold."""
    total = len(predictions)
    rows  = []
    for t in thresholds:
        answered  = [p for p in predictions if p["confidence"] >= t]
        abstained = [p for p in predictions if p["confidence"] < t]
        if not answered:
            continue
        correct           = sum(p["is_correct"] for p in answered)
        wrong             = len(answered) - correct
        answered_accuracy = correct / len(answered)
        coverage          = len(answered) / total
        wrong_answer_rate = wrong / total
        abstention_rate   = len(abstained) / total
        rows.append({
            "threshold"         : round(t, 3),
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

def sweep_entropy(predictions, thresholds):
    """Abstain if entropy > threshold."""
    total = len(predictions)
    rows  = []
    for t in thresholds:
        answered  = [p for p in predictions if p["entropy"] <= t]
        abstained = [p for p in predictions if p["entropy"] > t]
        if not answered:
            continue
        correct           = sum(p["is_correct"] for p in answered)
        wrong             = len(answered) - correct
        answered_accuracy = correct / len(answered)
        coverage          = len(answered) / total
        wrong_answer_rate = wrong / total
        abstention_rate   = len(abstained) / total
        rows.append({
            "threshold"         : round(t, 3),
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

# ── 4. Run Sweeps ──────────────────────────────────────────────────────────────
maxprob_thresholds = np.arange(0.0, 1.0, 0.05).tolist()
entropy_thresholds = np.arange(0.0, 1.401, 0.05).tolist()  # finer steps

print("\nSweeping max-prob thresholds...")
baseline_maxprob  = sweep_maxprob(baseline_preds, maxprob_thresholds)
finetuned_maxprob = sweep_maxprob(finetuned_preds, maxprob_thresholds)

print("Sweeping entropy thresholds...")
baseline_entropy  = sweep_entropy(baseline_preds, entropy_thresholds)
finetuned_entropy = sweep_entropy(finetuned_preds, entropy_thresholds)

# ── 5. Print Tables ────────────────────────────────────────────────────────────
def print_table(rows, title):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"{'Threshold':>12} {'Ans.Accuracy':>14} {'Coverage':>10} "
          f"{'WrongRate':>11} {'Abstained':>10} {'Answered':>9}")
    print(f"{'-'*80}")
    for row in rows:
        print(f"{row['threshold']:>12.3f} "
              f"{row['answered_accuracy']*100:>13.2f}% "
              f"{row['coverage']*100:>9.2f}% "
              f"{row['wrong_answer_rate']*100:>10.2f}% "
              f"{row['abstention_rate']*100:>9.2f}% "
              f"{row['answered']:>9}")

print_table(baseline_maxprob,  "BASELINE — Max-Prob Abstention")
print_table(finetuned_maxprob, "FINE-TUNED — Max-Prob Abstention")
print_table(baseline_entropy,  "BASELINE — Entropy Abstention")
print_table(finetuned_entropy, "FINE-TUNED — Entropy Abstention")

# ── 6. Find Sweet Spots ────────────────────────────────────────────────────────
def find_sweet_spot(rows, min_coverage=0.4, min_accuracy=0.70):
    candidates = [
        r for r in rows
        if r["coverage"] >= min_coverage and r["answered_accuracy"] >= min_accuracy
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["wrong_answer_rate"])

print(f"\n{'='*80}")
print("SWEET SPOT COMPARISON (coverage >= 40%, accuracy >= 70%)")
print(f"{'='*80}")

ft_maxprob_sweet = find_sweet_spot(finetuned_maxprob)
ft_entropy_sweet = find_sweet_spot(finetuned_entropy)
bl_maxprob_sweet = find_sweet_spot(baseline_maxprob)
bl_entropy_sweet = find_sweet_spot(baseline_entropy)

def print_sweet_spot(label, sweet):
    print(f"\n{label}:")
    if sweet:
        print(f"  Threshold        : {sweet['threshold']}")
        print(f"  Answered Accuracy: {sweet['answered_accuracy']*100:.2f}%")
        print(f"  Coverage         : {sweet['coverage']*100:.2f}%")
        print(f"  Wrong Rate       : {sweet['wrong_answer_rate']*100:.2f}%")
        print(f"  Answered         : {sweet['answered']} / {sweet['answered'] + sweet['abstained']}")
    else:
        print("  No sweet spot found")

print_sweet_spot("Baseline   — Max-Prob", bl_maxprob_sweet)
print_sweet_spot("Baseline   — Entropy",  bl_entropy_sweet)
print_sweet_spot("Fine-tuned — Max-Prob", ft_maxprob_sweet)
print_sweet_spot("Fine-tuned — Entropy",  ft_entropy_sweet)

# ── 7. Head-to-Head at ~50% Coverage ──────────────────────────────────────────
def find_closest_coverage(rows, target=0.5):
    return min(rows, key=lambda r: abs(r["coverage"] - target))

print(f"\n{'='*80}")
print("HEAD-TO-HEAD: Max-Prob vs Entropy at ~50% Coverage")
print(f"{'='*80}")

for label, mp, ent in [
    ("Baseline",   baseline_maxprob,  baseline_entropy),
    ("Fine-tuned", finetuned_maxprob, finetuned_entropy),
]:
    mp_50  = find_closest_coverage(mp)
    ent_50 = find_closest_coverage(ent)
    print(f"\n{label}:")
    print(f"  {'Method':<20} {'Ans.Accuracy':>14} {'Coverage':>10} "
          f"{'Wrong Rate':>12} {'Answered':>9}")
    print(f"  {'-'*65}")
    print(f"  {'Max-Prob':<20} "
          f"{mp_50['answered_accuracy']*100:>13.2f}% "
          f"{mp_50['coverage']*100:>9.2f}% "
          f"{mp_50['wrong_answer_rate']*100:>11.2f}% "
          f"{mp_50['answered']:>9}")
    print(f"  {'Entropy':<20} "
          f"{ent_50['answered_accuracy']*100:>13.2f}% "
          f"{ent_50['coverage']*100:>9.2f}% "
          f"{ent_50['wrong_answer_rate']*100:>11.2f}% "
          f"{ent_50['answered']:>9}")

# ── 8. Save Results ────────────────────────────────────────────────────────────
with open("entropy_abstention_results.json", "w") as f:
    json.dump({
        "baseline_maxprob"   : baseline_maxprob,
        "finetuned_maxprob"  : finetuned_maxprob,
        "baseline_entropy"   : baseline_entropy,
        "finetuned_entropy"  : finetuned_entropy,
        "bl_maxprob_sweet"   : bl_maxprob_sweet,
        "bl_entropy_sweet"   : bl_entropy_sweet,
        "ft_maxprob_sweet"   : ft_maxprob_sweet,
        "ft_entropy_sweet"   : ft_entropy_sweet,
    }, f, indent=2)

print(f"\nSaved entropy_abstention_results.json")
