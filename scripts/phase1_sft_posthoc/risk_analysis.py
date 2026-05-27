"""
risk_analysis.py
----------------
Extract examples for manual risk-weighted evaluation.

Selects:
  - 10 high-confidence wrong answers (model was sure but wrong)
  - 10 low-confidence examples (confidence < 0.50, abstained at sweet spot)

These are printed for manual qualitative analysis.
Goal: categorize errors as critical vs benign in medical context.

Input  : finetuned_results.json
Output : risk_analysis_examples.json
"""

import json

# ── 1. Load Results ────────────────────────────────────────────────────────────
print("Loading results...")
with open("finetuned_results.json", "r") as f:
    finetuned_data = json.load(f)

predictions = finetuned_data["predictions"]
print(f"Total examples: {len(predictions)}")

# ── 2. Extract High-Confidence Wrong Answers ───────────────────────────────────
# Most dangerous — model is very sure but wrong
high_conf_wrong = [
    p for p in predictions
    if not p["is_correct"] and p["confidence"] >= 0.70
]
high_conf_wrong = sorted(high_conf_wrong, key=lambda p: p["confidence"], reverse=True)
top_wrong = high_conf_wrong[:10]

# ── 3. Extract Low-Confidence Examples ────────────────────────────────────────
# Examples model abstains on at sweet-spot threshold = 0.50
# Using < 0.35 to get the most uncertain subset for clearer analysis
low_conf = [
    p for p in predictions
    if p["confidence"] < 0.35   # strict subset of threshold=0.50 abstentions
]
low_conf = sorted(low_conf, key=lambda p: p["confidence"])
top_abstained = low_conf[:10]

# Count correct vs wrong among abstained
low_conf_correct = sum(p["is_correct"] for p in low_conf)
low_conf_wrong   = len(low_conf) - low_conf_correct

# ── 4. Print High-Confidence Wrong Answers ────────────────────────────────────
print(f"\n{'='*80}")
print("HIGH-CONFIDENCE WRONG ANSWERS (top 10)")
print("Most dangerous — model was very sure but wrong")
print(f"{'='*80}")

for i, p in enumerate(top_wrong):
    print(f"\n[Example {i+1}]")
    print(f"  Confidence  : {p['confidence']*100:.1f}%")
    print(f"  Predicted   : {p['prediction']} — {p['options'][p['prediction']]}")
    print(f"  Correct     : {p['ground_truth']} — {p['options'][p['ground_truth']]}")
    print(f"  All probs   : A={p['all_probs']['A']:.3f} B={p['all_probs']['B']:.3f} "
          f"C={p['all_probs']['C']:.3f} D={p['all_probs']['D']:.3f}")
    print(f"  Question    : {p['question']}")
    print(f"  Options     :")
    for k, v in p["options"].items():
        marker = "← CORRECT" if k == p["ground_truth"] else \
                 "← PREDICTED" if k == p["prediction"] else ""
        print(f"    {k}: {v} {marker}")

# ── 5. Print Low-Confidence Examples ──────────────────────────────────────────
print(f"\n{'='*80}")
print("LOW-CONFIDENCE EXAMPLES (top 10 — strict subset of abstentions)")
print("Confidence < 0.35 — model is most uncertain here")
print(f"{'='*80}")

for i, p in enumerate(top_abstained):
    status = "CORRECT" if p["is_correct"] else "WRONG"
    print(f"\n[Example {i+1}]")
    print(f"  Confidence  : {p['confidence']*100:.1f}%")
    print(f"  Predicted   : {p['prediction']} — {p['options'][p['prediction']]} ({status})")
    print(f"  Correct     : {p['ground_truth']} — {p['options'][p['ground_truth']]}")
    print(f"  All probs   : A={p['all_probs']['A']:.3f} B={p['all_probs']['B']:.3f} "
          f"C={p['all_probs']['C']:.3f} D={p['all_probs']['D']:.3f}")
    print(f"  Question    : {p['question']}")
    print(f"  Options     :")
    for k, v in p["options"].items():
        marker = "← CORRECT" if k == p["ground_truth"] else \
                 "← PREDICTED" if k == p["prediction"] else ""
        print(f"    {k}: {v} {marker}")

# ── 6. Summary Stats ───────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"  Total high-confidence wrong (>=70%) : {len(high_conf_wrong)}")
print(f"  Total low-confidence (<35%)         : {len(low_conf)}")
print(f"  Low-confidence correct              : {low_conf_correct}")
print(f"  Low-confidence wrong                : {low_conf_wrong}")
print(f"  Showing top 10 of each for manual review")

# ── 7. Save for Manual Review ──────────────────────────────────────────────────
with open("risk_analysis_examples.json", "w") as f:
    json.dump({
        "high_confidence_wrong" : top_wrong,
        "low_confidence"        : top_abstained,
        "stats": {
            "total_high_conf_wrong" : len(high_conf_wrong),
            "total_low_conf"        : len(low_conf),
            "low_conf_correct"      : low_conf_correct,
            "low_conf_wrong"        : low_conf_wrong,
        }
    }, f, indent=2)

print(f"\nSaved risk_analysis_examples.json")
print(f"\nNext step: manually read each question and categorize as:")
print(f"  critical  — wrong answer could cause direct harm")
print(f"  benign    — wrong answer is suboptimal but not dangerous")
print(f"  ambiguous — hard to tell without full clinical context")
