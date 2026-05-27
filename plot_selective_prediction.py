"""
plot_selective_prediction.py
----------------------------
Build the headline figure for Project 2: coverage vs answered-accuracy curves
for each model, obtained by thresholding the calibrated P(E) abstention score.
Each model's NATURAL operating point (its own argmax decision, no threshold) is
marked as a dot so the viewer sees both the tunable curve and the default point.

Reads the saved full-eval JSONs (each has natural_operating_point + the
pe_soft_abstention_sweep array). Add/remove entries in MODELS as needed.

Usage:
  python plot_selective_prediction.py
Output: selective_prediction_curve.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, json_path, color) -- point at your saved results.
MODELS = [
    ("DPO v2 (safety)",        "final_dpo_v3_results/../results_dpo_v2/dpo_eval_full_results.json", "#1b6ca8"),
    ("DPO v3 ckpt-540 (balanced)", "final_dpo_v3_results/results_v3_ck540.json",                     "#c44e52"),
    ("DPO v3 ckpt-190 (alt safety)", "final_dpo_v3_results/results_v3_ck190.json",                   "#55a868"),
]
OUT = "selective_prediction_curve.png"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    natural_points = []   # collect for separate, de-cluttered annotation

    for label, path, color in MODELS:
        try:
            d = load(path)
        except FileNotFoundError:
            print(f"SKIP (not found): {path}")
            continue

        sweep = d.get("pe_soft_abstention_sweep", [])
        pts = [(s["coverage"], s["answered_accuracy"]) for s in sweep
               if s.get("answered_accuracy") == s.get("answered_accuracy")]  # drop NaN
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=color, alpha=0.85, lw=2,
                    label=f"{label} — P(E) threshold curve")

        nop = d.get("natural_operating_point", {})
        if nop:
            natural_points.append((label, color, nop["coverage"],
                                   nop["answered_accuracy"]))

    # --- natural (learned-abstention) operating points, plotted distinctly ---
    # These use the model's 5-way argmax (E can win), a DIFFERENT selection
    # mechanism than P(E)-thresholding. They sit ABOVE the P(E) curves at low
    # coverage -> learned abstention outperforms post-hoc thresholding there.
    for label, color, cov, acc in natural_points:
        ax.plot(cov, acc, "*", color=color, markersize=20,
                markeredgecolor="black", markeredgewidth=1.0, zorder=6)

    # De-cluttered annotations: stack them in a text box instead of overlapping.
    lines = ["Natural operating points (learned 5-way abstention):"]
    for label, color, cov, acc in natural_points:
        lines.append(f"  \u2605 {label}: {cov:.0%} coverage, {acc:.0%} acc")
    ax.text(0.015, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    # SFT baseline reference line
    ax.axhline(0.5224, ls="--", color="gray", alpha=0.6, lw=1)
    ax.annotate("SFT baseline (~52%, answers everything)",
                (0.99, 0.5224), fontsize=8, color="gray", va="bottom", ha="right")

    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Answered accuracy")
    ax.set_title("Selective prediction on MedQA-USMLE (Mistral-7B + warm-start + DPO)\n"
                 "Lines: tune coverage via P(E) threshold.  Stars: learned-abstention "
                 "default points.", fontsize=10.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.45, 0.85)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
