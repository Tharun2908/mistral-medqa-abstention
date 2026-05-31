# eval_grpo_sweep.py
"""
Day 6: evaluate every GRPO v4 checkpoint on the full MedQA test set.

CRITICAL: GRPO LoRA was trained on top of the MERGED WARM-START, not raw Mistral.
The eval must reconstruct that base in memory before attaching the GRPO adapter,
otherwise the adapter is composed against the wrong weights.

Builds checkpoint_tradeoff.json mirroring the DPO v3 format, plus a per-checkpoint
results JSON. Use this output to pick the headline operating point, the same way
DPO ckpt-540 was selected.
"""
import os, json, glob
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import dpo_eval_full

BASE      = "mistralai/Mistral-7B-v0.3"
WARMSTART = "/workspace/mistral-medqa-warmstart/policy"
GRPO_DIR  = "/workspace/mistral-medqa-grpo-v4.2"
OUT_JSON  = "/workspace/grpo_v4_2_checkpoint_tradeoff.json"

def build_eval_model(grpo_adapter_path: str):
    """Reconstruct: raw Mistral bf16 -> merge warm-start -> attach GRPO adapter."""
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # match dpo_eval_full's padding side
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto",
    )
    merged = PeftModel.from_pretrained(base, WARMSTART).merge_and_unload()
    model = PeftModel.from_pretrained(merged, grpo_adapter_path)
    model.eval()
    return model, tok


def evaluate_one(adapter_path: str, limit: int = None) -> dict:
    """Monkeypatch dpo_eval_full.load_model to build the correct stack, then run."""
    def loader():
        print(f"  loading: raw Mistral -> warm-start merge -> {adapter_path}")
        return build_eval_model(adapter_path)

    dpo_eval_full.load_model = loader

    # dpo_eval_full.main() uses argparse; we set sys.argv to control it.
    import sys
    old_argv = sys.argv[:]
    sys.argv = ["dpo_eval_full.py"]
    if limit:
        sys.argv += ["--limit", str(limit)]
    try:
        dpo_eval_full.main()
    finally:
        sys.argv = old_argv

    # main() writes dpo_eval_full_results.json in cwd
    with open("dpo_eval_full_results.json") as f:
        return json.load(f)


def main():
    # find checkpoints: both "checkpoint-N" and "final"
    ckpts = sorted(glob.glob(f"{GRPO_DIR}/checkpoint-*"),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    final_dir = f"{GRPO_DIR}/final"
    if os.path.isdir(final_dir):
        ckpts.append(final_dir)

    print(f"Found {len(ckpts)} checkpoints to evaluate:")
    for c in ckpts:
        print(f"  {c}")

    trajectory = {}
    for ckpt in ckpts:
        name = os.path.basename(ckpt)
        print(f"\n=== {name} ===")
        try:
            results = evaluate_one(ckpt)
            nop = results["natural_operating_point"]
            pe  = results["pe_diagnostic"]
            trajectory[name] = {
                "coverage":       nop["coverage"],
                "abstain_rate":   nop["abstain_rate"],
                "answered_acc":   nop["answered_accuracy"],
                "dataset_wrong":  nop["dataset_wrong_rate"],
                "pe_auroc":       pe["auroc_pe_as_wrongness_detector"],
                "max_pe":         pe["max_pe"],
            }
            # archive the full per-checkpoint result
            os.makedirs(f"{GRPO_DIR}/eval", exist_ok=True)
            with open(f"{GRPO_DIR}/eval/results_{name}.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  cov={nop['coverage']:.3f}  acc={nop['answered_accuracy']:.3f}  "
                  f"wrong={nop['dataset_wrong_rate']:.3f}  P(E) AUROC={pe['auroc_pe_as_wrongness_detector']:.3f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            trajectory[name] = {"error": str(e)}

        with open(OUT_JSON, "w") as f:                  # save after EACH ckpt
            json.dump(trajectory, f, indent=2)

        # free memory between checkpoints
        torch.cuda.empty_cache()

    print(f"\nTrajectory saved -> {OUT_JSON}")
    print("Done. Inspect to pick the headline operating point.")


if __name__ == "__main__":
    main()
