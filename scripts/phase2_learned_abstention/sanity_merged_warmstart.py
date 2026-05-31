# sanity_merged_warmstart.py
"""
Day-0 sanity eval for the in-memory merged warm-start (no persisted merge).

Loads Mistral-7B-v0.3 in bf16, attaches the warm-start LoRA, merge_and_unload()
in memory, then runs the EXISTING dpo_eval_full pipeline against that model by
monkeypatching its load_model(). Reuses dpo_eval_full's scoring + metrics
entirely -- nothing is reimplemented.

Pass gate (from warmstart.log):
    max P(E) ~ 0.10-0.30   (abstain string has real, directed mass)
    abstain rate ~ 2-10%   (not collapsed, not over-abstaining)
    answered acc ~ 0.51    (NOT ~0.49 raw-Mistral -> would mean merge corrupted)

Usage:
    python sanity_merged_warmstart.py --limit 200
"""
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import dpo_eval_full  # the validated eval; we reuse its pipeline

BASE      = "mistralai/Mistral-7B-v0.3"
WARMSTART = "/workspace/mistral-medqa-warmstart/policy"   # 3e-5, v3 lineage


def load_model_merged_bf16():
    """Drop-in replacement for dpo_eval_full.load_model().

    bf16 (NOT 4-bit): merging a LoRA into a quantized base is the silent-
    corruption trap. A100 does bf16 natively, so this is also the faithful
    precision for how GRPO will load the policy.
    Returns (model, tok) -- same contract as the original load_model().
    """
    print("Loading tokenizer (base)...")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # match original eval's padding side

    print("Loading base in bf16 (NOT 4-bit)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto",
    )

    print(f"Attaching warm-start adapter: {WARMSTART}")
    model = PeftModel.from_pretrained(base, WARMSTART)

    print("Merging in memory (merge_and_unload)...")
    model = model.merge_and_unload()
    model.eval()
    print("Merged model ready.")
    return model, tok


# Swap the loader, then run the unchanged pipeline.
dpo_eval_full.load_model = load_model_merged_bf16

if __name__ == "__main__":
    dpo_eval_full.main()
