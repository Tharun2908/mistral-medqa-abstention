"""
eval_checkpoints.py
-------------------
Evaluate every surviving DPO v2 checkpoint on a test subset, to see whether any
intermediate checkpoint sits at a better coverage/accuracy tradeoff than the
over-conservative final adapter.

NOTE: v2 trained with save_total_limit=3, so only the LAST ~3 checkpoints exist;
the early/mid checkpoints (where abstention was still descending) were deleted.
This script maps whatever survived. Reuses dpo_eval_full's scoring exactly.

Usage:
  python eval_checkpoints.py --limit 300          # fast subset
  python eval_checkpoints.py --limit 300 --ckpt_dir ./mistral-medqa-dpo-v2
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import json
import glob
import argparse
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score

BASE_MODEL = "mistralai/Mistral-7B-v0.3"
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./mistral-medqa-dpo-v2-final")
DATASET = "GBaker/MedQA-USMLE-4-options"

ANSWER_SET = ["A", "B", "C", "D"]
LETTERS    = ["A", "B", "C", "D", "E"]
MAX_LEN    = 1024
COMPLETIONS = {
    "A": " The answer is A.", "B": " The answer is B.",
    "C": " The answer is C.", "D": " The answer is D.",
    "E": " I cannot answer confidently.",
}


def build_prompt(question, options):
    opt_lines = "\n".join(f"{k}: {options[k]}" for k in ANSWER_SET)
    return f"Question: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:"


@torch.no_grad()
def score_completion_batch(model, tok, prompt, completions):
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id
    prompt_ids = tok(prompt, add_special_tokens=True).input_ids
    seqs, spans = [], []
    for comp in completions:
        c_ids = tok(comp, add_special_tokens=False).input_ids
        budget = MAX_LEN - len(c_ids)
        p_ids = prompt_ids[-budget:] if len(prompt_ids) > budget else prompt_ids
        ids = p_ids + c_ids
        seqs.append(ids); spans.append((len(p_ids), len(ids)))
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s); attn[i, :len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)
    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    scores = []
    for i, (start, end) in enumerate(spans):
        pl = logits[i, start - 1:end - 1, :]
        lab = input_ids[i, start:end]
        lp = torch.log_softmax(pl, dim=-1)
        scores.append(float(lp.gather(1, lab.unsqueeze(1)).mean()))
    return scores


def score_example(model, tok, prompt):
    s = score_completion_batch(model, tok, prompt, [COMPLETIONS[L] for L in LETTERS])
    sm = {L: s[i] for i, L in enumerate(LETTERS)}
    decision = max(LETTERS, key=lambda L: sm[L])
    would_be = max(ANSWER_SET, key=lambda L: sm[L])
    conf = float(torch.softmax(torch.tensor([sm[L] for L in ANSWER_SET]), 0).max())
    p_abst = float(torch.softmax(torch.tensor([sm[L] for L in LETTERS]), 0)[4])
    return decision, would_be, conf, p_abst


def eval_adapter(adapter_path, tok, test):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    rows = []
    for ex in test:
        prompt = build_prompt(ex["question"], ex["options"])
        d, w, c, pe = score_example(model, tok, prompt)
        gold = ex["answer_idx"]
        rows.append({"abstain": d == "E", "answered": d in ANSWER_SET,
                     "decision_correct": d in ANSWER_SET and d == gold,
                     "wouldbe_correct": w == gold, "conf": c, "p_abstain": pe})
    n = len(rows)
    ans = [r for r in rows if r["answered"]]
    coverage = len(ans) / n
    abstain = sum(r["abstain"] for r in rows) / n
    acc = sum(r["decision_correct"] for r in ans) / len(ans) if ans else 0.0
    wrong = sum((not r["decision_correct"]) and r["answered"] for r in rows) / n
    yt = np.array([int(r["wouldbe_correct"]) for r in rows])
    pe = np.array([r["p_abstain"] for r in rows])
    pe_auroc = (roc_auc_score(1 - yt, pe) if 0 < (yt == 0).sum() < n else float("nan"))

    del model, base
    torch.cuda.empty_cache()
    return {"coverage": coverage, "abstain_rate": abstain,
            "answered_acc": acc, "dataset_wrong": wrong, "pe_auroc": pe_auroc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./mistral-medqa-dpo-v2")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--include_final", action="store_true",
                    help="Also eval the final saved adapter for reference.")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint-*")),
                   key=lambda p: int(p.split("-")[-1]))
    targets = [(os.path.basename(c), os.path.join(c, "policy")
                if os.path.isdir(os.path.join(c, "policy")) else c) for c in ckpts]
    if args.include_final:
        targets.append(("FINAL", "./mistral-medqa-dpo-v2-final/policy"))

    if not targets:
        print(f"No checkpoints found under {args.ckpt_dir}. "
              f"(save_total_limit may have removed them, or path differs.)")
        return

    print(f"Found {len(targets)} adapters to evaluate:")
    for name, path in targets:
        print(f"  {name}: {path}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    test = load_dataset(DATASET, split="test").select(range(args.limit))
    print(f"\nEvaluating on {args.limit} test examples each...\n")

    results = {}
    for name, path in targets:
        print(f"--- {name} ({path}) ---")
        try:
            m = eval_adapter(path, tok, test)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        results[name] = m
        print(f"  coverage={m['coverage']:.3f}  abstain={m['abstain_rate']:.3f}  "
              f"ans_acc={m['answered_acc']:.3f}  wrong={m['dataset_wrong']:.3f}  "
              f"P(E)_AUROC={m['pe_auroc']:.3f}")

    print("\n================ CHECKPOINT TRADEOFF TABLE ================")
    print(f"{'ckpt':<14}{'cov':>7}{'abst':>7}{'ans_acc':>9}{'wrong':>8}{'PE_AUROC':>10}")
    for name, m in results.items():
        print(f"{name:<14}{m['coverage']:>7.3f}{m['abstain_rate']:>7.3f}"
              f"{m['answered_acc']:>9.3f}{m['dataset_wrong']:>8.3f}{m['pe_auroc']:>10.3f}")

    with open("checkpoint_tradeoff.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> checkpoint_tradeoff.json")
    print("Pick the row with the best coverage WHILE P(E)_AUROC stays >~0.65 and")
    print("wrong-rate stays low. If all rows cluster near cov~0.30, v3 is needed.")


if __name__ == "__main__":
    main()
