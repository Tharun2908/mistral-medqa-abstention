"""
dpo_eval_full.py
----------------
Evaluate the DPO learned-abstention model on the MedQA-USMLE test set.

CORRECT FORMAT: scores FULL-SENTENCE completions, matching dpo_pairs.json:
  A -> " The answer is A."
  ...
  E -> " I cannot answer confidently."

Scoring = mean per-token log-prob of each completion (length-normalized so the
longer abstain sentence isn't penalized for having more tokens).

Per example we record:
  decision   = argmax over {A,B,C,D,E} full-sentence scores  (E = abstain).
               Drives the NATURAL operating point (no threshold).
  would_be   = argmax over {A,B,C,D} only (ignores abstain).
  conf       = max softmax over the four ANSWER scores (SFT-comparable).
  p_abstain  = softmax-over-all-five, the E component. The SOFT abstention
               signal: did DPO raise abstain preference on wrong cases even
               when the hard argmax still picked an answer?
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import json
import argparse
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score

BASE_MODEL     = "mistralai/Mistral-7B-v0.3"
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./mistral-medqa-dpo-final")
DPO_ADAPTER    = os.environ.get("DPO_ADAPTER", "./mistral-medqa-dpo-final/policy")
DATASET        = "GBaker/MedQA-USMLE-4-options"
OUT_JSON       = "dpo_eval_full_results.json"
OUT_ROWS       = "dpo_eval_full_predictions.jsonl"

ANSWER_SET = ["A", "B", "C", "D"]
LETTERS    = ["A", "B", "C", "D", "E"]
MAX_LEN    = 1024   # prompt is left-trimmed to fit; completion always kept whole

COMPLETIONS = {
    "A": " The answer is A.",
    "B": " The answer is B.",
    "C": " The answer is C.",
    "D": " The answer is D.",
    "E": " I cannot answer confidently.",
}


def build_prompt(question, options):
    """Match the exact DPO/SFT prompt format (verified vs dpo_pairs.json)."""
    opt_lines = "\n".join(f"{k}: {options[k]}" for k in ANSWER_SET)
    return f"Question: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:"


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok_src = TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH) else BASE_MODEL
    tok = AutoTokenizer.from_pretrained(tok_src)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, DPO_ADAPTER)
    model.eval()
    return model, tok


@torch.no_grad()
def score_completion_batch(model, tok, prompt, completions):
    """Mean per-token log-prob of each full completion.

    The prompt is left-trimmed (oldest tokens dropped) so the completion is
    ALWAYS kept whole -- this prevents the -inf/NaN that arises when a long
    prompt fills the length budget and the completion gets truncated away.
    """
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id
    prompt_ids = tok(prompt, add_special_tokens=True).input_ids

    seqs, comp_spans = [], []
    for comp in completions:
        comp_ids = tok(comp, add_special_tokens=False).input_ids
        budget = MAX_LEN - len(comp_ids)
        p_ids = prompt_ids[-budget:] if len(prompt_ids) > budget else prompt_ids
        ids = p_ids + comp_ids
        seqs.append(ids)
        comp_spans.append((len(p_ids), len(ids)))   # (start, end) of completion

    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()

    scores = []
    for i, (start, end) in enumerate(comp_spans):
        pred_logits = logits[i, start - 1:end - 1, :]
        labels = input_ids[i, start:end]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        token_logps = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        scores.append(float(token_logps.mean().item()))
    return scores


def score_example(model, tok, prompt):
    scores = score_completion_batch(model, tok, prompt,
                                    [COMPLETIONS[L] for L in LETTERS])
    score_map = {L: scores[i] for i, L in enumerate(LETTERS)}

    decision = max(LETTERS, key=lambda L: score_map[L])       # includes E
    would_be = max(ANSWER_SET, key=lambda L: score_map[L])    # excludes E

    answer_scores = torch.tensor([score_map[L] for L in ANSWER_SET])
    conf = float(torch.softmax(answer_scores, dim=0).max().item())

    five_scores = torch.tensor([score_map[L] for L in LETTERS])
    p_abstain = float(torch.softmax(five_scores, dim=0)[LETTERS.index("E")].item())

    return decision, would_be, conf, p_abstain, score_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("Loading test set...")
    test = load_dataset(DATASET, split="test")
    if args.limit:
        test = test.select(range(args.limit))

    print("Loading model...")
    model, tok = load_model()

    if args.inspect:
        ex = test[0]
        prompt = build_prompt(ex["question"], ex["options"])
        decision, would_be, conf, p_abstain, score_map = score_example(model, tok, prompt)
        print("\n----- FIRST PROMPT -----")
        print(repr(prompt))
        print("Gold answer_idx:", ex.get("answer_idx"))
        print("\nFull-sentence scores (mean logprob):")
        for k, v in score_map.items():
            print(f"  {k}: {v:.4f}")
        print(f"decision={decision}  would_be={would_be}  "
              f"conf={conf:.4f}  P(E)={p_abstain:.4f}")
        return

    rows = []
    for i, ex in enumerate(test):
        prompt = build_prompt(ex["question"], ex["options"])
        decision, would_be, conf, p_abstain, score_map = score_example(model, tok, prompt)
        gold = ex["answer_idx"]
        rows.append({
            "idx": i, "decision": decision, "would_be": would_be,
            "conf": conf, "p_abstain": p_abstain, "gold": gold,
            "abstain": decision == "E",
            "answered": decision in ANSWER_SET,
            "decision_correct": (decision in ANSWER_SET) and (decision == gold),
            "wouldbe_correct": would_be == gold,
            "scores": score_map,
        })
        if (i + 1) % 50 == 0:
            print(f"Evaluated {i + 1}/{len(test)}")

    n = len(rows)

    # Save the expensive per-row output IMMEDIATELY, before any metric that
    # could raise. A metric error must never cost the forward-pass compute.
    with open(OUT_ROWS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved rows -> {OUT_ROWS} ({n} rows)")

    answered = [r for r in rows if r["answered"]]
    n_answered = len(answered)
    coverage = n_answered / n
    abstain_rate = sum(r["abstain"] for r in rows) / n
    answered_acc = (sum(r["decision_correct"] for r in answered) / n_answered
                    if n_answered else 0.0)
    dataset_wrong = sum((not r["decision_correct"]) and r["answered"]
                        for r in rows) / n

    y_true  = np.array([int(r["wouldbe_correct"]) for r in rows])
    y_conf  = np.array([r["conf"] for r in rows])
    pe      = np.array([r["p_abstain"] for r in rows])

    # Backstop: drop any non-finite scores before AUROC so one bad row can't
    # crash the whole summary. n_dropped is reported for transparency.
    finite = np.isfinite(y_conf) & np.isfinite(pe)
    n_dropped = int((~finite).sum())
    yt, yc, pf = y_true[finite], y_conf[finite], pe[finite]
    auroc_conf = (roc_auc_score(yt, yc)
                  if 0 < yt.sum() < len(yt) else float("nan"))

    # --- P(E) diagnostic: is abstain preference higher on WRONG cases? ------
    pe_correct = pf[yt == 1]
    pe_wrong   = pf[yt == 0]
    pe_auroc = (roc_auc_score(1 - yt, pf)      # label 1 = WRONG
                if 0 < (yt == 0).sum() < len(yt) else float("nan"))

    # --- Soft-abstention curve: abstain when P(E) >= tau --------------------
    # Thresholds chosen as P(E) quantiles so coverage spans a useful range
    # regardless of the absolute scale of P(E).
    qs = np.quantile(pf, np.arange(0.50, 1.0, 0.05))
    pe_curve = []
    for tau in np.unique(np.round(qs, 4)):
        ans = [r for r in rows if r["p_abstain"] < tau]
        cov = len(ans) / n
        acc = (sum(r["wouldbe_correct"] for r in ans) / len(ans)
               if ans else float("nan"))
        pe_curve.append({"pe_threshold": float(tau), "coverage": cov,
                         "answered_accuracy": acc})

    results = {
        "n_test": n,
        "natural_operating_point": {
            "coverage": coverage, "abstain_rate": abstain_rate,
            "answered_accuracy": answered_acc,
            "dataset_wrong_rate": dataset_wrong,
        },
        "auroc_confidence": auroc_conf,
        "n_dropped_nonfinite": n_dropped,
        "pe_diagnostic": {
            "mean_pe_on_wouldbe_correct": float(pe_correct.mean()) if len(pe_correct) else float("nan"),
            "mean_pe_on_wouldbe_wrong": float(pe_wrong.mean()) if len(pe_wrong) else float("nan"),
            "max_pe": float(pf.max()) if len(pf) else float("nan"),
            "auroc_pe_as_wrongness_detector": pe_auroc,
        },
        "pe_soft_abstention_sweep": pe_curve,
        "config": {"base": BASE_MODEL, "tokenizer": TOKENIZER_PATH,
                   "adapter": DPO_ADAPTER, "dataset": DATASET,
                   "scoring": "mean_completion_logprob",
                   "completions": COMPLETIONS},
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========== DPO FULL-COMPLETION EVAL ==========")
    print(f"Test examples            : {n}")
    print("---- Natural operating point (learned abstain) ----")
    print(f"Coverage                 : {coverage:.4f}")
    print(f"Abstain rate             : {abstain_rate:.4f}")
    print(f"Answered accuracy        : {answered_acc:.4f}")
    print(f"Dataset-level wrong rate : {dataset_wrong:.4f}")
    print(f"AUROC (confidence)       : {auroc_conf:.4f}")
    if n_dropped:
        print(f"(dropped {n_dropped} non-finite rows from AUROC/P(E) metrics)")
    print("---- P(E) soft-abstention diagnostic ----")
    print(f"Mean P(E) | would-be CORRECT : {results['pe_diagnostic']['mean_pe_on_wouldbe_correct']:.4f}")
    print(f"Mean P(E) | would-be WRONG   : {results['pe_diagnostic']['mean_pe_on_wouldbe_wrong']:.4f}")
    print(f"Max P(E) observed            : {results['pe_diagnostic']['max_pe']:.4f}")
    print(f"AUROC of P(E) as wrongness   : {pe_auroc:.4f}")
    print(f"\nSaved summary -> {OUT_JSON}")
    print(f"Saved rows    -> {OUT_ROWS}")


if __name__ == "__main__":
    main()
