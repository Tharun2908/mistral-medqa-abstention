"""
build_dpo_pairs_v2.py
---------------------
Rebuild DPO preference pairs to fix the failed-abstention run.

Design (locked in):
  * Abstain-chosen pairs come from the FULL wrong pool, sampled with
    probability proportional to confidence (tilts toward high-confidence-wrong
    -- the dangerous, learnable cases -- without a hard threshold).
      chosen   = abstain sentence
      rejected = the wrong answer the model actually gave
  * Answer-chosen pairs come from CORRECT predictions, sampled uniformly so
    they span the full confidence range (teaches "don't abstain just because
    you're a bit unsure when you're actually right").
      chosen   = the correct answer sentence
      rejected = abstain sentence
  * Ratio 1.5 : 1  (abstain : answer).
  * Full-sentence completions, matching dpo_eval_full.py exactly.
  * A DISJOINT slice of wrong cases is reserved for the SFT warm-start, so DPO
    never just reinforces memorized warm-start examples. Warm-start data is
    written here too (answer-heavy, to introduce the abstain string without
    teaching always-abstain).

Inputs : train_inference_results.json  (keys: accuracy, predictions[])
Outputs: dpo_pairs_v2.json             (train/val preference pairs)
         warmstart_data.json           (SFT examples: prompt + target completion)
"""

import json
import random
import numpy as np

# ---- knobs ----------------------------------------------------------------
SEED            = 42
N_ABSTAIN_PAIRS = 1200     # abstain-chosen DPO pairs
N_ANSWER_PAIRS  = 1200     # answer-chosen DPO pairs  (1.5:1 abstain:answer)
WARMSTART_WRONG = 800      # wrong cases reserved for warm-start (target=abstain)
WARMSTART_ANSWER= 1500     # correct cases for warm-start (target=answer; answer-heavy)
VAL_FRAC        = 0.10
INFILE          = "train_inference_results.json"
PAIRS_OUT       = "dpo_pairs_v3.json"
WARMSTART_OUT   = "warmstart_data.json"

ANSWER_SET = ["A", "B", "C", "D"]
ABSTAIN    = " I cannot answer confidently."
ANSWER_TMPL = " The answer is {}."


def build_prompt(question, options):
    """Identical to dpo_eval_full.py / SFT format."""
    opt_lines = "\n".join(f"{k}: {options[k]}" for k in ANSWER_SET)
    return f"Question: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:"


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    with open(INFILE) as f:
        data = json.load(f)
    preds = data["predictions"]
    print(f"Loaded {len(preds)} predictions  (reported accuracy {data['accuracy']:.4f})")

    # Keep only well-formed records with an A-D prediction.
    preds = [p for p in preds if p.get("prediction") in ANSWER_SET]

    correct = [p for p in preds if p["prediction"] == p["answer_idx"]]
    wrong   = [p for p in preds if p["prediction"] != p["answer_idx"]]
    print(f"Correct: {len(correct)}   Wrong: {len(wrong)}")

    need_wrong = WARMSTART_WRONG + N_ABSTAIN_PAIRS
    need_correct = WARMSTART_ANSWER + N_ANSWER_PAIRS
    if len(wrong) < need_wrong:
        raise ValueError(f"Not enough wrong cases: need {need_wrong}, have {len(wrong)}")
    if len(correct) < need_correct:
        raise ValueError(f"Not enough correct cases: need {need_correct}, have {len(correct)}")

    # --- partition wrong: reserve a uniform slice for warm-start ------------
    random.shuffle(wrong)
    warm_wrong = wrong[:WARMSTART_WRONG]
    dpo_wrong  = wrong[WARMSTART_WRONG:]

    # --- confidence-weighted sample of abstain-chosen pairs -----------------
    conf = np.array([p["confidence"] for p in dpo_wrong], dtype=float)
    probs = conf / conf.sum()
    idx = rng.choice(len(dpo_wrong), size=N_ABSTAIN_PAIRS, replace=False, p=probs)
    abstain_src = [dpo_wrong[i] for i in idx]

    abstain_pairs = []
    for p in abstain_src:
        abstain_pairs.append({
            "prompt": build_prompt(p["question"], p["options"]),
            "chosen": ABSTAIN,
            "rejected": ANSWER_TMPL.format(p["prediction"]),  # the wrong answer
            "type": "abstain", "confidence": p["confidence"],
        })

    # --- partition correct: disjoint DPO answer pairs + warm-start answers --
    random.shuffle(correct)
    answer_src = correct[:N_ANSWER_PAIRS]
    warm_correct = correct[N_ANSWER_PAIRS:N_ANSWER_PAIRS + WARMSTART_ANSWER]

    answer_pairs = []
    for p in answer_src:
        answer_pairs.append({
            "prompt": build_prompt(p["question"], p["options"]),
            "chosen": ANSWER_TMPL.format(p["answer_idx"]),
            "rejected": ABSTAIN,
            "type": "answer", "confidence": p["confidence"],
        })

    # --- shuffle + train/val split ------------------------------------------
    all_pairs = abstain_pairs + answer_pairs
    random.shuffle(all_pairs)
    n_val = int(len(all_pairs) * VAL_FRAC)
    val, train = all_pairs[:n_val], all_pairs[n_val:]

    with open(PAIRS_OUT, "w") as f:
        json.dump({"train": train, "val": val}, f, indent=2)

    # --- warm-start SFT data (answer-heavy) ---------------------------------
    warm = []
    for p in warm_wrong:
        warm.append({"prompt": build_prompt(p["question"], p["options"]),
                     "completion": ABSTAIN, "type": "abstain"})
    for p in warm_correct:
        warm.append({"prompt": build_prompt(p["question"], p["options"]),
                     "completion": ANSWER_TMPL.format(p["answer_idx"]),
                     "type": "answer"})
    random.shuffle(warm)
    with open(WARMSTART_OUT, "w") as f:
        json.dump(warm, f, indent=2)

    # --- distribution report (eyeball before training) ---------------------
    def conf_stats(pairs):
        c = np.array([p["confidence"] for p in pairs])
        return c.mean(), c.min(), c.max()

    print("\n================ DPO PAIRS v2 ================")
    print(f"Abstain pairs : {len(abstain_pairs)}")
    print(f"Answer pairs  : {len(answer_pairs)}")
    print(f"Ratio (abst:ans): {len(abstain_pairs)/len(answer_pairs):.2f} : 1")
    print(f"Train / Val   : {len(train)} / {len(val)}")
    am, alo, ahi = conf_stats(abstain_pairs)
    print(f"Abstain-source confidence  mean={am:.3f}  min={alo:.3f}  max={ahi:.3f}")
    print(f"  (vs all-wrong mean confidence = {conf.mean():.3f}  -- should be higher)")
    nm, nlo, nhi = conf_stats(answer_pairs)
    print(f"Answer-source  confidence  mean={nm:.3f}  min={nlo:.3f}  max={nhi:.3f}")

    # train/val type balance
    for name, split in [("train", train), ("val", val)]:
        na = sum(1 for p in split if p["type"] == "abstain")
        print(f"{name}: abstain={na}  answer={len(split)-na}")

    print("\n---- warm-start data ----")
    nwa = sum(1 for w in warm if w["type"] == "abstain")
    print(f"Total {len(warm)}  (abstain={nwa}, answer={len(warm)-nwa}, "
          f"abstain frac={nwa/len(warm):.2f})")
    print(f"Warm-start wrong cases are DISJOINT from DPO abstain pairs.")

    print("\n---- sample abstain pair ----")
    print(json.dumps(abstain_pairs[0], indent=2)[:600])
    print("\n---- sample answer pair ----")
    print(json.dumps(answer_pairs[0], indent=2)[:600])

    print(f"\nSaved -> {PAIRS_OUT}, {WARMSTART_OUT}")


if __name__ == "__main__":
    main()
