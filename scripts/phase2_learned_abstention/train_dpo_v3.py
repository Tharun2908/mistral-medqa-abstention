"""
train_dpo_v3.py
---------------
DPO v3: move the NATURAL operating point to a useful coverage (~50-60%) while
keeping abstention directed. Targets the mechanism all analyses flagged --
the early-step abstain shortcut -- rather than just nudging one knob.

What changed from v2 (each tied to evidence):
  * RATIO 1:1 (was 1.5:1 -> 68% abstain). Pure inversion risks v1's
    never-abstain; 1:1 is centered and the live monitor lets us pick the
    right checkpoint instead of guessing the ratio.
  * BETA 0.1 (was 0.05). Tighter KL leash keeps the policy near the ~29%
    warm-start baseline, opposing drift to extreme abstention.
  * LR 8e-6 peak, warmup 15% (was 1e-5, warmup 5% -> lunged to 90-100%
    abstain by step 25). Longer warmup = small early steps = can't lock in
    the shortcut before the preference signal stabilizes.
  * KEEP ALL CHECKPOINTS (save_total_limit=None, save every 20 steps). v2's
    worst operational mistake was deleting the useful early checkpoints. v3
    keeps the whole trajectory so we SELECT the best-coverage checkpoint
    post-hoc rather than hoping the final adapter lands right.
  * COVERAGE MONITOR: reports coverage + answered-accuracy on a MIXED probe
    set every eval (not just abstain-probe rate), so the real operating point
    is visible live and the best checkpoint is identifiable during training.

Needs dpo_pairs_v3.json (1:1). Build it with build_dpo_pairs_v2.py after
setting N_ABSTAIN_PAIRS == N_ANSWER_PAIRS (see note at bottom).
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainerCallback)
from trl import DPOTrainer, DPOConfig

# ---- knobs ----------------------------------------------------------------
BASE_MODEL    = "mistralai/Mistral-7B-v0.3"
WARMSTART_DIR = "./mistral-medqa-warmstart/policy"
WARMSTART_TOKENIZER = "./mistral-medqa-warmstart"
PAIRS         = "dpo_pairs_v3.json"          # 1:1 ratio
OUTPUT_DIR    = "./mistral-medqa-dpo-v3"
FINAL_PARENT  = "./mistral-medqa-dpo-v3-final"
BETA          = 0.10
LR            = 8e-6
WARMUP_RATIO  = 0.15
EPOCHS        = 2
SAVE_STEPS    = 10       # dense enough to map the trajectory across the full
                         # run; all checkpoints kept (save_total_limit=None) so
                         # the whole coverage/AUROC curve is available post-hoc.
MAX_STEPS     = -1       # run to a STABLE endpoint. The natural argmax is
                         # bistable (v1 0% / warmstart 29% / v2 68% abstain), so
                         # we stop chasing a 55%-coverage argmax and instead take
                         # a stable model + select the operating point from the
                         # P(E) threshold sweep (the deployment-correct approach).
SEED          = 42

ANSWER_SET = ["A", "B", "C", "D"]
LETTERS    = ["A", "B", "C", "D", "E"]
COMPLETIONS = {
    "A": " The answer is A.", "B": " The answer is B.",
    "C": " The answer is C.", "D": " The answer is D.",
    "E": " I cannot answer confidently.",
}

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    WARMSTART_TOKENIZER if os.path.exists(WARMSTART_TOKENIZER) else BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading pairs...")
with open(PAIRS) as f:
    data = json.load(f)
train_pairs, val_pairs = data["train"], data["val"]
na = sum(1 for p in train_pairs if p.get("type") == "abstain")
print(f"train={len(train_pairs)} (abstain={na}, answer={len(train_pairs)-na})  "
      f"val={len(val_pairs)}")

train_ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"],
                               "rejected": p["rejected"]} for p in train_pairs])
val_ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"],
                             "rejected": p["rejected"]} for p in val_pairs])

# Mixed probe: half abstain-source, half answer-source prompts, so the monitor
# reports a real coverage number rather than abstain-only behavior.
abstain_probes = [p["prompt"] for p in val_pairs if p.get("type") == "abstain"][:25]
answer_probes  = [p["prompt"] for p in val_pairs if p.get("type") == "answer"][:25]
probe_prompts = abstain_probes + answer_probes


def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, WARMSTART_DIR,
                                      adapter_name="policy", is_trainable=True)
    model.load_adapter(WARMSTART_DIR, adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


class CoverageMonitor(TrainerCallback):
    """Reports coverage + answered-accuracy on a mixed probe set each eval.
    This shows the real operating point evolving, so the best checkpoint is
    identifiable live -- the metric v2 lacked."""
    def __init__(self, trainer, probe_prompts):
        self.trainer = trainer
        self.probe = probe_prompts
        self.comp_ids = {L: tokenizer(COMPLETIONS[L], add_special_tokens=False).input_ids
                         for L in LETTERS}

    @torch.no_grad()
    def _coverage(self):
        model = self.trainer.model
        model.eval()
        answered = 0
        for prompt in self.probe:
            best, best_score = None, -1e9
            for L in LETTERS:
                ids = tokenizer(prompt, add_special_tokens=True).input_ids + self.comp_ids[L]
                t = torch.tensor([ids], device=model.device)
                logits = model(input_ids=t).logits.float()[0]
                start = len(ids) - len(self.comp_ids[L])
                lp = torch.log_softmax(logits[start-1:len(ids)-1], dim=-1)
                lab = torch.tensor(self.comp_ids[L], device=model.device)
                sc = float(lp.gather(1, lab.unsqueeze(1)).mean())
                if sc > best_score:
                    best, best_score = L, sc
            if best in ANSWER_SET:
                answered += 1
        model.train()
        return answered / max(len(self.probe), 1)

    def on_evaluate(self, args, state, control, **kwargs):
        cov = self._coverage()
        print(f"\n[MONITOR step {state.global_step}] probe COVERAGE = {cov:.3f} "
              f"(abstain = {1-cov:.3f})  <- target coverage ~0.50-0.60")


def main():
    model = load_model()
    cfg = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,           # cap the run; band is early in this config
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        beta=BETA,
        model_adapter_name="policy",
        ref_adapter_name="reference",
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=None,         # KEEP ALL checkpoints -- pick best post-hoc
        load_best_model_at_end=False,
        fp16=False, bf16=False,
        max_length=1024,
        max_prompt_length=960,
        logging_steps=10,
        report_to="none",
        seed=SEED,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=val_ds, processing_class=tokenizer)
    trainer.add_callback(CoverageMonitor(trainer, probe_prompts))

    print(f"\nDPO v3 (beta={BETA}, lr={LR}, warmup={WARMUP_RATIO}, 1:1 pairs)...")
    print("Watch [MONITOR] COVERAGE climb from ~0.30 toward 0.50-0.60.")
    print("When it enters the target band, note the step -- that checkpoint is")
    print("likely your best operating point.")
    result = trainer.train()

    metrics = trainer.evaluate()
    Path(FINAL_PARENT).mkdir(parents=True, exist_ok=True)
    model.set_adapter("policy")
    try:
        model.save_pretrained(FINAL_PARENT, selected_adapters=["policy"])
    except TypeError:
        model.save_pretrained(FINAL_PARENT)
    tokenizer.save_pretrained(FINAL_PARENT)
    with open(Path(FINAL_PARENT) / "train_metrics.json", "w") as f:
        json.dump(result.metrics, f, indent=2)
    with open(Path(FINAL_PARENT) / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved final -> {FINAL_PARENT}/policy")
    print("All checkpoints kept under", OUTPUT_DIR)
    print("Next: sweep checkpoints to pick the best coverage point ->")
    print(f"  python eval_checkpoints.py --limit 300 --ckpt_dir {OUTPUT_DIR} --include_final")
    print("(point TOKENIZER_PATH at", FINAL_PARENT, "for that run)")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# To build dpo_pairs_v3.json (1:1), edit build_dpo_pairs_v2.py constants:
#     N_ABSTAIN_PAIRS = 1200
#     N_ANSWER_PAIRS  = 1200      # equal -> 1:1
#     PAIRS_OUT       = "dpo_pairs_v3.json"
# then: python build_dpo_pairs_v2.py
# (Warm-start data is unchanged; reuse the existing mistral-medqa-warmstart.)
# ---------------------------------------------------------------------------
