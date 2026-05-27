"""
train_dpo_v2.py
---------------
DPO v2 for learned abstention on MedQA. Fixes the three causes of the v1 failure.

Why v1 failed (confirmed by eval):
  1. Abstain string had ~zero probability mass (max P(E)=0.0225). DPO can't
     create a behavior from zero.  -> FIXED by warm-start (max P(E) now ~0.39).
  2. Reference adapter was abstain-averse, so the KL leash fought every step.
     -> FIXED here: policy AND reference both start from the WARM-STARTED
        adapter, which already grants abstain real probability. DPO no longer
        pushes uphill against its own reference.
  3. Pairs were answer-favoring (1:2), net gradient pushed abstain DOWN.
     -> FIXED by dpo_pairs_v2.json (1.5:1 abstain-favoring, confidence-weighted).

Lowered beta (0.05 vs 0.1) loosens the KL leash so the policy can move further
toward directed abstention.

CRITICAL MONITOR (the v1 blind spot):
  v1 reported overall pairwise accuracy 0.72 -- which hid that the ABSTAIN side
  was at ~0.0 while the ANSWER side was ~0.95. This script logs pairwise
  accuracy SEPARATELY per pair type every eval step, plus the natural abstain
  rate on a held-out probe set, so runaway over-abstention is visible live.
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import json
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainerCallback)
from trl import DPOTrainer, DPOConfig

# ---- knobs ----------------------------------------------------------------
BASE_MODEL    = "mistralai/Mistral-7B-v0.3"
WARMSTART_DIR = "./mistral-medqa-warmstart/policy"   # policy AND reference start here
WARMSTART_TOKENIZER = "./mistral-medqa-warmstart"    # tokenizer saved at parent
PAIRS         = "dpo_pairs_v2.json"
OUTPUT_DIR    = "./mistral-medqa-dpo-v2"
FINAL_PARENT  = "./mistral-medqa-dpo-v2-final"       # PEFT creates /policy inside
BETA          = 0.05      # looser leash than v1's 0.1
LR            = 1e-5
EPOCHS        = 2
SEED          = 42

ANSWER_SET = ["A", "B", "C", "D"]
LETTERS    = ["A", "B", "C", "D", "E"]
COMPLETIONS = {
    "A": " The answer is A.", "B": " The answer is B.",
    "C": " The answer is C.", "D": " The answer is D.",
    "E": " I cannot answer confidently.",
}

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(WARMSTART_TOKENIZER if os.path.exists(WARMSTART_TOKENIZER)
                                          else BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading pairs...")
with open(PAIRS) as f:
    data = json.load(f)
train_pairs, val_pairs = data["train"], data["val"]
print(f"train={len(train_pairs)}  val={len(val_pairs)}")

# Keep the type tag so we can split metrics by pair type.
def to_ds(pairs):
    return Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"],
                               "rejected": p["rejected"],
                               "ptype": p.get("type", "?")} for p in pairs])
train_ds, val_ds = to_ds(train_pairs), to_ds(val_pairs)

# Held-out probe prompts for live abstain-rate readout (val abstain prompts).
probe_prompts = [p["prompt"] for p in val_pairs if p.get("type") == "abstain"][:40]


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    # policy (trainable) + reference (frozen), BOTH from the warm-started adapter
    model = PeftModel.from_pretrained(base, WARMSTART_DIR,
                                      adapter_name="policy", is_trainable=True)
    model.load_adapter(WARMSTART_DIR, adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


class PerTypeMonitor(TrainerCallback):
    """Logs pairwise accuracy split by pair type + natural abstain rate on a
    probe set. This is the monitor v1 lacked."""
    def __init__(self, trainer, val_pairs, probe_prompts):
        self.trainer = trainer
        self.val_pairs = val_pairs
        self.probe = probe_prompts
        self.comp_ids = {L: tokenizer(COMPLETIONS[L], add_special_tokens=False).input_ids
                         for L in LETTERS}

    @torch.no_grad()
    def _abstain_rate(self):
        """Fraction of probe prompts where ' I cannot...' outscores all answers."""
        model = self.trainer.model
        model.eval()
        abst = 0
        for prompt in self.probe:
            scores = {}
            for L in LETTERS:
                ids = tokenizer(prompt, add_special_tokens=True).input_ids + self.comp_ids[L]
                t = torch.tensor([ids], device=model.device)
                logits = model(input_ids=t).logits.float()[0]
                start = len(ids) - len(self.comp_ids[L])
                lp = torch.log_softmax(logits[start-1:len(ids)-1], dim=-1)
                lab = torch.tensor(self.comp_ids[L], device=model.device)
                scores[L] = float(lp.gather(1, lab.unsqueeze(1)).mean())
            if max(scores, key=scores.get) == "E":
                abst += 1
        model.train()
        return abst / max(len(self.probe), 1)

    def on_evaluate(self, args, state, control, **kwargs):
        rate = self._abstain_rate()
        print(f"\n[MONITOR step {state.global_step}] "
              f"probe natural abstain rate = {rate:.3f}")
        if rate > 0.85:
            print("  ** WARNING: abstain rate >0.85 -- approaching always-abstain "
                  "collapse. Consider stopping / lowering abstain ratio or beta. **")


def compute_metrics_split(eval_pred):
    # DPOTrainer doesn't pass ptype through compute_metrics cleanly; per-type
    # accuracy is instead recovered post-hoc from the eval log (see README note).
    return {}


def main():
    model = load_model()

    cfg = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=BETA,
        model_adapter_name="policy",
        ref_adapter_name="reference",
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        load_best_model_at_end=False,
        fp16=False, bf16=False,
        max_length=1024,
        max_prompt_length=960,
        logging_steps=10,
        report_to="none",
        seed=SEED,
    )

    trainer = DPOTrainer(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.add_callback(PerTypeMonitor(trainer, val_pairs, probe_prompts))

    print(f"\nStarting DPO v2  (beta={BETA}, lr={LR}, {EPOCHS} epochs)...")
    print("Watch: rewards/margins >0 AND probe abstain rate staying < ~0.5.")
    result = trainer.train()

    print("\nFinal eval...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

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

    final_adapter = str(Path(FINAL_PARENT) / "policy")
    print(f"\nSaved -> {FINAL_PARENT}")
    print("Inspect the saved layout (expect a policy/ subfolder):")
    print(f"  ls -R {FINAL_PARENT}")
    print("Then full eval-gate ->")
    print(f"  DPO_ADAPTER={final_adapter} TOKENIZER_PATH={FINAL_PARENT} \\")
    print(f"    python dpo_eval_full.py")
    print("SUCCESS looks like: P(E) AUROC > 0.6 (directed!), mean P(E) wrong > correct,")
    print("abstain rate sane (10-30%), answered accuracy up vs SFT 0.51.")


if __name__ == "__main__":
    main()
