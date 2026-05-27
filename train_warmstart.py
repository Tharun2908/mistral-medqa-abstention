"""
train_warmstart.py
------------------
Warm-start SFT to lift the abstain sentence off zero probability before DPO v2.

Root cause being fixed: DPO v1 failed because the abstain completion had almost
no probability mass (max P(E)=0.0225). DPO amplifies relative preferences but
cannot create a near-unseen behavior from zero against a KL leash. This pass
teaches the model that the abstain string is a legal completion, lifting its
baseline probability so DPO v2 has mass to work with.

Key design points:
  * CONTINUES the existing SFT adapter (mistral-medqa-lora-v3) -- keeps all the
    learned MedQA ability. Saves as a NEW adapter so v3 stays intact.
  * COMPLETION-ONLY loss: prompt tokens are masked (-100); the model is trained
    only to produce the answer/abstain sentence given the prompt, not to
    reproduce the prompt.
  * EXACT same prompt+completion format as dpo_eval_full.py / build_dpo_pairs_v2
    -- no format drift.
  * LIGHT touch: 1 epoch, LR 5e-5. Goal is to lift max P(E) into ~0.10-0.30,
    NOT to master abstention (that's DPO's job). Eval-gate after, before DPO.

Output: ./mistral-medqa-warmstart/policy  (layout matches dpo_eval_full.py path)
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
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)

# ---- knobs ----------------------------------------------------------------
BASE_MODEL  = "mistralai/Mistral-7B-v0.3"
SFT_ADAPTER = "Primeinvincible/mistral-medqa-lora-v3"   # continue THIS
WARMSTART_DATA = "warmstart_data.json"
OUTPUT_DIR  = "./mistral-medqa-warmstart"
SAVE_DIR    = "./mistral-medqa-warmstart/policy"        # eval path expects /policy
LR          = 3e-5     # gentler: v1 warm-start (5e-5) overshot to 33% abstain;
                       # aiming for ~8-10% so DPO only has to DIRECT abstention.
EPOCHS      = 1
MAX_LEN     = 1024
SEED        = 42

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def build_example(rec):
    """Tokenize prompt+completion with completion-only labels.

    prompt comes pre-formatted in warmstart_data.json (built by
    build_dpo_pairs_v2.py with the exact eval format). We append EOS so the
    model learns to STOP after the sentence rather than ramble.
    """
    prompt = rec["prompt"]
    completion = rec["completion"] + tokenizer.eos_token

    p_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids

    input_ids = (p_ids + c_ids)[:MAX_LEN]
    # Mask the prompt: loss only on completion tokens.
    labels = ([-100] * len(p_ids) + c_ids)[:MAX_LEN]
    return {"input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels}


def main():
    print("Loading warm-start data...")
    with open(WARMSTART_DATA) as f:
        warm = json.load(f)
    n_abstain = sum(1 for w in warm if w.get("type") == "abstain")
    print(f"{len(warm)} examples  (abstain={n_abstain}, "
          f"answer={len(warm)-n_abstain}, abstain frac={n_abstain/len(warm):.2f})")

    ds = Dataset.from_list([build_example(r) for r in warm])

    # Sanity: confirm completion-only masking on the first example.
    ex = ds[0]
    n_supervised = sum(1 for x in ex["labels"] if x != -100)
    print(f"First example: {len(ex['input_ids'])} tokens, "
          f"{n_supervised} supervised (completion-only).")
    assert n_supervised < len(ex["input_ids"]), "prompt not masked!"

    print("\nLoading base model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    print("Loading SFT adapter as TRAINABLE (continue training v3)...")
    model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
    model.config.use_cache = False
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, padding=True,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",          # we save manually at the end
        fp16=False, bf16=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=ds, data_collator=collator,
    )

    print(f"\nWarm-start: {EPOCHS} epoch, LR {LR} ...")
    result = trainer.train()
    print("train metrics:", result.metrics)

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)   # tokenizer at parent, matches eval
    with open(Path(OUTPUT_DIR) / "warmstart_metrics.json", "w") as f:
        json.dump(result.metrics, f, indent=2)
    print(f"\nSaved adapter -> {SAVE_DIR}")
    print("Next: eval-gate before DPO:")
    print(f"  DPO_ADAPTER={SAVE_DIR} TOKENIZER_PATH={OUTPUT_DIR} \\")
    print(f"    python dpo_eval_full.py --limit 200")
    print("Check max P(E) (~0.10-0.30 = good), abstain rate (2-10% = good),")
    print("answered accuracy (should NOT collapse from ~0.51).")


if __name__ == "__main__":
    main()
