# grpo_v4_full.py
"""
Project 2 v4.2 — GRPO/RLVR with G=4 on H200.
v4.1 (G=2 on A100-40GB) had corrected reward shape but plateaued with P(E) AUROC ~0.39:
G=2 advantages are bimodal/zero, providing weak per-example direction signal.
H200 with 143GB VRAM removes the G=2 memory constraint. Same reward (+1/+0.3/-2),
same LR (3e-6), same 1 epoch — only G=2 -> G=4 changes.
"""
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import json
import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from dpo_eval_full import build_prompt
from reward_fn import classify_completion, reward_single

BASE       = "mistralai/Mistral-7B-v0.3"
WARMSTART  = "/workspace/mistral-medqa-warmstart/policy"
OUTPUT_DIR = "/workspace/mistral-medqa-grpo-v4.2"
N_TRAIN    = 2160
SEED       = 42

print("Loading base bf16 + warm-start, merging in memory...")
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, WARMSTART).merge_and_unload()
print("Merged model ready.")

full = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=SEED)
train = full.select(range(N_TRAIN))
dataset = Dataset.from_list([
    {"prompt": build_prompt(ex["question"], ex["options"]), "ground_truth": ex["answer_idx"]}
    for ex in train
])
print(f"Train: {len(dataset)} prompts. First gt={dataset[0]['ground_truth']}")

_log = {"n": 0}
def logging_reward(prompts=None, completions=None, ground_truth=None, **kwargs):
    types = {"correct": 0, "wrong": 0, "abstain": 0, "malformed": 0}
    rewards = []
    for c, gt in zip(completions, ground_truth):
        types[classify_completion(c, gt)] += 1
        rewards.append(reward_single(c, gt))
    r = np.array(rewards)
    _log["n"] += 1
    if _log["n"] % 5 == 0 or _log["n"] <= 5:
        print(f"[reward {_log['n']}] mean={r.mean():.3f} std={r.std():.3f} types={types}")
    return rewards

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM",
)

cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=16,
    max_prompt_length=384,
    temperature=1.0,
    beta=0.0,
    learning_rate=3e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=1,                 # v4.1: 1 epoch (~270 steps); extend only if trajectory not converged
    logging_steps=1,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=None,              # KEEP ALL for post-hoc selection
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    report_to="none",
    seed=SEED,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[logging_reward],
    args=cfg,
    train_dataset=dataset,
    processing_class=tok,
    peft_config=lora,
)

print(f"\n=== GRPO v4 full run starting ===")
print(f"Output: {OUTPUT_DIR}  (checkpoints every 25 steps, all kept)")
result = trainer.train()

print("\n=== GRPO v4 COMPLETE ===")
print(result.metrics)

with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w") as f:
    json.dump(result.metrics, f, indent=2)
print(f"Metrics saved to {OUTPUT_DIR}/train_metrics.json")

# Explicit final save: save_steps=25 skips the last 15 steps of a 540-step run.
# This captures the exact final adapter as one more candidate for post-hoc selection
# (not assumed best -- the trajectory eval picks the headline).
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")
trainer.save_model(FINAL_DIR)
tok.save_pretrained(FINAL_DIR)
print(f"Final adapter + tokenizer saved to {FINAL_DIR}")
