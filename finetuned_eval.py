"""
finetuned_eval.py
-----------------
Evaluate fine-tuned Mistral-7B-v0.3 (mistral-medqa-lora-v3) on MedQA USMLE test set.
Uses same evaluation method as baseline_eval.py for fair comparison.
Saves confidence scores for abstention layer analysis.

Dataset : GBaker/MedQA-USMLE-4-options (1,273 test examples)
Model   : Primeinvincible/mistral-medqa-lora-v3 (4-bit quantized + LoRA adapters)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import json

# ── DEBUG FLAG ─────────────────────────────────────────────────────────────────
# Set DEBUG = True to run on 10 examples only and print detailed output
# Set DEBUG = False for full evaluation (1,273 examples)
DEBUG = False

# ── 1. Load Tokenizer ──────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Primeinvincible/mistral-medqa-lora-v3")

# ── 2. Load Base Model with 4-bit Quantization ────────────────────────────────
print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)

# ── 3. Load LoRA Adapters ──────────────────────────────────────────────────────
# Load the fine-tuned LoRA adapters on top of the base model
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, "Primeinvincible/mistral-medqa-lora-v3")
model.eval()
print("Model loaded!")

# ── 4. Format Prompt ───────────────────────────────────────────────────────────
def format_prompt(example):
    """
    Format prompt — must match training format exactly.
    Options ordered A→B→C→D, no trailing space after 'Answer:'.
    """
    options_str = "\n".join([f"{k}: {example['options'][k]}" for k in ["A", "B", "C", "D"]])
    return f"Question: {example['question']}\n\nOptions:\n{options_str}\n\nAnswer:"

# ── 5. Verify A/B/C/D Token IDs (Prompt-Aware) ────────────────────────────────
def get_answer_token_ids(tokenizer, sample_prompt):
    """
    Verify each option is a single-token continuation after the actual prompt.
    """
    answer_token_ids = {}
    base_ids = tokenizer(sample_prompt, add_special_tokens=False).input_ids

    print("\nVerifying answer option tokenization (prompt-aware)...")
    for opt in ["A", "B", "C", "D"]:
        full_ids = tokenizer(sample_prompt + " " + opt, add_special_tokens=False).input_ids
        continuation_ids = full_ids[len(base_ids):]

        print(f"  Option '{opt}' continuation IDs: {continuation_ids}")

        if len(continuation_ids) != 1:
            raise ValueError(
                f"Option {opt} is not a single-token continuation: {continuation_ids}."
            )
        answer_token_ids[opt] = continuation_ids[0]

    print("All options are single-token continuations. ✓\n")
    return answer_token_ids

# ── 6. Predict Answer with Confidence ─────────────────────────────────────────
def predict_answer(model, tokenizer, prompt, answer_token_ids, debug=False):
    """
    Run inference and return predicted answer with confidence scores.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]

    option_order = ["A", "B", "C", "D"]
    option_id_tensor = torch.tensor(
        [answer_token_ids[opt] for opt in option_order],
        device=last_logits.device
    )
    option_logits = last_logits[option_id_tensor]
    option_probs = torch.softmax(option_logits, dim=0)

    best_idx = torch.argmax(option_probs).item()
    best_answer = option_order[best_idx]
    confidence = option_probs[best_idx].item()
    all_probs = {opt: option_probs[i].item() for i, opt in enumerate(option_order)}

    if debug:
        print(f"  All probs → A: {all_probs['A']:.3f} | B: {all_probs['B']:.3f} | "
              f"C: {all_probs['C']:.3f} | D: {all_probs['D']:.3f}")
        print(f"  Predicted: {best_answer} (confidence: {confidence:.3f})")

    return best_answer, confidence, all_probs

# ── 7. Load Dataset & Get Token IDs ───────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
print(f"Test set size: {len(dataset)} examples")

sample_prompt = format_prompt(dataset[0])
answer_token_ids = get_answer_token_ids(tokenizer, sample_prompt)

# ── 8. Select Subset if Debug ──────────────────────────────────────────────────
if DEBUG:
    dataset = dataset.select(range(10))
    print(f"DEBUG MODE: Running on {len(dataset)} examples only\n")
else:
    print(f"FULL MODE: Running on all {len(dataset)} examples\n")

# ── 9. Evaluate ────────────────────────────────────────────────────────────────
correct = 0
results = []

for i, example in enumerate(tqdm(dataset, desc="Evaluating fine-tuned model")):
    prompt = format_prompt(example)

    if DEBUG:
        print(f"\n[Example {i+1}]")
        print(f"  Question : {example['question'][:80]}...")
        print(f"  Answer   : {example['answer_idx']}")

    prediction, confidence, all_probs = predict_answer(
        model, tokenizer, prompt, answer_token_ids, debug=DEBUG
    )
    is_correct = (prediction == example["answer_idx"])
    correct += int(is_correct)

    if DEBUG:
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"  Result   : {status}")

    results.append({
        "question"     : example["question"],
        "prediction"   : prediction,
        "ground_truth" : example["answer_idx"],
        "is_correct"   : is_correct,
        "confidence"   : confidence,
        "all_probs"    : all_probs,
    })

# ── 10. Report & Save Results ──────────────────────────────────────────────────
accuracy = correct / len(results)
output_file = "finetuned_results_debug.json" if DEBUG else "finetuned_results.json"

print(f"\n{'='*40}")
print(f"Mode              : {'DEBUG (10 examples)' if DEBUG else 'FULL (1273 examples)'}")
print(f"Finetuned Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correct           : {correct} / {len(results)}")
print(f"{'='*40}")

# ── 11. Compare with Baseline ──────────────────────────────────────────────────
if not DEBUG:
    try:
        with open("baseline_results.json", "r") as f:
            baseline = json.load(f)
        print(f"\n{'='*40}")
        print(f"COMPARISON")
        print(f"{'='*40}")
        print(f"Baseline  : {baseline['accuracy']*100:.2f}%")
        print(f"Fine-tuned: {accuracy*100:.2f}%")
        print(f"Improvement: +{(accuracy - baseline['accuracy'])*100:.2f}%")
        print(f"{'='*40}")
    except FileNotFoundError:
        print("baseline_results.json not found — skipping comparison")

with open(output_file, "w") as f:
    json.dump({
        "mode"        : "debug" if DEBUG else "full",
        "accuracy"    : accuracy,
        "correct"     : correct,
        "total"       : len(results),
        "predictions" : results,
    }, f, indent=2)

print(f"Saved {output_file}")
