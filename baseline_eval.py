"""
baseline_eval.py
----------------
Evaluate base Mistral-7B-v0.3 (NO fine-tuning) on MedQA USMLE test set.
Uses next-token logits over A/B/C/D options — consistent with fine-tuned evaluation.
Saves confidence scores for later use in abstention/safety layer.

Dataset : GBaker/MedQA-USMLE-4-options (1,273 test examples)
Model   : mistralai/Mistral-7B-v0.3 (4-bit quantized, no LoRA)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import json

# ── DEBUG FLAG ─────────────────────────────────────────────────────────────────
# Set DEBUG = True to run on 10 examples only and print detailed output
# Set DEBUG = False for full evaluation (1,273 examples, ~21 mins)
DEBUG = False

# ── 1. Load Model & Tokenizer ──────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()
print("Model loaded!")

# ── 2. Format Prompt ───────────────────────────────────────────────────────────
def format_prompt(example):
    """
    Format a MedQA example into a prompt string.
    No trailing space — the natural continuation after 'Answer:' is ' A'/' B' etc.
    The space before the answer letter is handled in get_answer_token_ids().
    Must match the format used during fine-tuning exactly.
    Options explicitly ordered A→B→C→D for deterministic prompt format.
    """
    options_str = "\n".join([f"{k}: {example['options'][k]}" for k in ["A", "B", "C", "D"]])
    return f"Question: {example['question']}\n\nOptions:\n{options_str}\n\nAnswer:"

# ── 3. Verify A/B/C/D Token IDs (Prompt-Aware) ────────────────────────────────
def get_answer_token_ids(tokenizer, sample_prompt):
    """
    Verify each option A/B/C/D is a single-token continuation AFTER
    the actual prompt format — not just in isolation.

    We check ' A', ' B', ' C', ' D' (with leading space) because that is
    the natural continuation after 'Answer:' — the tokenizer merges
    the space with the letter into one token e.g. ' A' → single token.
    """
    answer_token_ids = {}
    base_ids = tokenizer(sample_prompt, add_special_tokens=False).input_ids

    print("\nVerifying answer option tokenization (prompt-aware)...")
    for opt in ["A", "B", "C", "D"]:
        # Use space + letter as continuation — natural after "Answer:"
        full_ids = tokenizer(sample_prompt + " " + opt, add_special_tokens=False).input_ids
        continuation_ids = full_ids[len(base_ids):]

        print(f"  Option '{opt}' continuation IDs: {continuation_ids}")

        if len(continuation_ids) != 1:
            raise ValueError(
                f"Option {opt} is not a single-token continuation: {continuation_ids}. "
                f"Next-token logit method is invalid."
            )
        answer_token_ids[opt] = continuation_ids[0]

    print("All options are single-token continuations. ✓\n")
    return answer_token_ids

# ── 4. Predict Answer with Confidence ─────────────────────────────────────────
def predict_answer(model, tokenizer, prompt, answer_token_ids, debug=False):
    """
    Run inference and return:
      - best_answer : predicted option (A/B/C/D)
      - confidence  : normalized probability of best answer over options
      - all_probs   : normalized probabilities for all four options

    NOTE: These are relative confidence scores over A/B/C/D,
    not fully calibrated probabilities. Useful for abstention thresholding.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at last token position → shape: [vocab_size]
    last_logits = outputs.logits[0, -1, :]

    # Extract logits for A, B, C, D — keep everything in tensor form
    option_order = ["A", "B", "C", "D"]
    option_id_tensor = torch.tensor(
        [answer_token_ids[opt] for opt in option_order],
        device=last_logits.device
    )
    option_logits = last_logits[option_id_tensor]

    # Softmax over 4 options → normalized option probabilities
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

# ── 5. Load Dataset & Get Token IDs ───────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
print(f"Test set size: {len(dataset)} examples")

# Use a real prompt for token ID verification — not just "A" in isolation
sample_prompt = format_prompt(dataset[0])
answer_token_ids = get_answer_token_ids(tokenizer, sample_prompt)

# ── 6. Select Subset if Debug ──────────────────────────────────────────────────
if DEBUG:
    dataset = dataset.select(range(10))
    print(f"DEBUG MODE: Running on {len(dataset)} examples only\n")
else:
    print(f"FULL MODE: Running on all {len(dataset)} examples\n")

# ── 7. Evaluate ────────────────────────────────────────────────────────────────
correct = 0
results = []

for i, example in enumerate(tqdm(dataset, desc="Evaluating baseline")):
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

# ── 8. Report & Save Results ───────────────────────────────────────────────────
accuracy = correct / len(results)
output_file = "baseline_results_debug.json" if DEBUG else "baseline_results.json"

print(f"\n{'='*40}")
print(f"Mode              : {'DEBUG (10 examples)' if DEBUG else 'FULL (1273 examples)'}")
print(f"Baseline Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correct           : {correct} / {len(results)}")
print(f"{'='*40}")

with open(output_file, "w") as f:
    json.dump({
        "mode"        : "debug" if DEBUG else "full",
        "accuracy"    : accuracy,
        "correct"     : correct,
        "total"       : len(results),
        "predictions" : results,
    }, f, indent=2)

print(f"Saved {output_file}")
