"""
predict.py
----------
Run inference on a single MedQA-style question using the fine-tuned model.
Returns prediction, confidence, and abstention decision.

Usage:
  python predict.py \
    --question "A patient presents with..." \
    --option_a "Aspirin" \
    --option_b "Ibuprofen" \
    --option_c "Acetaminophen" \
    --option_d "Morphine" \
    --threshold 0.50

Output:
  {
    "prediction": "A",
    "confidence": 0.72,
    "abstained": false,
    "threshold": 0.50,
    "all_probs": {"A": 0.72, "B": 0.15, "C": 0.08, "D": 0.05}
  }
"""

import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── 1. Argument Parser ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MedQA inference with abstention")
parser.add_argument("--question",   required=True,  help="Question text")
parser.add_argument("--option_a",   required=True,  help="Option A text")
parser.add_argument("--option_b",   required=True,  help="Option B text")
parser.add_argument("--option_c",   required=True,  help="Option C text")
parser.add_argument("--option_d",   required=True,  help="Option D text")
parser.add_argument("--threshold",  type=float, default=0.50,
                    help="Abstention threshold (default: 0.50)")
parser.add_argument("--model_id",   default="Primeinvincible/mistral-medqa-lora-v3",
                    help="HuggingFace model ID for LoRA adapter")
parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.3",
                    help="Base model ID")
args = parser.parse_args()

# ── 2. Load Tokenizer ──────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.base_model)

# ── 3. Load Model ──────────────────────────────────────────────────────────────
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, args.model_id)
model.eval()
print("Model loaded!\n")

# ── 4. Verify Answer Token IDs ─────────────────────────────────────────────────
def get_answer_token_ids(tokenizer, sample_prompt):
    """
    Verify each option A/B/C/D is a single-token continuation
    after the actual prompt — prompt-aware verification.
    """
    answer_token_ids = {}
    base_ids = tokenizer(sample_prompt, add_special_tokens=False).input_ids

    for opt in ["A", "B", "C", "D"]:
        full_ids = tokenizer(sample_prompt + " " + opt, add_special_tokens=False).input_ids
        continuation_ids = full_ids[len(base_ids):]
        if len(continuation_ids) != 1:
            raise ValueError(
                f"Option {opt} is not a single-token continuation: {continuation_ids}"
            )
        answer_token_ids[opt] = continuation_ids[0]

    return answer_token_ids

# ── 5. Format Prompt ───────────────────────────────────────────────────────────
def format_prompt(question, options):
    """
    Format prompt — identical to training and evaluation format.
    Options ordered A→B→C→D explicitly.
    """
    options_str = "\n".join([f"{k}: {options[k]}" for k in ["A", "B", "C", "D"]])
    return f"Question: {question}\n\nOptions:\n{options_str}\n\nAnswer:"

# ── 6. Predict ─────────────────────────────────────────────────────────────────
def predict(model, tokenizer, prompt, answer_token_ids, threshold):
    """
    Run inference and apply abstention threshold.
    Returns prediction, confidence, abstention decision, all probs.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]

    # Extract logits for A, B, C, D only
    option_order = ["A", "B", "C", "D"]
    option_id_tensor = torch.tensor(
        [answer_token_ids[opt] for opt in option_order],
        device=last_logits.device
    )
    option_logits = last_logits[option_id_tensor]

    # Softmax over 4 options → normalized confidence scores
    option_probs = torch.softmax(option_logits, dim=0)

    best_idx      = torch.argmax(option_probs).item()
    best_answer   = option_order[best_idx]
    confidence    = option_probs[best_idx].item()
    all_probs     = {opt: round(option_probs[i].item(), 4) for i, opt in enumerate(option_order)}

    # Apply abstention threshold
    if confidence < threshold:
        return {
            "prediction" : None,
            "confidence" : round(confidence, 4),
            "abstained"  : True,
            "threshold"  : threshold,
            "all_probs"  : all_probs,
            "message"    : "Abstained due to low confidence"
        }

    return {
        "prediction" : best_answer,
        "confidence" : round(confidence, 4),
        "abstained"  : False,
        "threshold"  : threshold,
        "all_probs"  : all_probs,
        "message"    : f"Answered with {confidence*100:.1f}% confidence"
    }

# ── 7. Run Inference ───────────────────────────────────────────────────────────
options = {
    "A": args.option_a,
    "B": args.option_b,
    "C": args.option_c,
    "D": args.option_d,
}

prompt = format_prompt(args.question, options)
answer_token_ids = get_answer_token_ids(tokenizer, prompt)
result = predict(model, tokenizer, prompt, answer_token_ids, args.threshold)

# ── 8. Print Result ────────────────────────────────────────────────────────────
print("=" * 50)
print("QUESTION:")
print(f"  {args.question}")
print("\nOPTIONS:")
for k, v in options.items():
    print(f"  {k}: {v}")
print("\nRESULT:")
print(json.dumps(result, indent=2))
print("=" * 50)
