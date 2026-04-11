![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-Mistral--7B-orange)
![Task](https://img.shields.io/badge/Task-MedQA-green)

# Mistral-7B MedQA Abstention — Selective Prediction for Reliable Medical QA

A safety-focused fine-tuning project that enables Mistral-7B to abstain from answering
when uncertain, reducing high-confidence errors in medical QA.

> 🚀 Improves reliability from 52.2% → 70.3% using confidence-based selective prediction

---

## 🎯 Problem

Standard fine-tuning optimizes for accuracy but can make models **overconfident in wrong answers**.
In medical AI, a confident wrong answer is more dangerous than no answer at all.

This project addresses that by adding an **abstention mechanism** — the model refuses to answer
when its confidence falls below a threshold, reducing wrong answer rates while maintaining
controllable coverage.

---

## 📊 Results

### Baseline vs Fine-Tuned (No Abstention)

| Model | Accuracy | Coverage | Wrong Answer Rate |
|-------|----------|----------|-------------------|
| Mistral-7B (base) | 49.33% | 100% | 50.67% |
| Mistral-7B (fine-tuned) | 52.24% | 100% | 47.76% |

### Fine-Tuned with Abstention — Threshold Analysis

| Threshold | Answered Accuracy | Coverage | Wrong Rate | Abstained |
|-----------|-------------------|----------|------------|-----------|
| 0.00 | 52.24% | 100.00% | 47.76% | 0.00% |
| 0.30 | 53.05% | 96.70% | 45.40% | 3.30% |
| 0.35 | 55.33% | 85.47% | 38.18% | 14.53% |
| 0.40 | 59.89% | 70.70% | 28.36% | 29.30% |
| 0.45 | 64.62% | 57.50% | 20.35% | 42.50% |
| **0.50** | **70.33%** | **45.80%** | **13.59%** | **54.20%** |
| 0.55 | 74.57% | 36.14% | 9.19% | 63.86% |
| 0.60 | 75.69% | 28.44% | 6.91% | 71.56% |
| 0.65 | 79.32% | 23.17% | 4.79% | 76.83% |
| 0.70 | 84.89% | 17.67% | 2.67% | 82.33% |
| 0.75 | 88.95% | 14.22% | 1.57% | 85.78% |
| 0.80 | 91.24% | 10.76% | 0.94% | 89.24% |
| 0.90 | 98.25% | 4.48% | 0.08% | 95.52% |

### 🏆 Sweet Spot (threshold = 0.50)

| Metric | No Abstention | With Abstention |
|--------|---------------|-----------------|
| Overall Accuracy | 52.24% | — |
| Answered Accuracy | 52.24% | 70.33% |
| Coverage | 100% | 45.80% |
| Wrong Answer Rate | 47.76% | 13.59% |

At comparable thresholds, the fine-tuned model achieves higher answered accuracy at higher
coverage, indicating improved confidence calibration and more reliable selective prediction.

---

## 💡 Key Insight

Fine-tuning not only improves raw accuracy (+2.9%), but also makes confidence scores
better aligned with prediction correctness (improved calibration), enabling reliable
abstention decisions based on calibrated confidence. This leads to significantly better
accuracy–coverage tradeoffs compared to the base model.

---

## 🧠 Concepts Demonstrated

- Parameter-efficient fine-tuning (QLoRA)
- Selective prediction (accuracy–coverage tradeoff)
- Confidence calibration for LLMs
- Reliability-aware evaluation beyond accuracy

---

## 🏗️ Architecture
```
Base Model : mistralai/Mistral-7B-v0.3
Method     : QLoRA (4-bit quantization + LoRA adapters)
LoRA rank  : r=16, alpha=32
Target     : q_proj, v_proj
Training   : Early stopping (patience=3, monitoring eval_loss)
Dataset    : GBaker/MedQA-USMLE-4-options
```

---

## 📁 Project Structure
```
mistral-medqa-abstention/
│
├── baseline_eval.py          # Evaluate base Mistral-7B on MedQA test set
├── train_lora.py             # Fine-tune with QLoRA + early stopping
├── finetuned_eval.py         # Evaluate fine-tuned model on MedQA test set
├── abstention_analysis.py    # Sweep thresholds, compute abstention metrics
│
├── baseline_results.json     # Baseline evaluation results + confidence scores
├── finetuned_results.json    # Fine-tuned evaluation results + confidence scores
├── abstention_results.json   # Full threshold analysis results
│
└── README.md
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets accelerate bitsandbytes huggingface_hub trl
```

### 2. Run Baseline Evaluation
```bash
python baseline_eval.py
```

### 3. Fine-Tune with Early Stopping
```bash
# Smoke test first
# Set SMOKE_TEST = True in train_lora.py
python train_lora.py

# Full training
# Set SMOKE_TEST = False in train_lora.py
python train_lora.py
```

### 4. Evaluate Fine-Tuned Model
```bash
python finetuned_eval.py
```

### 5. Run Abstention Analysis
```bash
python abstention_analysis.py
```

---

## 🔧 Load Fine-Tuned Model
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Primeinvincible/mistral-medqa-lora-v3")

# Load base model in 4-bit
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

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "Primeinvincible/mistral-medqa-lora-v3")
model.eval()
```

---

## 💡 Key Design Decisions

**Why abstention over accuracy maximization?**
In medical AI, a confident wrong answer is worse than no answer. A model that
abstains on uncertain cases is safer than one that always answers.

**Why confidence thresholding?**
We extract softmax probabilities over the four answer options (A/B/C/D) at the
last token position. If the highest probability falls below the threshold,
the model abstains. This is simple, interpretable, and requires no additional training.

**Why early stopping?**
Previous experiments showed overfitting after 2+ epochs (eval loss climbed while
train loss dropped). Early stopping with patience=3 automatically stops training
when validation loss stops improving.

**Why QLoRA?**
Fine-tuning a 7B model requires significant GPU memory. QLoRA reduces memory usage
by quantizing the base model to 4-bit while training only the small LoRA adapter
weights (~0.09% of total parameters).

---

## 📈 How to Choose a Threshold

| Use Case | Recommended Threshold | Reasoning |
|----------|-----------------------|-----------|
| High safety (triage) | 0.70+ | Minimize wrong answers, accept low coverage |
| Balanced | 0.50 | Best accuracy/coverage tradeoff |
| High coverage | 0.35 | Answer more questions, accept more errors |

---

## 🗂️ Dataset

**GBaker/MedQA-USMLE-4-options**
- 10,178 training examples
- 1,273 test examples
- Source: USMLE Step 1/2/3 medical licensing exam questions
- Format: 4-option multiple choice

Train/val split: 90/10 from training set (seed=42).
Official test set kept untouched for final evaluation.

---

## 🖥️ Hardware

- GPU: NVIDIA Tesla V100S-PCIE-32GB
- Training time: ~1.5 hours (early stopping at epoch 1.4)
- Inference time: ~4 minutes for full test set (1,273 examples)

---

## 🔗 Model on HuggingFace

[Primeinvincible/mistral-medqa-lora-v3](https://huggingface.co/Primeinvincible/mistral-medqa-lora-v3)

---

## 👤 Author

Master's student specializing in NLP/LLM safety and production deployment.
Part of a portfolio focused on making LLM systems reliable for real-world use.