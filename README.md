![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-Mistral--7B-orange)
![Task](https://img.shields.io/badge/Task-MedQA-green)

# Mistral-7B MedQA Abstention — Selective Prediction for Reliable Medical QA

A safety-focused fine-tuning project that enables Mistral-7B to abstain from answering
when uncertain, reducing high-confidence errors in medical QA.

> 🚀 Selective prediction raises answered-question accuracy from 52.2% to 70.3% at 45.8% coverage, reducing total wrong-answer rate from 47.8% to 13.6%

---

## 🎯 Problem

Standard fine-tuning optimizes for accuracy but can make models **overconfident in wrong answers**.
In medical AI, a confident wrong answer is more dangerous than no answer at all.

This project addresses that by adding an **abstention mechanism** — an external abstention rule suppresses the model's answer when confidence falls below a threshold, reducing wrong answer rates while maintaining
controllable coverage.

---

## 📊 Results

### Baseline vs Fine-Tuned (No Abstention)

| Model | Accuracy | Coverage | Wrong Answer Rate |
|-------|----------|----------|-------------------|
| Mistral-7B (base) | 49.33% | 100% | 50.67% |
| Mistral-7B (fine-tuned) | 52.24% | 100% | 47.76% |

### Fine-Tuned with Abstention — Threshold Analysis

| Threshold | Answered Accuracy | Coverage | Dataset-Level Wrong Rate | Abstained |
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

### 🏆 Balanced Operating Point: threshold = 0.50

| Metric | No Abstention | With Abstention |
|--------|---------------|-----------------|
| Overall Accuracy | 52.24% | — |
| Answered Accuracy | 52.24% | 70.33% |
| Coverage | 100% | 45.80% |
| Wrong Answer Rate | 47.76% | 13.59% |



**Threshold Selection Criterion:**
We choose threshold = 0.50 as the balanced operating point because it reduces the
total wrong-answer rate from 47.76% to 13.59% while preserving nearly half of
test-set coverage. Higher thresholds reduce errors further but abstain on more than 70–90% of questions, which may be unsuitable for use cases requiring broad coverage.

At comparable thresholds, the fine-tuned model achieves higher answered accuracy
at higher coverage than the base model, indicating more reliable selective prediction.

---

### 📈 Matched-Coverage Comparison — Baseline vs Fine-Tuned

Comparing at fixed thresholds is unfair because baseline and fine-tuned models
answer different numbers of questions at the same threshold. Matched-coverage
comparison ensures we compare equal-sized subsets.

| Coverage | Base Accuracy | FT Accuracy | Gain | Base Wrong | FT Wrong |
|----------|---------------|-------------|------|------------|----------|
| 90% | 51.51% | 54.40% | +2.89% | 42.97% | 41.08% |
| 80% | 53.18% | 56.87% | +3.69% | 37.55% | 34.25% |
| 70% | 54.92% | 59.89% | +4.97% | 32.05% | 28.36% |
| 60% | 57.25% | 63.85% | +6.60% | 26.16% | 21.52% |
| 50% | 61.90% | 68.40% | +6.50% | 18.85% | 15.71% |
| 40% | 67.07% | 72.73% | +5.66% | 12.80% | 10.84% |
| 30% | 69.67% | 74.93% | +5.26% | 9.27% | 7.38% |
| 20% | 77.95% | 83.27% | +5.32% | 4.40% | 3.30% |

**Fine-tuned wins at all 8 coverage levels.**
Average accuracy gain: +5.11% | Average wrong rate reduction: -2.70%

Fine-tuning improves selective prediction (+5.11% average) more than raw accuracy
(+2.91%), indicating the model's confidence scores become more reliable after
fine-tuning — not just its predictions.

---

## 💡 Key Insight

Fine-tuning improves raw accuracy by +2.91%, but its larger impact is on
selective prediction behavior. Under matched-coverage comparison, the fine-tuned
model outperforms the baseline at every coverage level with an average accuracy
gain of +5.11% — proving that fine-tuning improves confidence reliability,
not just raw predictions.

At threshold 0.50, answered-question accuracy increases from 52.24% to 70.33%
while coverage drops to 45.80%, reducing dataset-level wrong answers from
47.76% to 13.59%.

Calibration results are mixed: ECE slightly worsens from 0.0304 to 0.0322, while
MCE improves from 0.1179 to 0.0690. The main result should therefore be framed
as improved selective prediction behavior, not improved average calibration.

---

## 🔬 Extended Analysis

### Max-Prob vs Entropy Abstention

We compared two selective prediction strategies:
- **Max-probability thresholding**: abstain if `max(probs) < threshold`
- **Entropy-based abstention**: abstain if `entropy(probs) > threshold`

Entropy measures how concentrated the model's probability mass is over the four
answer options — high entropy means the model is confused.

**Head-to-head at ~50% coverage:**

| Method | Answered Accuracy | Coverage | Dataset-Level Wrong Rate |
|--------|-------------------|----------|------------|
| Max-Prob | 70.33% | 45.80% | 13.59% |
| Entropy | 67.73% | 49.41% | 15.95% |

Max-prob outperforms entropy on the fine-tuned model. Fine-tuning sharpens
confidence distributions, making max-prob a more reliable signal than entropy.

---

### Confidence Calibration (ECE)

Expected Calibration Error measures how well confidence aligns with actual accuracy.
A perfectly calibrated model follows the diagonal: "70% confident → correct 70% of the time."

| Model | ECE | MCE |
|-------|-----|-----|
| Baseline | 0.0304 | 0.1179 |
| Fine-tuned | 0.0322 | 0.0690 |

Fine-tuning did not improve average calibration as measured by ECE (0.0304 → 0.0322,
a negligible increase). However, it significantly reduced worst-bin miscalibration
as measured by MCE (11.79% → 6.90%). Low-confidence bins became better calibrated
after fine-tuning, which appears to improve abstention behavior in the uncertain region.

---

### ⚠️ Risk-Weighted Evaluation

Manual analysis of 10 high-confidence wrong answers and 10 low-confidence abstentions:

**High-confidence wrong answers:**

| Example | Predicted | Correct | Risk |
|---------|-----------|---------|------|
| Bone disease diagnosis | Osteitis fibrosa cystica | Osteitis deformans | Benign |
| Epidemiology bias | Lead-time bias | Measurement bias | Benign |
| Drug mechanism | Decreased phosphodiesterase | Increased adenylate cyclase | Ambiguous |
| Bleeding disorder (infant) | Bernard-Soulier | Glanzmann's thrombasthenia | **Critical** |
| Abdominal mass | Renal artery stenosis | Common iliac artery aneurysm | **Critical** |
| Neurological diagnosis | Degenerated caudate | Cerebellar demyelination | Ambiguous |
| Tropical disease | Dengue fever | Chikungunya | Benign |
| GI diagnosis | Crohn's disease | Ulcerative colitis | **Critical** |
| Psychiatric diagnosis | Schizophreniform | Schizoaffective | Ambiguous |
| Drug side effect | Breast cancer | Pulmonary embolism | **Critical** |

**In this small manually reviewed sample, 4/10 high-confidence wrong answers were categorized as critical.**

**Low-confidence abstentions (model correctly refused):**

| Example | Risk | Verdict |
|---------|------|---------|
| Wrong vaccine injection site → nerve damage | **Critical** | ✅ Correctly abstained |
| Wrong elbow reduction technique | **Critical** | ✅ Correctly abstained |
| Missed toxic shock syndrome history | **Critical** | ✅ Correctly abstained |
| Wrong test for aplastic anemia | **Critical** | ✅ Correctly abstained |

The abstention mechanism correctly refused to answer on all 4 critical cases
in the low-confidence set — suggesting potential clinical safety value, though larger expert-reviewed samples are needed.

---

## 🧠 Concepts Demonstrated

- Parameter-efficient fine-tuning (QLoRA)
- Selective prediction (accuracy–coverage tradeoff)
- Confidence calibration for LLMs (ECE, MCE)
- Max-probability vs entropy-based abstention
- Risk-weighted evaluation for medical AI
- Reliability-aware evaluation beyond accuracy

---

## 🛠️ Engineering Highlights

- Built end-to-end fine-tuning and evaluation pipeline for Mistral-7B on MedQA
- Implemented QLoRA training with early stopping for single-GPU fine-tuning on V100
- Designed post-hoc abstention layer using normalized A/B/C/D answer probabilities
- Added threshold-sweep evaluation to expose accuracy–coverage tradeoffs
- Compared max-probability and entropy-based uncertainty signals
- Produced reproducible JSON outputs for baseline, fine-tuned, calibration, and risk analyses
- Built `predict.py` CLI for single-question inference with configurable abstention threshold
- Documented deployment constraints including GPU type, training time, and inference time

---

## 🧪 Example Inference

```bash
python predict.py \
  --question "A patient presents with..." \
  --option_a "Aspirin" \
  --option_b "Ibuprofen" \
  --option_c "Acetaminophen" \
  --option_d "Morphine" \
  --threshold 0.50
```

**Answered (confidence above threshold):**
```json
{
  "prediction": "B",
  "confidence": 0.72,
  "abstained": false,
  "threshold": 0.50,
  "all_probs": {"A": 0.12, "B": 0.72, "C": 0.09, "D": 0.07},
  "message": "Answered with 72.0% confidence"
}
```

**Abstained (confidence below threshold):**
```json
{
  "prediction": null,
  "confidence": 0.41,
  "abstained": true,
  "threshold": 0.50,
  "all_probs": {"A": 0.41, "B": 0.28, "C": 0.19, "D": 0.12},
  "message": "Abstained due to low confidence"
}
```

---

## 🚢 Deployment Considerations

This project is an evaluation and abstention pipeline, not a clinical product.

For production-style deployment, the abstention layer wraps around model inference:

1. Format the medical multiple-choice prompt
2. Run the fine-tuned Mistral-7B model
3. Extract logits for answer options A/B/C/D
4. Normalize probabilities over answer choices only
5. Return answer only if max probability exceeds configured threshold
6. Otherwise return abstention response

The threshold can be configured per use case:
- Higher threshold for safety-critical settings (fewer answers, more reliable)
- Lower threshold for higher coverage (more answers, more errors)
- Separate calibration split recommended before deployment

**Note:** This system is designed as a research and evaluation prototype.
It should not be used for real clinical decision-making.

---

## ⚙️ Example Configuration

```yaml
model:
  base_model: mistralai/Mistral-7B-v0.3
  adapter: Primeinvincible/mistral-medqa-lora-v3
  quantization: 4bit

inference:
  answer_options: ["A", "B", "C", "D"]
  confidence_method: max_probability
  abstention_threshold: 0.50

evaluation:
  dataset: GBaker/MedQA-USMLE-4-options
  split: test
  metrics:
    - accuracy
    - coverage
    - dataset_level_wrong_rate
    - answered_accuracy
```

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
├── baseline_eval.py              # Evaluate base Mistral-7B on MedQA test set
├── train_lora.py                 # Fine-tune with QLoRA + early stopping
├── finetuned_eval.py             # Evaluate fine-tuned model on MedQA test set
├── abstention_analysis.py        # Sweep thresholds, compute abstention metrics
├── entropy_abstention.py         # Compare max-prob vs entropy abstention
├── reliability_diagram.py        # ECE calibration analysis
├── risk_analysis.py              # Manual risk-weighted evaluation
├── compare_abstention.py         # Matched-coverage baseline vs fine-tuned
├── predict.py                    # Single question inference with abstention
│
├── baseline_results.json         # Baseline evaluation results + confidence scores
├── finetuned_results.json        # Fine-tuned evaluation results + confidence scores
├── abstention_results.json       # Full threshold analysis results
├── entropy_abstention_results.json # Max-prob vs entropy comparison
├── reliability_results.json      # ECE calibration results
├── risk_analysis_examples.json   # Examples for manual review
├── risk_analysis_summary.md      # Manual risk categorization
├── comparison_results.json       # Matched-coverage comparison results
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

### 5. Run All Analyses
```bash
python abstention_analysis.py       # threshold sweep
python entropy_abstention.py        # max-prob vs entropy
python reliability_diagram.py       # ECE calibration
python risk_analysis.py             # risk-weighted examples
```

---

## 🔧 Load Fine-Tuned Model
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("Primeinvincible/mistral-medqa-lora-v3")

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
model = PeftModel.from_pretrained(base_model, "Primeinvincible/mistral-medqa-lora-v3")
model.eval()
```

---

## 💡 Key Design Decisions

**Why abstention over accuracy maximization?**
In medical AI, a confident wrong answer is worse than no answer. A model that
abstains on uncertain cases is safer than one that always answers.

**Why max-prob over entropy?**
After fine-tuning, the model's distributions become sharper — max-prob is a more
reliable signal than entropy for this model at this operating point.

**Why early stopping?**
Previous experiments showed overfitting after 2+ epochs. Early stopping with
patience=3 automatically stops when validation loss stops improving.

**Why QLoRA?**
Quantizing the base model to 4-bit while training only LoRA adapters
(~0.09% of total parameters) makes 7B model fine-tuning feasible on a single V100.

---

## 🔢 Confidence Computation

For each question, we:
1. Tokenize the prompt ending with `"Answer:"`
2. Run a forward pass and extract logits at the last token position
3. Select logits only for the four answer tokens `A`, `B`, `C`, `D`
4. Apply softmax over those four logits only — not the full vocabulary
5. Use the maximum probability as the confidence score

This gives a normalized confidence score over answer options only.
Note: this is **post-hoc confidence thresholding** — the model always
produces a prediction internally. The abstention decision is an external
rule applied on top of model probabilities, not a learned refusal behavior.

Prompt format is identical across baseline and fine-tuned evaluation.
Answer tokens are verified to be single-token continuations before evaluation.

---

## 📈 How to Choose a Threshold

| Use Case | Recommended Threshold | Reasoning |
|----------|-----------------------|-----------|
| High safety (triage) | 0.70+ | Minimize wrong answers, accept low coverage |
| Balanced | 0.50 | Best accuracy/coverage tradeoff |
| High coverage | 0.35 | Answer more questions, accept more errors |

---

## ⚠️ Limitations

- Evaluated on multiple-choice USMLE-style questions only — not open-ended clinical advice
- Abstention is post-hoc confidence thresholding, not a learned model-level refusal
- Threshold selection was performed on test-set results — a held-out calibration split would be more rigorous
- Risk-weighted analysis is preliminary and based on a small manually reviewed sample (n=10)
- Confidence intervals and statistical significance tests not yet computed
- The model should not be used for real clinical decision-making

---

## 🗂️ Dataset

**GBaker/MedQA-USMLE-4-options**
- 10,178 training examples
- 1,273 test examples
- Source: USMLE Step 1/2/3 medical licensing exam questions
- Format: 4-option multiple choice

Train/val split: 90/10 from training set (seed=42).
Official test set was not used for training or early stopping. Abstention thresholds were selected after inspecting test-set results, so future work should use a separate held-out calibration split for threshold selection.

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