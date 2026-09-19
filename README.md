# Mistral-7B MedQA Abstention

Selective prediction and learned abstention for medical multiple-choice QA with Mistral-7B.

This project started with a simple question: **can a model learn when not to answer?** It eventually became a controlled comparison of post-hoc confidence, supervised abstention, DPO, and GRPO/RLVR—and a study in how evaluation choices can change the apparent conclusion.

## Final takeaway

On MedQA, **strong answer-focused continued SFT plus post-hoc confidence was difficult to beat**.

After correcting historical TEST leakage, matching training compute, matching model initialization, using full-completion scoring consistently, freezing thresholds on DEV, and running a 10,000-replicate paired TEST bootstrap:

- **Continue-SFT** had the highest deployment wrongness AUROC in the original locked comparison: **0.7285**.
- **Supervised 5-way abstention** had the highest raw TEST answer accuracy: **0.5837**.
- **Correct-only SFT, Continue-SFT, and Supervised 5-way were mostly statistically indistinguishable** in paired selective-prediction comparisons.
- **DPO improved when rerun from the same strong initialization**, confirming that the earlier DPO comparison was partly base-confounded, but the corrected DPO still trailed Continue-SFT in deployment AUROC.
- **GRPO did not improve uncertainty ranking**. Both GRPO arms selected their earliest checkpoint because ranking deteriorated with more RL.

The main lesson is therefore not that learned abstention beats thresholding. It is that **strong supervised answer training is an extremely competitive selective-prediction baseline, and learned abstention claims are highly sensitive to protocol, initialization, scoring, and checkpoint selection.**

---

## Task and methods

Dataset: MedQA-USMLE, four answer choices (A–D).

Model family: Mistral-7B with parameter-efficient fine-tuning.

The final controlled comparison includes:

1. **Original SFT + confidence**
2. **Continue-SFT + confidence**
3. **Correct-only SFT + confidence**
4. **Supervised 5-way abstention** with explicit E = abstain
5. **DPO learned abstention**
6. **GRPO common initialization**
7. **GRPO Arm A** — answer-only reward
8. **GRPO Arm B** — explicit abstention reward

A later **common-init DPO rerun** is reported separately as a post-hoc robustness control because it was motivated after the original locked TEST had already been opened.

### Deployment scores

- Answer-focused models: A–D confidence
- Explicit abstention models: E margin

E margin is:

`score(E) - max(score(A), score(B), score(C), score(D))`

Higher margin means more abstention-like / error-like.

All learned-abstention evaluations use the same full-sentence completion scoring format as training.

---

## Clean evaluation protocol

Official MedQA split sizes:

- train: 10,178
- DEV: 1,272
- TEST: 1,273

Final protocol:

- TRAIN is used for parameter learning.
- Internal train-derived validation is used for checkpoint restoration where needed.
- Official DEV is used for model/configuration choices and to freeze numeric operating-point thresholds.
- Official TEST is used for final evaluation only.
- TEST thresholds are never reselected.
- Frozen DEV thresholds are applied literally to TEST, so actual TEST coverage can differ from the nominal DEV target.
- Final uncertainty estimates use a **10,000-replicate paired bootstrap** over the same 1,273 TEST examples.
- Reported 95% intervals are nominal and are not multiplicity-adjusted.

Historical exploratory experiments did inspect TEST. Those results are superseded by the clean rerun described here.

---

## Original locked TEST comparison

| Method | WB accuracy [95% CI] | Deployment AUROC [95% CI] |
|---|---:|---:|
| Original SFT | 0.4878 [0.4603, 0.5145] | 0.7186 [0.6901, 0.7461] |
| **Continue-SFT** | 0.5719 [0.5444, 0.5994] | **0.7285 [0.7009, 0.7557]** |
| Correct-only SFT | 0.5805 [0.5530, 0.6072] | 0.7204 [0.6923, 0.7478] |
| **Supervised 5-way** | **0.5837 [0.5569, 0.6112]** | 0.7099 [0.6818, 0.7384] |
| DPO 2:1 | 0.5208 [0.4925, 0.5483] | 0.6710 [0.6411, 0.7000] |
| GRPO common init | 0.5742 [0.5467, 0.6009] | 0.7207 [0.6929, 0.7479] |
| GRPO Arm A | 0.5813 [0.5538, 0.6080] | 0.7168 [0.6887, 0.7445] |
| GRPO Arm B | 0.5789 [0.5515, 0.6064] | 0.6632 [0.6333, 0.6935] |

### Three useful definitions of "best"

**Highest raw answer accuracy:** Supervised 5-way, 0.5837.

**Highest deployment error-ranking AUROC:** Continue-SFT, 0.7285.

**Low-coverage operating point:** there is no single fair winner from answered accuracy alone because DEV-frozen thresholds transferred to different actual TEST coverages. Coverage and accuracy must be quoted together.

For example, around the nominal 30% target:

| Method | Actual TEST coverage | Answered accuracy |
|---|---:|---:|
| Continue-SFT | 28.28% | 82.50% |
| Correct-only SFT | 26.32% | 83.58% |
| Supervised 5-way | 28.36% | 83.10% |
| GRPO common init | 26.16% | **84.98%** |
| GRPO Arm A | 26.87% | 83.92% |

This table is descriptive, not evidence that GRPO common init is statistically superior.

---

## What the paired bootstrap says

### Continue-SFT vs Original SFT

- WB accuracy difference: **+0.0841 [ +0.0589, +0.1100 ]**
- deployment AUROC difference: +0.0099 [ -0.0227, +0.0418 ]

Additional answer-focused training clearly improved answer accuracy, while the AUROC improvement was not statistically resolved.

### Continue-SFT vs Supervised 5-way

- WB accuracy difference: -0.0118 [ -0.0267, +0.0031 ]
- deployment AUROC difference: +0.0186 [ -0.0039, +0.0407 ]

Global accuracy and ranking are statistically unresolved. A nominal advantage for Continue-SFT appears at the transferred 40% operating point, but this isolated result is not treated as a broad significance claim because multiple operating points were examined.

### Continue-SFT vs Correct-only SFT

- WB accuracy difference: -0.0086 [ -0.0228, +0.0055 ]
- deployment AUROC difference: +0.0081 [ -0.0128, +0.0285 ]

All reported 30/40/50/60% answered-accuracy and utility differences cross zero.

### Correct-only SFT vs Supervised 5-way

- WB accuracy difference: -0.0031 [ -0.0165, +0.0102 ]
- deployment AUROC difference: +0.0105 [ -0.0117, +0.0331 ]

All selective operating-point intervals cross zero.

**Interpretation:** under matched controls, there is no evidence that explicit E-label supervision adds a consistent selective-prediction advantage over answer-focused continued training on MedQA.

---

## GRPO / RLVR: negative result

The clean GRPO experiment used a common E-aware initialization and verified that abstention was genuinely reachable before RL:

- 2,048 sampled completions in the structural preflight
- abstention rate: 7.28%
- 42.58% of groups contained both answer and abstention actions
- non-zero advantage in 79.69% of Arm-A groups and 85.94% of Arm-B groups

### Arm A: answer-only reward

Reward:

- correct: +1
- wrong: -1
- abstain: -1
- malformed: -1

Deployment score: A–D confidence.

### Arm B: explicit abstention reward

Reward:

- correct: +1
- wrong: -1
- abstain: 0
- malformed: -1

Deployment score: E margin.

### What happened

Both arms selected **checkpoint 50**, the earliest saved checkpoint, because selective ranking deteriorated with continued RL.

GRPO Arm A vs its own common initialization:

- WB accuracy difference: +0.0071 [ -0.0008, +0.0157 ]
- deployment AUROC difference: -0.0040 [ -0.0159, +0.0078 ]

All 30/40/50/60% answered-accuracy and utility intervals cross zero.

**Result:** answer-only GRPO did not measurably improve selective prediction over the strong initialization it started from.

GRPO Arm A vs Arm B:

- WB accuracy difference: +0.0024 [ -0.0031, +0.0086 ]
- deployment AUROC difference: **+0.0535 [ +0.0295, +0.0771 ]**

Raw answer accuracy is indistinguishable, but the explicit-abstention Arm B has substantially worse deployment ranking.

---

## Why did Arm B lose abstention?

The clean initialization did explore E, so "the model never sampled abstention" is not a sufficient explanation.

A deterministic replay of the first 50 Arm-B steps found:

- E completion rate: 1.97%
- E-containing groups: 52
- negative-centered-advantage groups: 26
- positive-centered-advantage groups: 24
- zero: 2

Weighted E samples were also approximately balanced, with slightly positive centered reward overall.

So the specific hypothesis that E was systematically punished by negative centered advantage was not supported.

A more cautious mechanism is **on-policy action extinction**: E was rare, received sparse updates, and lost probability mass while much more common answer trajectories repeatedly received reinforcement before conditional abstention could bootstrap.

### Post-hoc reward rescue

A single reviewer-motivated follow-up changed only Arm B's abstention reward from 0 to +0.3.

It modestly increased E usage:

- original E completions: 0.89%
- E=+0.3: 1.19%

But selected DEV margin AUROC stayed exactly **0.6428**, and checkpoint 50 was again selected. Natural coverage still collapsed toward 1 with continued RL.

This follow-up is post-hoc and is not part of the original locked TEST comparison.

---

## DPO after matching the initialization

The original clean DPO row used a weaker starting model than the later GRPO comparison. To remove that asymmetry, DPO 2:1 was rerun from the same strong common initialization.

Training:

- pair ratio: 2:1
- LR: 5e-6
- beta: 0.1
- epochs: 2
- best internal checkpoint: 475
- best eval loss: 0.5641

DEV:

- WB accuracy: 0.5338
- natural coverage: 0.5613
- E-margin wrongness AUROC: **0.6666**

Old DPO DEV margin AUROC was 0.6597.

### Post-hoc TEST robustness control

Because this rerun was motivated after the original locked TEST had already been opened, it is reported separately.

Common-init DPO:

- WB accuracy: **0.5577 [0.5302, 0.5844]**
- E-margin AUROC: **0.6909 [0.6608, 0.7190]**

Against old DPO:

- WB accuracy difference: **+0.0369 [ +0.0173, +0.0566 ]**
- AUROC difference: +0.0198 [ -0.0086, +0.0470 ]

At all four transferred operating points, common-init DPO had higher answered accuracy and utility than old DPO with nominal paired intervals above zero.

So the earlier DPO result was materially affected by its weaker starting model.

Against Continue-SFT:

- WB accuracy difference: -0.0141 [ -0.0322, +0.0039 ]
- deployment AUROC difference: **-0.0377 [ -0.0638, -0.0117 ]**

At the transferred 40%, 50%, and 60% targets, common-init DPO also had lower answered accuracy and utility.

**Conclusion:** fairer initialization improves DPO, but does not make it outperform the strongest answer-focused SFT baseline.

---

## What changed from the earlier project claims

Several earlier exploratory claims are superseded.

### Superseded: "DPO reaches operating regions post-hoc thresholding cannot"

The final clean comparison does not support this as a general advantage. Different learned policies can have different natural argmax operating points, but deployment comparisons must be made with frozen score thresholds and matched protocol.

### Superseded: "learned abstention clearly beats post-hoc confidence"

Not supported after the clean controls. Continue-SFT + confidence is among the strongest methods.

### Superseded: "GRPO failed because E was never explored"

False under the clean common initialization. E was demonstrably sampled in mixed-action groups before RL.

### Superseded: "DPO vs GRPO reveals a structural preference-learning advantage"

Too strong. The original DPO comparison was partly confounded by weaker initialization. After matching the base, DPO improves, but the broader result is that neither DPO nor GRPO beats the strongest answer-focused SFT baseline.

### Superseded: historical TEST-selected threshold results

Earlier exploratory threshold tables used TEST information for selection. They remain historical debugging evidence only and are not final evaluation results.

---

## Reproducibility map

### Training environment

Use an isolated **Linux / Python 3.12** environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_training_environment.py
```

`requirements.txt` is a compatibility baseline checked by CI, **not an archived
package freeze from the historical GPU runs**. The committed run summaries do not
record a complete environment. TRL is pinned to **0.14.0**, matching the version
referenced in the GRPO scripts and their prompt-batch semantics; the previous
0.12.2 pin did not provide `GRPOConfig` or `GRPOTrainer`.
See the [TRL 0.14 GRPO documentation](https://huggingface.co/docs/trl/v0.14.0/grpo_trainer).

The environment check runs offline on CPU: it imports the clean SFT/DPO/GRPO
entry points, constructs trainer configurations, and runs a tiny randomly
initialized Mistral LoRA forward/backward pass. It downloads no model or dataset
and does not evaluate TEST. CI uses the CPU build of the same PyTorch version.
This check does **not** validate CUDA, bitsandbytes GPU kernels, full trainer
execution, or reproduction of the published metrics. The clean GRPO runs used
an H200 with BF16; real training also requires the prepared training artifacts
and initialization checkpoints referenced by the scripts.

Before a new GPU run, save its actual environment alongside its outputs:

```bash
python -m pip freeze > environment-freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))" > gpu-environment.txt
```

### Committed results

Core clean-protocol artifacts live under:

```text
results/clean_protocol/
```

Important final outputs:

```text
results/clean_protocol/locked_test/final_comparison/
results/clean_protocol/locked_test/grpo/
results/clean_protocol/posthoc_test/dpo_common_init_ratio_2to1/
results/clean_protocol/learned_abstention/grpo/
results/clean_protocol/learned_abstention/dpo/
```

Final paired bootstrap:

```text
results/clean_protocol/locked_test/final_comparison/
final_test_paired_bootstrap.json
```

Common-init DPO post-hoc TEST:

```text
results/clean_protocol/posthoc_test/dpo_common_init_ratio_2to1/
dpo_common_init_ratio_2to1_test.json
```

---

## Final conclusion

This project began as an attempt to train explicit abstention into Mistral-7B. The clean experiments produced a more important result:

> **On MedQA, strong supervised answer training plus post-hoc confidence remained difficult to beat. Explicit abstention supervision, DPO, and GRPO produced useful behaviors and diagnostics, but under matched controls none demonstrated a consistent selective-prediction advantage over the strongest answer-focused SFT baselines.**

The largest practical lesson is methodological. Evaluation leakage, scoring mismatches, extra training, initialization asymmetry, and post-hoc checkpoint choices can all make learned abstention look stronger than it is. Once those factors were controlled, the project became a reproducible comparison of not only abstention methods, but also the evaluation discipline required to assess them fairly.
