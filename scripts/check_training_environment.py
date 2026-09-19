"""Offline CPU compatibility check; no checkpoints, datasets, or GPU required.

Run from the repository root: python scripts/check_training_environment.py
This checks imports and a tiny randomly initialized Mistral LoRA backward pass.
It does not reproduce the MedQA experiments or test CUDA quantization.
"""

import importlib
import os
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import MistralConfig, MistralForCausalLM
    from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTTrainer

    for name in ("torch", "transformers", "tokenizers", "accelerate", "peft",
                 "trl", "datasets", "bitsandbytes", "triton", "setuptools"):
        print(f"{name}=={version(name)}", flush=True)

    # Import the actual clean-protocol entry points without running their main().
    for name in ("train_clean_grpo", "train_grpo_posthoc_reward03",
                 "train_dpo_common_init", "train_continue_sft_control",
                 "train_supervised_5way"):
        importlib.import_module(f"scripts.phase2_learned_abstention.{name}")
    print("PASS: training entry points and SFT/DPO/GRPO trainer imports", flush=True)

    # Exercise configuration construction as well as TRL's lazy imports.
    with TemporaryDirectory(prefix="medqa-config-check-") as output_dir:
        GRPOConfig(
            output_dir=output_dir, use_cpu=True, bf16=False, fp16=False,
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_generations=8, max_prompt_length=384, max_completion_length=16,
            temperature=1.0, beta=0.0, report_to="none",
        )
        DPOConfig(
            output_dir=output_dir, use_cpu=True, bf16=False, fp16=False,
            model_adapter_name="policy", ref_adapter_name="reference",
            max_length=1024, max_prompt_length=768, report_to="none",
        )
    print("PASS: DPO and GRPO configuration construction", flush=True)

    # A successful import alone does not prove that PEFT can wrap this architecture.
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = get_peft_model(
        MistralForCausalLM(MistralConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                   task_type="CAUSAL_LM"),
    )
    tokens = torch.tensor([[1, 4, 5, 2]])
    loss = model(input_ids=tokens, labels=tokens).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Tiny Mistral loss is not finite")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("LoRA gradients are missing or non-finite")
    if not any(torch.count_nonzero(g).item() for g in gradients):
        raise RuntimeError("All LoRA gradients are zero")
    print("PASS: tiny Mistral LoRA forward/backward on CPU", flush=True)
    print("Environment compatibility check passed; GPU training was not tested.")


if __name__ == "__main__":
    main()
