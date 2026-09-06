"""
TRAIN-only diagnostic replay of the first 50 optimizer steps of clean GRPO Arm B.

Purpose:
Test the proposed mechanism for E collapse.

Arm B rewards:
    correct   +1
    abstain    0
    wrong     -1
    malformed -1

For an E sample, before GRPO normalization:

    centered E reward = 0 - group_mean_reward

Therefore:
    group mean < 0  -> E has positive advantage
    group mean = 0  -> E has zero advantage
    group mean > 0  -> E has negative advantage

We preserve the ORIGINAL 270-step scheduler horizon and stop via callback
after optimizer step 50. No DEV or TEST is loaded.
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.phase2_learned_abstention.train_clean_grpo import (
    BASE_MODEL,
    COMMON_INIT,
    OUTPUT_ROOT,
    SEED,
    GROUP_SIZE,
    PER_DEVICE_BATCH,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_PROMPT_LENGTH,
    MAX_COMPLETION_LENGTH,
    TEMPERATURE,
    EPOCHS,
    build_or_verify_manifest,
    make_dataset,
    seed_everything,
)

from scripts.phase2_learned_abstention.grpo_completion_parser import (
    classify_completion,
)


STOP_STEP = 50

OUT_DIR = (
    OUTPUT_ROOT
    / "diagnostics"
    / "arm_b_first50_replay"
)

OUT_JSON = (
    OUT_DIR
    / "arm_b_first50_advantage_diagnostic.json"
)


REWARD_MAP = {
    "correct": 1.0,
    "wrong": -1.0,
    "abstain": 0.0,
    "malformed": -1.0,
}


class StopAtStep(TrainerCallback):

    def __init__(self, stop_step):
        self.stop_step = stop_step

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):

        if state.global_step >= self.stop_step:
            control.should_training_stop = True

        return control


class ArmBDiagnosticReward:

    def __init__(
        self,
        group_size,
        trainer_holder,
    ):

        self.group_size = group_size
        self.trainer_holder = trainer_holder

        self.calls = 0
        self.rows = []

    def _expand_gold(
        self,
        ground_truth,
        n,
    ):

        gt = list(ground_truth)

        if len(gt) == n:
            return gt

        if len(gt) * self.group_size == n:

            expanded = []

            for x in gt:
                expanded.extend(
                    [x] * self.group_size
                )

            return expanded

        raise RuntimeError(
            f"Unexpected gold/completion sizes: "
            f"{len(gt)} vs {n}"
        )

    def __call__(
        self,
        prompts=None,
        completions=None,
        ground_truth=None,
        **kwargs,
    ):

        completions = list(completions)

        golds = self._expand_gold(
            ground_truth,
            len(completions),
        )

        classes = [
            classify_completion(c, g)
            for c, g
            in zip(completions, golds)
        ]

        rewards = [
            REWARD_MAP[c]
            for c in classes
        ]

        trainer = self.trainer_holder["trainer"]

        # state.global_step is the number of optimizer updates
        # already completed when this generation occurs.
        completed_step = int(
            trainer.state.global_step
        )

        target_step = (
            completed_step + 1
        )

        self.calls += 1

        if len(classes) % self.group_size != 0:
            raise RuntimeError(
                "Completion count is not divisible by G"
            )

        n_groups = (
            len(classes)
            // self.group_size
        )

        for gi in range(n_groups):

            a = gi * self.group_size
            b = a + self.group_size

            gc = classes[a:b]

            gr = np.asarray(
                rewards[a:b],
                dtype=float,
            )

            mean_reward = float(
                gr.mean()
            )

            n_correct = gc.count("correct")
            n_wrong = gc.count("wrong")
            n_abstain = gc.count("abstain")
            n_malformed = gc.count("malformed")

            # E reward is exactly 0.
            # Sign of centered E advantage is sign(-mean_reward).
            if mean_reward < -1e-12:
                e_adv_sign = "positive"
            elif mean_reward > 1e-12:
                e_adv_sign = "negative"
            else:
                e_adv_sign = "zero"

            self.rows.append(
                {
                    "optimizer_step":
                        target_step,

                    "reward_call":
                        self.calls,

                    "group_in_call":
                        gi,

                    "n_correct":
                        n_correct,

                    "n_wrong":
                        n_wrong,

                    "n_abstain":
                        n_abstain,

                    "n_malformed":
                        n_malformed,

                    "group_mean_reward":
                        mean_reward,

                    "e_centered_reward":
                        float(
                            -mean_reward
                        ),

                    "e_advantage_sign":
                        e_adv_sign,
                }
            )

        return rewards


def summarize(rows):

    bins = [
        (1, 10),
        (11, 25),
        (26, 50),
    ]

    result = {
        "overall": {},
        "step_bins": {},
    }

    def stats(sub):

        total_groups = len(sub)

        total_completions = (
            total_groups
            * GROUP_SIZE
        )

        e_samples = sum(
            r["n_abstain"]
            for r in sub
        )

        e_groups = [
            r
            for r in sub
            if r["n_abstain"] > 0
        ]

        group_signs = Counter(
            r["e_advantage_sign"]
            for r in e_groups
        )

        # Weight by number of E samples in each group.
        sample_signs = Counter()

        for r in e_groups:

            sample_signs[
                r["e_advantage_sign"]
            ] += r["n_abstain"]

        means_with_e = [
            r["group_mean_reward"]
            for r in e_groups
        ]

        return {
            "groups":
                total_groups,

            "completions":
                total_completions,

            "e_samples":
                e_samples,

            "e_completion_rate":
                (
                    e_samples
                    / total_completions
                    if total_completions
                    else 0.0
                ),

            "groups_with_e":
                len(e_groups),

            "e_group_rate":
                (
                    len(e_groups)
                    / total_groups
                    if total_groups
                    else 0.0
                ),

            "e_groups_by_advantage_sign":
                dict(group_signs),

            "e_samples_by_advantage_sign":
                dict(sample_signs),

            "mean_group_reward_given_e":
                (
                    float(
                        np.mean(means_with_e)
                    )
                    if means_with_e
                    else None
                ),
        }

    result["overall"] = stats(rows)

    for lo, hi in bins:

        sub = [
            r
            for r in rows
            if lo
            <= r["optimizer_step"]
            <= hi
        ]

        result[
            "step_bins"
        ][f"{lo}-{hi}"] = stats(sub)

    return result


def main():

    seed_everything(SEED)

    print("=" * 90)
    print("ARM B FIRST-50 ADVANTAGE DIAGNOSTIC REPLAY")
    print("=" * 90)

    print("\nTRAIN only")
    print("DEV loaded : NO")
    print("TEST loaded: NO")

    print(
        f"\nReward: {REWARD_MAP}"
    )

    print(
        f"Stop after optimizer step: "
        f"{STOP_STEP}"
    )

    examples = (
        build_or_verify_manifest()
    )

    dataset = make_dataset(
        examples,
        smoke=False,
    )

    print(
        f"Training prompts: "
        f"{len(dataset)}"
    )

    if OUT_DIR.exists():
        shutil.rmtree(
            OUT_DIR
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = "left"

    print(
        "\nLoading base bf16..."
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    base.config.use_cache = False

    print(
        "Loading common init trainable..."
    )

    model = (
        PeftModel.from_pretrained(
            base,
            str(COMMON_INIT),
            is_trainable=True,
        )
    )

    model.config.use_cache = False

    holder = {
        "trainer": None
    }

    diag = ArmBDiagnosticReward(
        GROUP_SIZE,
        holder,
    )

    def reward_func(
        prompts=None,
        completions=None,
        ground_truth=None,
        **kwargs,
    ):

        return diag(
            prompts=prompts,
            completions=completions,
            ground_truth=ground_truth,
            **kwargs,
        )

    reward_func.__name__ = (
        "arm_b_first50_diagnostic_reward"
    )

    # IMPORTANT:
    # no max_steps=50 here.
    # Keeping 1 epoch preserves the original ~270-step
    # cosine scheduler horizon.
    cfg = GRPOConfig(
        output_dir=str(OUT_DIR),

        per_device_train_batch_size=(
            PER_DEVICE_BATCH
        ),

        gradient_accumulation_steps=(
            GRAD_ACCUM
        ),

        num_generations=(
            GROUP_SIZE
        ),

        max_prompt_length=(
            MAX_PROMPT_LENGTH
        ),

        max_completion_length=(
            MAX_COMPLETION_LENGTH
        ),

        temperature=(
            TEMPERATURE
        ),

        beta=0.0,

        learning_rate=(
            LEARNING_RATE
        ),

        lr_scheduler_type="cosine",

        warmup_ratio=0.05,

        num_train_epochs=(
            EPOCHS
        ),

        logging_steps=10,

        save_strategy="no",

        bf16=True,
        fp16=False,

        gradient_checkpointing=False,

        report_to="none",

        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=model,

        reward_funcs=[
            reward_func
        ],

        args=cfg,

        train_dataset=dataset,

        processing_class=tokenizer,

        peft_config=None,
    )

    holder["trainer"] = trainer

    trainer.add_callback(
        StopAtStep(
            STOP_STEP
        )
    )

    print(
        f"\nTrainer planned max_steps: "
        f"{trainer.state.max_steps}"
    )

    print(
        "Starting diagnostic replay..."
    )

    trainer.train()

    rows = [
        r
        for r in diag.rows
        if r["optimizer_step"]
        <= STOP_STEP
    ]

    summary = summarize(
        rows
    )

    output = {
        "protocol":
            "arm_b_first50_train_only_diagnostic_v1",

        "replay":
            True,

        "important_note":
            (
                "This is a deterministic-intent diagnostic replay "
                "from the same initialization/configuration, not a "
                "reconstruction of per-group data absent from the "
                "original training log."
            ),

        "split":
            "train",

        "dev_used":
            False,

        "test_used":
            False,

        "stop_step":
            STOP_STEP,

        "reward_map":
            REWARD_MAP,

        "summary":
            summary,

        "groups":
            rows,
    }

    with open(
        OUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for name, s in [
        ("OVERALL", summary["overall"]),
        *[
            (
                f"STEPS {k}",
                v,
            )
            for k, v
            in summary[
                "step_bins"
            ].items()
        ],
    ]:

        print(f"\n{name}")

        print(
            f"E completion rate: "
            f"{s['e_completion_rate']:.2%}"
        )

        print(
            f"groups with E: "
            f"{s['groups_with_e']}"
        )

        print(
            "E groups by advantage sign: "
            f"{s['e_groups_by_advantage_sign']}"
        )

        print(
            "E samples by advantage sign: "
            f"{s['e_samples_by_advantage_sign']}"
        )

        print(
            "mean group reward | E present: "
            f"{s['mean_group_reward_given_e']}"
        )

    print(
        f"\nSaved:\n{OUT_JSON}"
    )

    print(
        "\nARM B FIRST-50 DIAGNOSTIC: PASS"
    )


if __name__ == "__main__":
    main()
