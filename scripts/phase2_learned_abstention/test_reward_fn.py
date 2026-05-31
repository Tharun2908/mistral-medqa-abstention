# test_reward_fn.py
import pytest
from reward_fn import classify_completion, reward_single, medqa_reward


@pytest.mark.parametrize("text,gt,expected", [
    # --- exact training strings ---
    (" The answer is A.", "A", "correct"),
    (" The answer is B.", "A", "wrong"),
    (" I cannot answer confidently.", "A", "abstain"),
    # --- malformed ---
    ("", "A", "malformed"),
    (" qwerty", "A", "malformed"),
    (None, "A", "malformed"),
    # --- robustness (parser must handle these) ---
    ("The answer is A.", "A", "correct"),          # no leading space
    ("  The  answer  is  A. ", "A", "correct"),     # extra whitespace
    (" the answer is a.", "A", "correct"),          # lowercase
    # --- online-generation leniency (the reason we don't anchor ^) ---
    (" Well, the answer is A.", "A", "correct"),    # preamble before answer
    (" The answer is A. The actual question is...", "A", "correct"),  # parse first only
    # --- documented policy choices ---
    (" I cannot answer", "A", "malformed"),         # truncated abstain -> malformed
    (" The answer is A or B.", "A", "malformed"),   # ambiguous -> malformed
    (" The answer is E.", "A", "malformed"),        # E-as-letter -> malformed (A-D only)
])
def test_classify(text, gt, expected):
    assert classify_completion(text, gt) == expected


def test_reward_values():
    # v4.1 reward magnitudes: stronger asymmetry to make abstention positive-EV
    # for a ~52%-accurate policy. See grpo_v4_full.py docstring for EV derivation.
    assert reward_single(" The answer is A.", "A") == 1.0
    assert reward_single(" The answer is B.", "A") == -2.0
    assert reward_single(" I cannot answer confidently.", "A") == 0.3
    assert reward_single("", "A") == -2.2


def test_batch_alignment():
    # the TRL-facing entry point: order + length must match inputs exactly
    out = medqa_reward(
        prompts=["p1", "p2", "p3"],
        completions=[" The answer is A.", " The answer is B.", " I cannot answer confidently."],
        ground_truth=["A", "A", "C"],
    )
    assert out == [1.0, -2.0, 0.3]
