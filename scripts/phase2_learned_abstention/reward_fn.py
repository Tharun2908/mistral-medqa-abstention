# reward_fn.py
"""
GRPO reward for MedQA learned abstention (TRL 0.14).
Called as: medqa_reward(prompts=[...], completions=[...], ground_truth=[...]) -> list[float]

Training completion formats (exact):
    answer  : " The answer is X."   where X in {A,B,C,D}
    abstain : " I cannot answer confidently."

Parser is LENIENT on preamble/whitespace by design: GRPO generations are
free-form, so a correct answer with a leading token ("Well, the answer is A.")
must still score +1.0, not malformed. Hence .search() not anchored ^match.
"""
import re

CORRECT, ABSTAIN, WRONG, MALFORMED = 1.0, 0.3, -2.0, -2.2

ABSTAIN_MARKER = "i cannot answer confidently"
_ANSWER_RE = re.compile(r"the answer is\s+([a-d])\b", re.IGNORECASE)
# Detects "<letter> or/and/, <letter>" right after the answer letter -> ambiguous.
_AMBIG_RE = re.compile(r"the answer is\s+[a-d]\s*(?:,|or|and)\s*[a-d]\b", re.IGNORECASE)


def classify_completion(text, gt):
    if text is None:
        return "malformed"
    norm = " ".join(text.lower().split())  # lowercase + collapse all whitespace
    if not norm:
        return "malformed"

    if ABSTAIN_MARKER in norm:          # explicit refusal wins
        return "abstain"

    if _ANSWER_RE.search(norm) is None:
        return "malformed"

    if _AMBIG_RE.search(norm):          # "A or B", "A, B", "A and B" -> ambiguous
        return "malformed"

    pred = _ANSWER_RE.search(norm).group(1).upper()
    return "correct" if pred == gt.strip().upper() else "wrong"


def reward_single(text, gt):
    return {"correct": CORRECT, "wrong": WRONG,
            "abstain": ABSTAIN, "malformed": MALFORMED}[classify_completion(text, gt)]


def medqa_reward(prompts=None, completions=None, ground_truth=None, **kwargs):
    assert completions is not None and ground_truth is not None
    return [reward_single(c, gt) for c, gt in zip(completions, ground_truth)]
