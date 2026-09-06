"""
Shared completion parser for clean MedQA GRPO experiments.

Conservative principle:
- accept clearly expressed single A/B/C/D answers in several common forms;
- accept the exact abstention phrase;
- reject vague prose, conflicting answers, and unrecognized text.

Both GRPO arms and the sampling preflight must use this exact parser.
"""

import re


ANSWER_LABELS = {"A", "B", "C", "D"}

ABSTAIN_MARKER = "i cannot answer confidently"


# Recognized answer forms.
_PATTERNS = [
    # "The answer is B."
    re.compile(
        r"\bthe answer is\s*([a-d])\b",
        re.IGNORECASE,
    ),

    # "Answer: B", "Answer B"
    re.compile(
        r"\banswer\s*:?\s*([a-d])\b",
        re.IGNORECASE,
    ),

    # "B is correct."
    re.compile(
        r"\b([a-d])\s+is\s+(?:the\s+)?(?:correct|best)\b",
        re.IGNORECASE,
    ),

    # "Most likely option is C."
    # "The best option is C."
    re.compile(
        r"\b(?:most likely|best|correct)\s+option\s+is\s*([a-d])\b",
        re.IGNORECASE,
    ),

    # "Option C is correct."
    re.compile(
        r"\boption\s+([a-d])\s+is\s+(?:the\s+)?(?:correct|best)\b",
        re.IGNORECASE,
    ),
]


# Bare answer only:
# "A"
# "A."
# "A)"
_BARE_ANSWER = re.compile(
    r"^\s*([a-d])[\.\)]?\s*$",
    re.IGNORECASE,
)


def extract_answer_letter(text):
    """
    Return A/B/C/D only when the completion expresses one
    unambiguous recognized answer. Otherwise return None.
    """

    if text is None:
        return None

    norm = " ".join(
        str(text).strip().split()
    )

    if not norm:
        return None

    found = []

    bare = _BARE_ANSWER.fullmatch(norm)

    if bare:
        found.append(
            bare.group(1).upper()
        )

    for pattern in _PATTERNS:
        for match in pattern.finditer(norm):
            found.append(
                match.group(1).upper()
            )

    if not found:
        return None

    unique = set(found)

    # Conflicting recognized answers => malformed.
    if len(unique) != 1:
        return None

    answer = next(iter(unique))

    if answer not in ANSWER_LABELS:
        return None

    return answer


def classify_completion(text, gold):
    """
    Classes:
        correct
        wrong
        abstain
        malformed
    """

    if text is None:
        return "malformed"

    norm = " ".join(
        str(text).lower().split()
    )

    if not norm:
        return "malformed"

    # Explicit abstention wins.
    if ABSTAIN_MARKER in norm:
        return "abstain"

    pred = extract_answer_letter(
        text
    )

    if pred is None:
        return "malformed"

    gold = str(gold).strip().upper()

    if gold not in ANSWER_LABELS:
        raise ValueError(
            f"Unexpected gold label: {gold!r}"
        )

    return (
        "correct"
        if pred == gold
        else "wrong"
    )


if __name__ == "__main__":

    tests = [
        ("The answer is B.", "B", "correct"),
        ("B is correct.", "B", "correct"),
        ("A.", "A", "correct"),
        ("Most likely option is C.", "A", "wrong"),
        ("Answer: D", "D", "correct"),
        ("There is no answer!", "A", "malformed"),
        (
            "All of the statements are true except statements",
            "A",
            "malformed",
        ),
    ]

    for text, gold, expected in tests:

        got = classify_completion(
            text,
            gold,
        )

        print(
            f"{text!r:55} "
            f"gold={gold} -> {got}"
        )

        assert got == expected, (
            text,
            got,
            expected,
        )

    print("\nPARSER SELF-TEST: PASS")
