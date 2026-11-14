"""Recognising an answer that declines ("I don't have that information")."""

from __future__ import annotations

import re

DECLINES = re.compile(
    r"\b(i don't have (that|any|specific|enough|information|the)|i do not have|i'm not sure|"
    r"i am not sure|i don't know|i can't (say|answer|tell|help|find)|i cannot|no information|"
    r"not able to|unable to|try searching|search for (it|that)|i'm unable|don't have access)",
    re.IGNORECASE,
)


def declines(answer: str) -> bool:
    """Whether an answer says it does not know or have the answer."""
    return bool(DECLINES.search(answer.replace("’", "'")))
