"""Spoken answers to the assistant's "Should I …?" before an action.

Only a clear answer counts. "Yes", "yeah, go ahead" and "sure, do it" agree;
"no", "cancel" and "never mind" refuse; "always" or "yes, and don't ask again"
agree and stop the assistant asking about that action. Anything else, such as
"yes, but the kitchen" or a new request, is not an answer: the action is dropped
and the words are handled as a request.
"""

from __future__ import annotations

import re

_YES = (
    r"yes|yeah|yea|yep|yup|sure|ok|okay|alright|all right|please|please do|do it|do that|"
    r"go ahead|go for it|confirm|confirmed|correct|that's right|right|affirmative|"
    r"absolutely|of course|sounds good|definitely|i do|you can|you may"
)
_NO = (
    r"no|nope|nah|no thanks|no thank you|don't|do not|don't do it|don't do that|"
    r"cancel|cancel that|cancel it|stop|never mind|nevermind|forget it|not now|"
    r"wait|hold on|leave it|i don't|you can't|negative"
)
_ALWAYS = (
    r"always|always do that|do that always|(?:and )?(?:don't|do not) ask (?:me )?"
    r"(?:that )?again|you don't (?:need|have) to ask(?: me)?(?: again)?|"
    r"no need to ask(?: me)?(?: again)?|from now on"
)
_FILLER = r"(?:(?:oh|uh|um|umm|well|hmm|ah)\s+)*"
_COURTESY = r"(?:\s+(?:please|thanks|thank you))*"


def _sequence(words: str) -> re.Pattern:
    return re.compile(rf"{_FILLER}(?:{words})(?:\s+(?:{words}))*{_COURTESY}")


YES = _sequence(_YES)
NO = _sequence(_NO)
# "Always", or any yes followed by an always ("yes, and don't ask again").
ALWAYS = re.compile(rf"{_FILLER}(?:(?:{_YES})\s+(?:and\s+)?)*(?:{_ALWAYS}){_COURTESY}")


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    return " ".join(text.split())


def confirmation_reply(text: str) -> str | None:
    """Classify an answer as "yes", "always" or "no"; None when it is not a clear one."""
    words = normalize(text)
    if not words:
        return None
    if ALWAYS.fullmatch(words):
        return "always"
    if NO.fullmatch(words):
        return "no"
    if YES.fullmatch(words):
        return "yes"
    return None
