from __future__ import annotations

import pytest

from arietta_voice.config import WakeWordConfig
from arietta_voice.wake import TranscriptWakeWordDetector, build_wake_word_detector


def test_transcript_wake_detector_matches_bare_wake_word() -> None:
    detector = TranscriptWakeWordDetector(("bridget",))

    match = detector.match("Bridget")

    assert match is not None
    assert match.phrase == "bridget"
    assert match.prompt_text is None


def test_transcript_wake_detector_matches_inline_command() -> None:
    detector = TranscriptWakeWordDetector(("bridget",))

    match = detector.match("Bridget what time is it")

    assert match is not None
    assert match.prompt_text == "what time is it"


def test_transcript_wake_detector_ignores_non_prefix_mentions() -> None:
    detector = TranscriptWakeWordDetector(("bridget",))

    match = detector.match("Tell me about Bridget the project")

    assert match is None


def test_build_wake_word_detector_uses_configured_backend() -> None:
    detector = build_wake_word_detector(
        WakeWordConfig(
            backend="transcript",
            phrases=("bridget", "arietta"),
        )
    )

    assert detector.match("Arietta are you there") is not None


@pytest.mark.parametrize("transcript, command", [
    ("Hey, Bridget: Set API_KEY to AbC_123!", "Set API_KEY to AbC_123!"),
    ("Bridget, is -5.2 less than +3?", "is -5.2 less than +3?"),
    ("Bridget: résumé for José, please.", "résumé for José, please."),
    ("Bridget, -5", "-5"),
    ("Bridget, .env is the filename", ".env is the filename"),
    ("Bridget .5 * 3", ".5 * 3"),
    ("  BRIDGET?!  ", None),
])
def test_inline_command_preserves_original_text(transcript, command) -> None:
    match = TranscriptWakeWordDetector(("bridget",)).match(transcript)
    assert match is not None
    assert match.transcript == transcript
    assert match.prompt_text == command


def test_multiword_and_unicode_wake_phrases_preserve_command() -> None:
    detector = TranscriptWakeWordDetector(("computer", "computer assistant", "Ária"))
    assert detector.match("Computer assistant, Call O'Neil!").prompt_text == "Call O'Neil!"
    assert detector.match("ÁRIA, Calculate 5*7.").prompt_text == "Calculate 5*7."
    assert detector.match("Computers are interesting") is None
