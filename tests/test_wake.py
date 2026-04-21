from __future__ import annotations

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
