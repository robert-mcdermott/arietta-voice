from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from arietta_voice.config import WakeWordConfig


WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
LEADING_FILLERS = {"hey", "hi", "hello", "okay", "ok"}


@dataclass(slots=True, frozen=True)
class WakeWordMatch:
    phrase: str
    transcript: str
    prompt_text: str | None = None


class WakeWordDetector(Protocol):
    def match(self, transcript: str) -> WakeWordMatch | None: ...


def normalize_phrase(value: str) -> str:
    return " ".join(match.group().casefold() for match in WORD_RE.finditer(value))


class TranscriptWakeWordDetector:
    def __init__(self, phrases: tuple[str, ...]) -> None:
        normalized = {
            normalize_phrase(phrase): phrase.strip()
            for phrase in phrases
            if normalize_phrase(phrase)
        }
        self._phrases = tuple(
            (normalized_phrase, original)
            for normalized_phrase, original in sorted(
                normalized.items(),
                key=lambda item: (-len(item[0]), item[0]),
            )
        )

    def match(self, transcript: str) -> WakeWordMatch | None:
        words = list(WORD_RE.finditer(transcript))
        while words and words[0].group().casefold() in LEADING_FILLERS:
            words.pop(0)
        if not words:
            return None

        for normalized_phrase, original_phrase in self._phrases:
            phrase_words = normalized_phrase.split()
            candidate = [word.group().casefold() for word in words[:len(phrase_words)]]
            if candidate == phrase_words:
                # Slice the original string so case, punctuation, numbers, and
                # non-ASCII command text survive wake-word matching.
                remainder = transcript[words[len(phrase_words) - 1].end():]
                prompt = re.sub(r"^[,.:;!?]+(?:\s+|$)", "", remainder).lstrip()
                return WakeWordMatch(
                    phrase=original_phrase,
                    transcript=transcript,
                    prompt_text=prompt or None,
                )
        return None


def build_wake_word_detector(config: WakeWordConfig) -> WakeWordDetector:
    backend = (config.backend or "transcript").strip().lower()
    if backend == "transcript":
        return TranscriptWakeWordDetector(config.phrases)
    raise ValueError(f"Unsupported wake-word backend: {config.backend}")
