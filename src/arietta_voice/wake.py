from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from arietta_voice.config import WakeWordConfig


NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")
LEADING_FILLERS = {"hey", "hi", "hello", "okay", "ok"}


@dataclass(slots=True, frozen=True)
class WakeWordMatch:
    phrase: str
    transcript: str
    prompt_text: str | None = None


class WakeWordDetector(Protocol):
    def match(self, transcript: str) -> WakeWordMatch | None: ...


def normalize_phrase(value: str) -> str:
    cleaned = NON_ALNUM_RE.sub(" ", value.lower())
    return WHITESPACE_RE.sub(" ", cleaned).strip()


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
        normalized = normalize_phrase(transcript)
        if not normalized:
            return None

        tokens = normalized.split()
        while tokens and tokens[0] in LEADING_FILLERS:
            tokens.pop(0)
        cleaned = " ".join(tokens)
        if not cleaned:
            return None

        for normalized_phrase, original_phrase in self._phrases:
            if cleaned == normalized_phrase:
                return WakeWordMatch(
                    phrase=original_phrase,
                    transcript=transcript,
                    prompt_text=None,
                )
            prefix = f"{normalized_phrase} "
            if cleaned.startswith(prefix):
                prompt = cleaned[len(prefix) :].strip()
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
