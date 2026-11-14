"""Feeding an utterance to the worker's streaming transcription as it is captured.

Used by the listening loop when [audio].streaming_stt is on. Every half second
of new audio is sent to the worker, which transcribes it in the background;
when the turn ends only the rest is sent, and the transcript comes back
sooner. Anything unexpected (the worker busy, a failure, an utterance that is
not the one being fed) leaves the usual whole-utterance transcription to run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger("arietta_voice.stt_feed")
FEED_SAMPLES = 8000  # Half a second at 16 kHz.


class StreamFeed:
    def __init__(self, call: Callable[[str, Any], Any]) -> None:
        self._call = call
        self.generation: int | None = None
        self.state = "idle"  # idle, open, done or failed, for the current utterance
        self.fed_chunks = 0
        self.fed_samples = 0
        self.prepared: tuple[int, int] | None = None

    def feed(self, buffer) -> None:
        """Send new audio from the utterance buffer, half a second at a time."""
        if buffer.generation != self.generation:
            self._abandon()
            self.generation, self.state = buffer.generation, "idle"
            self.fed_chunks = self.fed_samples = 0
            self.prepared = None
        if self.state in ("done", "failed") or not buffer.chunks:
            return
        if self.state == "idle":
            if not self._try("stt_stream_start", None):
                return
            self.state = "open"
        pending = buffer.chunks[self.fed_chunks :]
        size = sum(len(chunk) for chunk in pending)
        if size < FEED_SAMPLES:
            return
        if self._try("stt_stream_add", np.concatenate(pending)):
            self.fed_chunks = len(buffer.chunks)
            self.fed_samples += size

    def prepare(self, buffer) -> None:
        """The utterance about to be transcribed is this buffer, as it is now."""
        self.prepared = (buffer.generation, buffer.samples)

    def take_rest(self, audio: np.ndarray) -> np.ndarray | None:
        """The audio not yet sent if ``audio`` is the streamed utterance, else None.

        Either way the stream is used up: the next attempt on this utterance
        (after the speaker resumed) is transcribed the usual way.
        """
        ready = (
            self.state == "open"
            and self.prepared == (self.generation, len(audio))
            and self.fed_samples <= len(audio)
        )
        self.state = "done" if self.state in ("open", "idle") else self.state
        self.prepared = None
        return audio[self.fed_samples :] if ready else None

    def _abandon(self) -> None:
        if self.state == "open":
            self._try("stt_stream_cancel", None)

    def _try(self, operation: str, payload) -> bool:
        try:
            result = self._call(operation, payload)
        except Exception as error:  # Busy or failed: the usual transcription runs.
            logger.info("Streaming transcription off for this utterance (%s)", type(error).__name__)
            self.state = "failed"
            return False
        if result is False:
            self.state = "failed"
            return False
        return True
