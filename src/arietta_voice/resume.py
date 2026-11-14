"""Keep listening after an early end-of-turn decision.

An early decision lets transcription and generation start while the normal
silence window is still running. Until that window has passed the decision is
provisional: if the speaker carries on, the work is thrown away and the same
utterance continues, exactly as it would have without the early check.
"""

from __future__ import annotations

import queue
import threading
import time

# Two 32 ms chunks of speech, so a single noisy chunk does not undo a turn.
RESUME_SPEECH_CHUNKS = 2


class TurnResumed(Exception):
    """The speaker continued; the provisional turn must be discarded."""


class ResumeWatch:
    """Watches the microphone during the provisional part of a turn.

    Not thread-safe: poll it only from the thread that owns the turn.
    """

    def __init__(self, vad, *, window_end: float, clock=time.monotonic) -> None:
        self.vad = vad
        self.window_end = window_end
        self.clock = clock
        self.chunks: list = []
        self.active = True
        self.resumed = False
        self.on_expire = None
        # Set when the turn is being discarded, so the speech producer stops
        # after its current inference call instead of being cancelled mid-call.
        self.discard = threading.Event()
        self._speech = 0

    def poll(self, microphone) -> bool:
        """Read captured audio; return True once the speaker has resumed."""
        if not self.active:
            return self.resumed
        while True:
            try:
                chunk = microphone.get_nowait()
            except queue.Empty:
                break
            self.chunks.append(chunk)
            if self.vad.probability(chunk) > 0.5:
                self._speech += 1
                if self._speech >= RESUME_SPEECH_CHUNKS:
                    self.resumed = True
                    self.active = False
                    self.discard.set()
                    return True
            else:
                self._speech = 0
        if self.clock() >= self.window_end:
            self.finish()
        return False

    def finish(self, *, release: bool = True) -> None:
        """The decision stands: stop watching.

        With ``release``, run whatever was deferred until now (the thinking
        chime). Playback passes False because speech is about to start.
        """
        if not self.active:
            return
        self.active = False
        self.chunks.clear()
        callback, self.on_expire = self.on_expire, None
        if release and callback is not None:
            callback()
