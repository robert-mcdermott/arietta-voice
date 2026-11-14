from __future__ import annotations

import queue
import threading

import numpy as np

# Partial results are never used; transcribe the whole utterance in one pass at stop().
NO_PARTIAL_UPDATES_S = 3600.0


class MoonshineSpeechToText:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._transcriber = None

    def ensure_loaded(self) -> None:
        if self._transcriber is not None:
            return
        from moonshine_voice import Transcriber, get_model_for_language

        model_path, model_arch = get_model_for_language(self.language)
        self._transcriber = Transcriber(
            model_path=str(model_path),
            model_arch=model_arch,
            # Speaker labels are not used, so skip that work.
            options={"identify_speakers": "false"},
        )

    def is_loaded(self) -> bool:
        return self._transcriber is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe one complete utterance.

        Each utterance gets its own stream. Moonshine's shared batch path keeps
        state between calls, and identical audio has come back with a stray
        leading word depending on what was transcribed before it.
        """
        self.ensure_loaded()
        stream = self._transcriber.create_stream(update_interval=NO_PARTIAL_UPDATES_S)
        try:
            stream.start()
            stream.add_audio(audio.tolist(), sample_rate)
            transcript = stream.stop()
        finally:
            stream.close()
        return " ".join(line.text for line in transcript.lines if line.text).strip()


class StreamingTranscription:
    """One utterance transcribed while it is still being spoken.

    Audio is added from the listening loop as it arrives; a background thread
    hands it to Moonshine, whose stream works through it in passes, so when
    the speaker stops only the last part is left to do (about 150 ms instead
    of 270 to 360 ms for a three-second request). The listening loop never
    waits for a pass.
    """

    UPDATE_INTERVAL_S = 0.5

    def __init__(self, stt: MoonshineSpeechToText, sample_rate: int) -> None:
        stt.ensure_loaded()
        self.sample_rate = sample_rate
        self._stream = stt._transcriber.create_stream(update_interval=self.UPDATE_INTERVAL_S)
        self._stream.start()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="arietta-stt-stream", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while (audio := self._queue.get()) is not None:
            if self._error is not None:
                continue  # Drain; finish() reports the error.
            try:
                self._stream.add_audio(audio.tolist(), self.sample_rate)
            except Exception as error:  # Reported by finish(); the caller falls back.
                self._error = error

    def add(self, audio: np.ndarray) -> None:
        if len(audio):
            self._queue.put(np.asarray(audio, dtype=np.float32))

    def finish(self, rest: np.ndarray, timeout_s: float = 10.0) -> str:
        """Add the last audio, wait for the passes, and return the transcript."""
        self.add(rest)
        self._queue.put(None)
        self._thread.join(timeout_s)
        try:
            if self._thread.is_alive():
                raise TimeoutError("streaming transcription did not keep up")
            if self._error is not None:
                raise RuntimeError("streaming transcription failed") from self._error
            transcript = self._stream.stop()
            return " ".join(line.text for line in transcript.lines if line.text).strip()
        finally:
            self._close()

    def cancel(self) -> None:
        self._queue.put(None)
        self._thread.join(2.0)
        self._close()

    def _close(self) -> None:
        try:
            self._stream.close()
        except Exception:  # Already closed or broken; nothing more to free.
            pass
