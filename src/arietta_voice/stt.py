from __future__ import annotations

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
