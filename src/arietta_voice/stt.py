from __future__ import annotations

import numpy as np


class MoonshineSpeechToText:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._transcriber = None

    def ensure_loaded(self) -> None:
        if self._transcriber is not None:
            return
        from moonshine_voice import Transcriber, get_model_for_language

        model_path, model_arch = get_model_for_language(self.language)
        self._transcriber = Transcriber(model_path=str(model_path), model_arch=model_arch)

    def is_loaded(self) -> bool:
        return self._transcriber is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.ensure_loaded()
        result = self._transcriber.transcribe_without_streaming(audio.tolist(), sample_rate)
        return " ".join(line.text for line in result.lines if line.text).strip()
