from __future__ import annotations

from pathlib import Path
import logging
import tempfile

import numpy as np
import torch

from arietta_voice.assets import ensure_download


class SileroVoiceActivityDetector:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._vad = None

    def ensure_loaded(self) -> None:
        if self._vad is not None:
            return
        from silero_vad import load_silero_vad

        self._vad = load_silero_vad(onnx=True)

    def is_loaded(self) -> bool:
        return self._vad is not None

    def probability(self, chunk: np.ndarray) -> float:
        self.ensure_loaded()
        value = self._vad(torch.from_numpy(chunk), self.sample_rate)
        return value.item() if hasattr(value, "item") else float(value)

    def reset(self) -> None:
        if self._vad is not None:
            self._vad.reset_states()


class SmartTurnEndpointDetector:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._session = None
        self._extractor = None
        self._logger = logging.getLogger("arietta_voice.smart_turn")

    def ensure_loaded(self) -> None:
        if self._session is not None and self._extractor is not None:
            return

        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        destination = Path(tempfile.gettempdir()) / "arietta-smart-turn" / "smart-turn-v3.2-cpu.onnx"
        ensure_download(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
            destination,
        )
        self._logger.info("Loading Smart Turn model from %s", destination)
        self._session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
        self._extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def is_loaded(self) -> bool:
        return self._session is not None and self._extractor is not None

    def probability(self, audio: np.ndarray) -> float:
        self.ensure_loaded()
        max_samples = 8 * self.sample_rate
        trimmed = audio[-max_samples:]
        features = self._extractor(
            trimmed,
            sampling_rate=self.sample_rate,
            max_length=max_samples,
            padding="max_length",
            return_attention_mask=False,
            return_tensors="np",
        )
        outputs = self._session.run(
            None,
            {"input_features": features.input_features.astype(np.float32)},
        )
        return float(outputs[0].flatten()[0])
