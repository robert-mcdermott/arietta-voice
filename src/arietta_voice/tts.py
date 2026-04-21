from __future__ import annotations

import os
from pathlib import Path
import subprocess

import sounddevice as sd

from arietta_voice.assets import cache_root, ensure_download


def _language_from_voice(voice: str) -> str:
    prefix = voice[:1] if len(voice) > 1 and voice[1] == "_" else ""
    return {
        "a": "en-us",
        "b": "en-gb",
        "e": "es",
        "f": "fr-fr",
        "h": "hi",
        "i": "it",
        "j": "ja",
        "p": "pt-br",
        "z": "cmn",
    }.get(prefix, "en-us")


def language_from_voice(voice: str) -> str:
    return _language_from_voice(voice)


def _configure_espeak() -> None:
    try:
        prefix = subprocess.check_output(["brew", "--prefix", "espeak-ng"], text=True).strip()
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", f"{prefix}/lib/libespeak-ng.dylib")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def _kokoro_asset_paths() -> tuple[Path, Path]:
    model_dir = cache_root() / "kokoro"
    model_path = ensure_download(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        model_dir / "kokoro-v1.0.onnx",
    )
    voices_path = ensure_download(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        model_dir / "voices-v1.0.bin",
    )
    return model_path, voices_path


def load_kokoro_engine():
    _configure_espeak()

    from kokoro_onnx import Kokoro

    model_path, voices_path = _kokoro_asset_paths()
    return Kokoro(str(model_path), str(voices_path))


def list_kokoro_voices() -> list[str]:
    return list(load_kokoro_engine().get_voices())


def spoken_voice_name(voice: str) -> str:
    if "_" not in voice:
        return voice
    prefix, name = voice.split("_", 1)
    prefix_spoken = " ".join(prefix)
    return f"{prefix_spoken} {name.replace('_', ' ')}".strip()


class KokoroTextToSpeech:
    def __init__(self, voice: str, speed: float = 1.0) -> None:
        self.voice = voice
        self.speed = speed
        self._kokoro = None

    def ensure_loaded(self) -> None:
        if self._kokoro is not None:
            return

        self._kokoro = load_kokoro_engine()

    def is_loaded(self) -> bool:
        return self._kokoro is not None

    def speak(self, text: str) -> None:
        self.ensure_loaded()
        samples, sample_rate = self._kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=language_from_voice(self.voice),
        )
        sd.play(samples, sample_rate)
        sd.wait()

    def create_stream(self, text: str):
        self.ensure_loaded()
        return self._kokoro.create_stream(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=language_from_voice(self.voice),
        )
