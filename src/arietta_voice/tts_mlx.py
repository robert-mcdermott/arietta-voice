"""Experimental local-only MLX Kokoro adapter, owned by the inference process.

Optional dependencies are isolated from the default ONNX install. Native synthesis
is synchronous; the existing killable worker enforces cancellation and deadlines.
"""
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import re


def validate_trial(model_path, voice):
    path = Path(model_path).expanduser().resolve()
    if not re.fullmatch(r'[ab][fm]_[a-z0-9_]+', voice):
        raise ValueError('MLX trial supports named American/British English voices only')
    for name in ('config.json', 'kokoro-v1_0.safetensors', f'voices/{voice}.safetensors'):
        if not (path/name).is_file():
            raise ValueError(f'MLX trial requires local {name}; download assets before starting')
    # The upstream config omits architecture; its loader infers it from the path.
    if 'kokoro' not in path.name.casefold():
        raise ValueError('MLX model directory name must contain kokoro')
    try:
        for package, expected in (('mlx-audio', '0.5.8'), ('mlx', '0.31.1'),
                                  ('misaki', '0.9.4'), ('en-core-web-sm', '3.8.0')):
            if version(package) != expected:
                raise ValueError(f'MLX trial requires {package}=={expected}')
    except PackageNotFoundError as exc:
        raise ValueError('MLX trial dependencies are missing; use the isolated trial environment') from exc
    return path


class MLXKokoroTextToSpeech:
    def __init__(self, model_path, voice, speed=1.):
        self.model_path, self.voice, self.speed = model_path, voice, speed
        self._model = self._mx = None

    def is_loaded(self):
        return self._model is not None

    def ensure_loaded(self):
        if self.is_loaded():
            return
        path = validate_trial(self.model_path, self.voice)
        if not .5 <= self.speed <= 2:
            raise ValueError('MLX trial speed must be 0.5–2.0')
        # Imports stay inside the child and only happen after explicit opt-in.
        import mlx.core as mx
        import spacy.util
        if not spacy.util.is_package('en_core_web_sm'):
            raise ValueError('Preinstall en_core_web_sm; automatic installation is disabled')
        from mlx_audio.tts.utils import load_model
        self._model = load_model(path)
        self.model_path, self._mx = path, mx

    async def create_stream(self, text):
        self.ensure_loaded()
        import numpy as np
        stream = self._model.generate(text, voice=str(self.model_path/f'voices/{self.voice}.safetensors'),
                                      speed=self.speed, lang_code=self.voice[0])
        try:
            for result in stream:
                # Do not ship lazy GPU arrays across IPC or count them as ready PCM.
                self._mx.eval(result.audio)
                samples = np.asarray(result.audio, dtype=np.float32)
                if (result.sample_rate != 24000 or samples.ndim != 1 or not 0 < len(samples) <= 24000*30
                        or not np.isfinite(samples).all()):
                    raise ValueError('Invalid or oversized MLX TTS output')
                yield samples, result.sample_rate
        finally:
            stream.close()
