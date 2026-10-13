import asyncio
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice.config import default_config, load_config, validate_config
from arietta_voice.inference import InferenceEngine
from arietta_voice.tts import build_speaker, KokoroTextToSpeech
from arietta_voice.tts_mlx import MLXKokoroTextToSpeech, validate_trial


def test_default_stays_onnx_and_mlx_requires_explicit_opt_in(tmp_path):
    config = default_config(tmp_path)
    assert isinstance(build_speaker(config.audio), KokoroTextToSpeech)
    config.audio.tts_backend = 'mlx'
    with pytest.raises(ValueError, match='local directory'):
        validate_config(config)
    config.audio.mlx_kokoro_model = str(tmp_path/'Kokoro')
    validate_config(config)
    assert isinstance(build_speaker(config.audio), MLXKokoroTextToSpeech)
    config.audio.tts_backend = 'typo'
    with pytest.raises(ValueError):
        build_speaker(config.audio)


@pytest.mark.parametrize('voice', ['../af_heart', 'jf_alpha', '/tmp/voice.safetensors'])
def test_trial_rejects_unvalidated_languages_or_voice_paths(tmp_path, voice):
    with pytest.raises(ValueError, match='English'):
        validate_trial(tmp_path, voice)


def test_local_override_resolves_model_path_without_editing_base(tmp_path):
    base = tmp_path/'app.toml'
    base.write_text('[audio]\ntts_backend="onnx"\n')
    (tmp_path/'app.local.toml').write_text('[audio]\ntts_backend="mlx"\nmlx_kokoro_model="Kokoro"\n')
    config = load_config(base)
    assert config.audio.tts_backend == 'mlx'
    assert config.audio.mlx_kokoro_model == str(tmp_path/'Kokoro')
    assert 'onnx' in base.read_text()


def test_preflight_fails_without_local_assets_or_optional_dependencies(tmp_path, monkeypatch):
    from arietta_voice import tts_mlx
    path = tmp_path/'Kokoro'
    with pytest.raises(ValueError, match='local config.json'):
        validate_trial(path, 'af_heart')
    (path/'voices').mkdir(parents=True)
    for name in ('config.json', 'kokoro-v1_0.safetensors', 'voices/af_heart.safetensors'):
        (path/name).touch()
    def missing(_):
        raise PackageNotFoundError('synthetic')
    monkeypatch.setattr(tts_mlx, 'version', missing)
    with pytest.raises(ValueError, match='dependencies are missing'):
        validate_trial(path, 'af_heart')


def fake_speaker(samples, rate=24000):
    events = []
    def generate(*args, **kwargs):
        try:
            yield SimpleNamespace(audio=samples, sample_rate=rate)
        finally:
            events.append('closed')
    speaker = MLXKokoroTextToSpeech('/test/Kokoro', 'af_heart')
    speaker.model_path = Path(speaker.model_path)
    speaker.ensure_loaded = lambda: None
    speaker._model = Mock(generate=generate)
    speaker._mx = Mock(eval=lambda _: events.append('evaluated'))
    return speaker, events


def test_mlx_materializes_and_closes_before_reusing_worker(tmp_path):
    speaker, events = fake_speaker(np.ones(10_000, dtype=np.float32)*.1)
    engine = InferenceEngine(default_config(tmp_path))
    engine._speaker = speaker
    try:
        engine.handle('tts_start', 'Synthetic')
        samples, rate = engine.handle('tts_next', None)
        assert events == ['evaluated'] and samples.shape == (4096,) and rate == 24000
        engine.handle('speech_close', None)
        assert events == ['evaluated', 'closed']
        assert engine._tts is engine._tts_pending is None
        engine.handle('tts_start', 'New turn')
        assert len(engine.handle('tts_next', None)[0]) == 4096
    finally:
        engine.handle('speech_close', None)
        engine._loop.close()


@pytest.mark.parametrize('samples,rate', [(np.array([]),24000), (np.array([np.nan]),24000),
    (np.zeros((1,2)),24000), (np.zeros(720001),24000), (np.ones(1),16000)])
def test_mlx_rejects_bad_pcm_and_closes_generator(samples, rate):
    speaker, events = fake_speaker(samples, rate)
    async def consume():
        async for _ in speaker.create_stream('Synthetic'):
            raise AssertionError('Invalid frame escaped')
    with pytest.raises(ValueError, match='TTS output'):
        asyncio.run(consume())
    assert events == ['evaluated', 'closed']


def test_resource_snapshot_does_not_load_models(tmp_path):
    engine = InferenceEngine(default_config(tmp_path))
    try:
        result = engine.handle('resource_snapshot', None)
        assert not any(result[k] for k in ('stt_loaded','chat_loaded','tts_loaded','endpoint_loaded'))
        assert result['tts_backend'] == 'onnx'
        assert engine._speaker is engine._service is None
    finally:
        engine._loop.close()
