"""Offline benchmark contracts; never load a TTS engine or open an audio device."""
import asyncio
import json
from types import SimpleNamespace
import time
import wave

import numpy as np
import pytest

from scripts.kokoro_experiment import CASES, case_order, measure, run_bounded, summarize, validate_pcm, write_wav


def test_case_order_is_repeatable_balanced_and_shared_by_backends():
    first = case_order(3)
    assert first == case_order(3) and len(first) == 3*len(CASES)
    for repeat in range(3):
        assert sorted((name, text) for r, name, text in first if r == repeat) == sorted(CASES)


@pytest.mark.parametrize('pcm,rate', [([], 24000), ([np.nan], 24000), ([np.inf], 24000),
    ([[0.1]], 24000), ([0.1], 16000), (np.zeros(24000*60+1), 24000)])
def test_invalid_pcm_cannot_pass_signal_checks(pcm, rate):
    with pytest.raises(ValueError):
        validate_pcm(pcm, rate)


def test_measure_materialized_audio_and_close_stream():
    clock = [0.]
    closed = []
    async def frames():
        try:
            clock[0] = .2
            yield np.ones(2400, dtype=np.float32)*.1, 24000
            clock[0] = .4
            yield np.ones(2400, dtype=np.float32)*.2, 24000
            clock[0] = .5
        finally:
            closed.append(True)
    pcm, row = asyncio.run(measure(frames(), clock=lambda: clock[0], cpu=lambda: clock[0]*2))
    assert len(pcm) == 4800 and closed == [True]
    assert row['first_audio_ms'] == 200 and row['synthesis_ms'] == 500
    assert row['cpu_ms'] == 1000 and row['rtf'] == 2.5
    assert row['chunks'] == 2 and row['largest_boundary_jump'] == pytest.approx(.1)


def test_empty_or_unbounded_output_is_rejected_and_closed():
    closed = []
    async def frames(count):
        try:
            for _ in range(count):
                yield np.ones(1, dtype=np.float32), 24000
        finally:
            closed.append(True)
    for count in (0, 129):
        with pytest.raises(ValueError):
            asyncio.run(measure(frames(count)))
    assert closed == [True, True]


def test_waveform_metrics_do_not_claim_perceptual_quality(tmp_path):
    async def frames():
        yield np.array([0., 0., 1.1, -1.2, 0.], dtype=np.float32), 24000
    pcm, row = asyncio.run(measure(frames()))
    assert row['clipped_fraction'] == .4 and not row['silent']
    path = tmp_path/'synthetic.wav'
    write_wav(path, pcm)
    with wave.open(str(path), 'rb') as file:
        assert file.getnchannels() == 1 and file.getframerate() == 24000 and file.getnframes() == 5
    assert path.stat().st_mode & 0o777 == 0o600


def hangs(args):
    time.sleep(10)


def test_native_hang_cannot_leave_a_successful_report(tmp_path):
    args = SimpleNamespace(output=tmp_path/'result.json', timeout=.1)
    with pytest.raises(TimeoutError):
        run_bounded(args, target=hangs)
    assert json.loads(args.output.read_text()) == {'status': 'failed', 'reason': 'process_timeout'}


def report(backend):
    return dict(backend=backend, status='completed', corpus_sha256='synthetic', harness_sha256='same',
        voice='af_heart', speed=1., python='3.13', repeats=1, signal_checks_passed=True,
        peak_rss_mib=100, import_and_load_ms=200, first_inference={'first_audio_ms': 300},
        cases=[dict(id=name, repeat=0, first_audio_ms=10, synthesis_ms=20, cpu_ms=30, audio_s=2, rtf=.01)
               for name, _ in CASES])


def test_summary_uses_complete_balanced_warm_cases():
    summary = summarize([report('onnx'), report('mlx')])
    assert summary['onnx']['metrics']['first_audio_ms'] == {'n': 8, 'p50': 10, 'p95': 10, 'maximum': 10}
    assert summary['mlx']['first_inference_ms'] == [300]


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'corpus', 'harness', 'failed', 'unbalanced'])
def test_summary_rejects_incomplete_or_incomparable_runs(fault):
    a, b = report('onnx'), report('mlx')
    reports = [a, b]
    if fault == 'missing': b['cases'].pop()
    elif fault == 'duplicate': b['cases'][-1] = b['cases'][0]
    elif fault == 'corpus': b['corpus_sha256'] = 'other'
    elif fault == 'harness': b['harness_sha256'] = 'other'
    elif fault == 'failed': b['status'] = 'failed'
    else: reports.append(report('onnx'))
    with pytest.raises(ValueError):
        summarize(reports)


def test_listening_level_matching_retains_samples_and_caps_peaks():
    from scripts.kokoro_listening import match_level
    pcm = np.sin(np.arange(2400)*.1).astype(np.float32)*.1
    matched, gain = match_level(pcm)
    assert len(matched) == len(pcm)
    np.testing.assert_allclose(matched, pcm*gain)
    assert np.sqrt(np.mean(matched**2)) == pytest.approx(10**(-25/20))
    sparse = np.zeros(24000, dtype=np.float32)
    sparse[0] = .1
    assert np.max(match_level(sparse)[0]) <= .950001


def test_listening_rejects_tampered_or_escaping_samples(tmp_path):
    import hashlib
    from scripts.kokoro_listening import read_sample
    audio = tmp_path/'sample.wav'
    write_wav(audio, np.ones(100, dtype=np.float32)*.1)
    row = {'id': 'numbers', 'repeat': 0, 'wav': 'sample.wav',
           'wav_sha256': hashlib.sha256(audio.read_bytes()).hexdigest()}
    path = tmp_path/'report.json'
    def save():
        path.write_text(json.dumps({'status': 'completed', 'cases': [row]}))
    save()
    assert len(read_sample(path, 'numbers')) == 100
    row['wav_sha256'] = 'wrong'
    save()
    with pytest.raises(ValueError, match='hash'):
        read_sample(path, 'numbers')
    row['wav'] = '../outside.wav'
    save()
    with pytest.raises(ValueError, match='path'):
        read_sample(path, 'numbers')
