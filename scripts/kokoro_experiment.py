"""Opt-in Kokoro comparison: synthetic text, local assets, no app config or audio devices.

Use the locked project environment for ONNX and a separate pinned MLX environment.
See evals/KOKORO_EXPERIMENT.md for setup, metric boundaries and listening protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('acknowledgement', "Yes? I'm here. How can I help?"),
    ('answer', 'The answer is six. Three plus three equals six.'),
    ('numbers', 'The temperature is minus 5.2 degrees, and the total is $12.50.'),
    ('date', 'Your appointment is on October 12, 2026, at 3:45 p.m.'),
    ('names', 'Bridget uses Arietta, Moonshine, and Kokoro for spoken conversations.'),
    ('technical', 'Deploy Kubernetes with PostgreSQL, then check the API and USB connection.'),
    ('question', 'Did you mean fifteen or fifty? Please tell me which one you prefer.'),
    ('paragraph', 'The garden is quiet this morning. Sunlight reaches the leaves, and a small bird sings nearby. '
                  'After watering the plants, close the gate and bring the blue notebook inside.'),
)
FIRST_TEXT = 'Hello. I am ready to help you today.'
MODEL_REVISION = 'a71e4d38b236d968966a2002c4c895dbd12b1c3c'
SEED = 1729


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('w') as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def fingerprint(paths):
    result = {}
    for path in paths:
        with path.open('rb') as handle:
            result[path.name] = {'bytes': path.stat().st_size,
                                 'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()}
    return result


def case_order(repeats):
    result = []
    rng = random.Random(SEED)
    for repeat in range(repeats):
        rows = list(CASES)
        rng.shuffle(rows)
        result.extend((repeat, name, text) for name, text in rows)
    return result


def validate_pcm(pcm, rate):
    import numpy as np
    pcm = np.asarray(pcm, dtype=np.float32)
    if rate != 24000 or pcm.ndim != 1 or not 0 < len(pcm) <= 24000*60:
        raise ValueError('Expected nonempty mono 24 kHz audio, at most 60 seconds')
    if not np.isfinite(pcm).all():
        raise ValueError('Nonfinite waveform')
    return pcm


async def measure(stream, *, clock=time.perf_counter, cpu=time.process_time):
    import numpy as np
    chunks, arrivals = [], []
    started, cpu_start = clock(), cpu()
    try:
        async for pcm, rate in stream:
            ready = clock()
            pcm = validate_pcm(pcm, rate)
            if len(chunks) >= 128 or sum(len(c) for c in chunks)+len(pcm) > 24000*60:
                raise ValueError('Utterance exceeded the experiment audio bound')
            chunks.append(pcm)
            arrivals.append((ready-started)*1000)
        elapsed, cpu_elapsed = clock()-started, cpu()-cpu_start
    finally:
        await stream.aclose()
    if not chunks:
        raise ValueError('No audio generated')
    pcm = np.concatenate(chunks)
    duration = len(pcm)/24000
    # Diagnostics, not perceptual-quality scores. Raw output is never normalized.
    active = np.flatnonzero(np.abs(pcm) > .001)
    boundaries = np.cumsum([len(c) for c in chunks])[:-1]
    return pcm, dict(first_audio_ms=arrivals[0], synthesis_ms=elapsed*1000,
        cpu_ms=cpu_elapsed*1000, audio_s=duration, rtf=elapsed/duration,
        chunks=len(chunks), chunk_arrival_ms=arrivals,
        chunk_audio_s=[len(c)/24000 for c in chunks],
        peak=float(np.max(np.abs(pcm))), rms=float(np.sqrt(np.mean(pcm.astype(np.float64)**2))),
        clipped_fraction=float(np.mean(np.abs(pcm) >= 1)), silent=not len(active),
        leading_silence_ms=float(active[0]/24) if len(active) else duration*1000,
        trailing_silence_ms=float((len(pcm)-1-active[-1])/24) if len(active) else duration*1000,
        largest_boundary_jump=float(max((abs(pcm[n]-pcm[n-1]) for n in boundaries), default=0)))


def write_wav(path, pcm):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes((np.clip(pcm, -1, 1)*32767).astype('<i2').tobytes())
    os.chmod(path, 0o600)


def load_backend(args):
    import numpy as np
    if args.backend == 'onnx':
        if version('kokoro-onnx') != '0.5.0':
            raise ValueError('Expected production kokoro-onnx 0.5.0')
        from kokoro_onnx import Kokoro
        sys.path.insert(0, str(ROOT/'src'))
        from arietta_voice.tts import _configure_espeak
        _configure_espeak()
        engine = Kokoro(str(args.model), str(args.voices))
        async def frames(text):
            stream = engine.create_stream(text, voice='af_heart', speed=1.0, lang='en-us')
            try:
                async for pcm, rate in stream:
                    yield pcm, rate
            finally:
                await stream.aclose()
        return frames, {'packages': {n: version(n) for n in ('kokoro-onnx', 'onnxruntime', 'numpy')},
                        'providers': engine.sess.get_providers()}, lambda: {}
    if version('mlx-audio') != '0.5.8' or version('mlx') != '0.31.1':
        raise ValueError('Expected pinned MLX candidate versions')
    import mlx.core as mx
    mx.random.seed(SEED)
    import spacy.util
    if not spacy.util.is_package('en_core_web_sm'):
        raise ValueError('Install the pinned English G2P model before running; no auto-install allowed')
    from mlx_audio.tts.utils import load_model
    from mlx.utils import tree_flatten
    engine = load_model(args.model)
    voice = args.model/'voices/af_heart.safetensors'
    if not voice.is_file():
        raise ValueError('Missing explicit local voice')
    async def frames(text):
        stream = engine.generate(text, voice=str(voice.resolve()), speed=1.0, lang_code='a')
        try:
            for result in stream:
                mx.eval(result.audio)
                yield np.asarray(result.audio, dtype=np.float32), result.sample_rate
        finally:
            stream.close()
    def memory():
        return dict(mlx_active_mib=mx.get_active_memory()/2**20,
                    mlx_peak_mib=mx.get_peak_memory()/2**20,
                    mlx_cache_mib=mx.get_cache_memory()/2**20)
    return frames, {'packages': {n: version(n) for n in ('mlx-audio', 'mlx', 'numpy', 'misaki', 'spacy', 'en-core-web-sm')},
                    'parameter_dtypes': sorted({str(v.dtype) for _, v in tree_flatten(engine.parameters())})}, memory


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HOME=str(ROOT/'tmp/kokoro-experiment/hf-cache'))
    report = dict(backend=args.backend, status='running', python=platform.python_version(),
        platform=platform.platform(), voice='af_heart', speed=1., seed=SEED, repeats=args.repeats,
        corpus_sha256=hashlib.sha256(json.dumps(CASES).encode()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    paths = ([args.model, args.voices] if args.backend == 'onnx' else
             [args.model/'config.json', args.model/'kokoro-v1_0.safetensors', args.model/'voices/af_heart.safetensors'])
    report['assets'] = fingerprint(paths)
    started = time.perf_counter()
    frames, details, memory = load_backend(args)
    report.update(details, import_and_load_ms=(time.perf_counter()-started)*1000)
    async def generate():
        _, report['first_inference'] = await measure(frames(FIRST_TEXT))
        for repeat, name, text in case_order(args.repeats):
            pcm, row = await measure(frames(text))
            row.update(id=name, repeat=repeat, **memory())
            if repeat == 0:
                path = args.output.parent/(args.output.stem+'-audio')/(name+'.wav')
                write_wav(path, pcm)
                row['wav'] = str(path.relative_to(args.output.parent))
                row['wav_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            report['cases'].append(row)
            write_json(args.output, report)
    asyncio.run(generate())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report.update(peak_rss_mib=usage.ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024),
                  process_cpu_s=usage.ru_utime+usage.ru_stime, status='completed', **memory())
    report['signal_checks_passed'] = all(not row['silent'] and row['clipped_fraction'] == 0
                                        for row in [report['first_inference'], *report['cases']])
    write_json(args.output, report)


def summarize(reports):
    """Reject partial/mismatched runs before aggregating paired synthetic cases."""
    import numpy as np
    if {r['backend'] for r in reports} != {'onnx', 'mlx'}:
        raise ValueError('Both backends are required')
    for key in ('corpus_sha256', 'harness_sha256', 'voice', 'speed', 'python', 'repeats'):
        if len({r[key] for r in reports}) != 1:
            raise ValueError(f'Mismatched {key}')
    def percentiles(values):
        return dict(n=len(values), p50=float(np.percentile(values, 50)),
                    p95=float(np.percentile(values, 95)), maximum=float(max(values)))
    result = {}
    for backend in ('onnx', 'mlx'):
        blocks = [r for r in reports if r['backend'] == backend]
        rows = []
        for block in blocks:
            expected = sorted((r, name) for r, name, text in case_order(block['repeats']))
            if block['status'] != 'completed' or sorted((r['repeat'], r['id']) for r in block['cases']) != expected:
                raise ValueError('Incomplete or duplicate run')
            rows.extend(block['cases'])
        result[backend] = dict(blocks=len(blocks), signal_checks_passed=all(r['signal_checks_passed'] for r in blocks),
            metrics={key: percentiles([r[key] for r in rows]) for key in
                     ('first_audio_ms', 'synthesis_ms', 'cpu_ms', 'audio_s', 'rtf')},
            peak_rss_mib=[r['peak_rss_mib'] for r in blocks],
            import_and_load_ms=[r['import_and_load_ms'] for r in blocks],
            first_inference_ms=[r['first_inference']['first_audio_ms'] for r in blocks],
            by_case={name: {key: percentiles([r[key] for r in rows if r['id'] == name])
                           for key in ('first_audio_ms', 'synthesis_ms', 'cpu_ms', 'audio_s', 'rtf')}
                     for name, _ in CASES})
    if result['onnx']['blocks'] != result['mlx']['blocks']:
        raise ValueError('Unbalanced backend blocks')
    return result


def run_bounded(args, target=run):
    process = multiprocessing.get_context('spawn').Process(target=target, args=(args,))
    process.start()
    try:
        process.join(args.timeout)
        if process.is_alive():
            write_json(args.output, {'status': 'failed', 'reason': 'process_timeout'})
            raise TimeoutError('Experiment process exceeded deadline')
        if process.exitcode:
            write_json(args.output, {'status': 'failed', 'reason': 'worker_failure'})
            raise RuntimeError('Experiment worker failed; inspect its local log')
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['onnx', 'mlx'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--voices', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    os.umask(0o077)
    if not args.output.resolve().is_relative_to(ROOT/'tmp') or args.output.exists():
        parser.error('Use a fresh output file under repository tmp')
    if not 1 <= args.repeats <= 10 or not 30 <= args.timeout <= 600:
        parser.error('repeats must be 1..10 and timeout 30..600 seconds')
    if not args.model.exists() or (args.backend == 'onnx' and (args.voices is None or not args.voices.is_file())):
        parser.error('Explicit existing local model/voice assets are required')
    run_bounded(args)


if __name__ == '__main__':
    main()
