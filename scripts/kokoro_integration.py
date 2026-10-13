"""Prepare identical fixtures, summarize full-turn trials, and check MLX recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kokoro_experiment import ROOT, run_bounded, write_json
from scripts.latency_baseline import QUESTIONS, summarize_turns


def prepare(args):
    import asyncio
    import numpy as np
    from arietta_voice.assets import cache_root
    from arietta_voice.tts import KokoroTextToSpeech
    for name in ('kokoro-v1.0.onnx', 'voices-v1.0.bin'):
        if not (cache_root()/'kokoro'/name).is_file():
            raise ValueError('Prepare cached ONNX assets before this offline trial')
    speaker = KokoroTextToSpeech('af_heart', 1.)
    fixtures = {}
    async def synthesize():
        for index, text in enumerate(QUESTIONS):
            chunks = []
            async for chunk, rate in speaker.create_stream(text):
                chunks.append(chunk)
            pcm = np.concatenate(chunks)
            pcm = np.interp(np.arange(int(len(pcm)*16000/rate))*rate/16000,
                            np.arange(len(pcm)), pcm).astype(np.float32)
            nonquiet = np.flatnonzero(np.abs(pcm) > max(float(np.max(np.abs(pcm)))*.001, 1e-6))
            if not len(nonquiet):
                raise ValueError('Silent fixture')
            fixtures[f'q{index}'] = pcm[:nonquiet[-1]+1]
    asyncio.run(synthesize())
    args.fixture_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez(args.fixture_path, **fixtures)
    write_json(args.output, dict(status='completed', questions=QUESTIONS, voice='af_heart', speed=1.,
        sha256=hashlib.sha256(args.fixture_path.read_bytes()).hexdigest()))


def summarize(reports):
    if not reports or any(r['status'] != 'passed' or not all(r['gates'].values()) for r in reports):
        raise ValueError('Only successful gated trials may be compared')
    if any(len(r['rows']) != r['requested_warm_turns']+1 or
           sum(row['first_turn'] for row in r['rows']) != 1 or
           not all(row['valid'] for row in r['rows']) for r in reports):
        raise ValueError('Incomplete or invalid turn rows')
    for key in ('fixture_sha256', 'conversation_turns', 'requested_warm_turns'):
        if any(r[key] != reports[0][key] for r in reports):
            raise ValueError(f'Mismatched {key}')
    if {r['tts_backend'] for r in reports} != {'onnx', 'mlx'}:
        raise ValueError('Both backends are required')
    if sum(r['tts_backend'] == 'onnx' for r in reports) != sum(r['tts_backend'] == 'mlx' for r in reports):
        raise ValueError('Unbalanced backend blocks')
    expected = [(r['case_id'], r['first_turn'], r['history_turns']) for r in reports[0]['rows']]
    if any([(r['case_id'], r['first_turn'], r['history_turns']) for r in b['rows']] != expected for b in reports):
        raise ValueError('Mismatched conversation cases')
    result = {}
    for backend in sorted({r['tts_backend'] for r in reports}):
        blocks = [r for r in reports if r['tts_backend'] == backend]
        rows = [row for block in blocks for row in block['rows']]
        resources = [row['model_resources'] for row in rows]
        if not all(all(row.get(k) for k in ('stt_loaded', 'chat_loaded', 'tts_loaded', 'endpoint_loaded'))
                   for row in resources):
            raise ValueError('Required models were not resident')
        result[backend] = dict(blocks=len(blocks), **summarize_turns(rows),
            worker_peak_rss_mib=max(r['peak_rss_mib'] for r in resources),
            worker_mlx_peak_mib=max(r['mlx_peak_mib'] for r in resources),
            worker_mlx_active_end_mib=[b['rows'][-1]['model_resources']['mlx_active_mib'] for b in blocks],
            cached_startup_s=[b['cached_process_startup_s'] for b in blocks])
    return result


def recovery(args):
    """Cancel a blocked native synthesis call, then answer in a fresh worker."""
    import time
    from arietta_voice.config import load_config, validate_config
    from arietta_voice.worker import InferenceWorker, WorkerCancelled
    from arietta_voice.turns import CancellationToken
    from scripts.hardware_validation import isolated_config, validation_lock
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    with validation_lock(ROOT/'tmp/hardware-validation'):
        config = isolated_config(load_config(args.config), args.output.with_suffix('')/'profile')
        config.audio.tts_backend = 'mlx'
        config.audio.mlx_kokoro_model = str(args.model.resolve())
        config.audio.voice, config.audio.tts_speed = 'af_heart', 1.
        config.audio.tts_enabled = config.audio.smart_turn_enabled = True
        validate_config(config)
        worker = InferenceWorker(config)
        report = dict(status='failed', playback=False)
        try:
            worker.call('warmup', timeout_s=120)
            original_pid = worker.pid
            worker.call('tts_start', 'Please count slowly from one through twenty and explain each number.', timeout_s=10)
            token, requested = CancellationToken(), []
            def cancel():
                if not requested:
                    requested.append(time.monotonic())
                    token.cancel()
            try:
                worker.call('tts_next', timeout_s=30, cancellation=token, on_wait=cancel)
            except WorkerCancelled:
                report['cancel_to_worker_stopped_ms'] = (time.monotonic()-requested[0])*1000
                report['worker_stopped'] = worker.pid is None
            else:
                raise RuntimeError('Synthesis completed before cancellation; check is inconclusive')
            worker.call('warmup', timeout_s=120)
            report['fresh_worker'] = worker.pid is not None and worker.pid != original_pid
            result = worker.call('chat', dict(message='What is three plus three?', history=(),
                use_tools=False, enabled_tools=(), enabled_model_tools=(), max_tokens=64), timeout_s=60)
            report['response_sha256'] = hashlib.sha256(result['response'].encode()).hexdigest()
            if not result['response'].strip():
                raise RuntimeError('Empty recovery response')
            worker.call('tts_start', result['response'], timeout_s=10)
            samples, deadline = 0, time.monotonic()+30
            while True:
                frame = worker.call('tts_next', timeout_s=max(.001, deadline-time.monotonic()))
                if frame is None:
                    break
                pcm, rate = frame
                if len(pcm) > 4096 or rate != 24000:
                    raise RuntimeError('Invalid recovery frame')
                samples += len(pcm)
            worker.call('speech_close', timeout_s=10)
            report['recovery_samples'] = samples
            report['model_resources'] = worker.call('resource_snapshot', timeout_s=10)
            if not all(report['model_resources'][k] for k in
                       ('stt_loaded', 'chat_loaded', 'tts_loaded', 'endpoint_loaded')):
                raise RuntimeError('Required models were not resident after recovery')
            if not report['worker_stopped'] or not report['fresh_worker'] or not samples:
                raise RuntimeError('Recovery check failed')
            report['status'] = 'passed'
        finally:
            worker.close()
            write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fixture = sub.add_parser('prepare')
    fixture.add_argument('--output', type=Path, required=True, help='Fresh .npz under repository tmp')
    summary = sub.add_parser('summarize')
    summary.add_argument('reports', nargs='+', type=Path)
    summary.add_argument('--output', type=Path, required=True)
    recover = sub.add_parser('recovery')
    recover.add_argument('--config', type=Path)
    recover.add_argument('--model', type=Path, required=True)
    recover.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if not args.output.resolve().is_relative_to(ROOT/'tmp') or args.output.exists():
        parser.error('Use a fresh output path under repository tmp')
    if args.command == 'prepare':
        if args.output.suffix != '.npz' or args.output.with_suffix('.json').exists():
            parser.error('Use a fresh .npz path and matching .json manifest')
        run_bounded(SimpleNamespace(fixture_path=args.output, output=args.output.with_suffix('.json'), timeout=180),
                    target=prepare)
    elif args.command == 'recovery':
        recovery(args)
    else:
        write_json(args.output, summarize([json.loads(p.read_text()) for p in args.reports]))


if __name__ == '__main__':
    main()
