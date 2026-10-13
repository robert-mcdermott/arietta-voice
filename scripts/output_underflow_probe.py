"""Isolated speaker/write probe: fixed PCM controls and streamed MLX answers.

No microphone, recordings, private context or settings writes. Native output is
muted unless --audible is explicitly supplied. Run in the existing MLX trial env.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hardware_validation import isolated_config, protected_fingerprints, validation_lock, write_report

PROMPTS = (
    'Count from one to twenty, saying every number.',
    'Give exactly three short sentences about the sun, the moon, and the stars.',
)


async def scheduled_pcm(pcm, *, gap_s=0.0):
    for repetition in range(2):
        if repetition and gap_s:
            deadline = time.monotonic() + gap_s
            while time.monotonic() < deadline:
                await asyncio.sleep(min(.01, max(0, deadline - time.monotonic())))
                yield None, None
        for offset in range(0, len(pcm), 4096):
            yield pcm[offset:offset + 4096].copy(), 24000


def run(args):
    import numpy as np
    from arietta_voice.audio import ChimePlayer, play_tts_stream_passive
    from arietta_voice.config import load_config
    from arietta_voice.devices import resolve_device
    from arietta_voice.runtime_status import read_runtime_status
    from arietta_voice.streaming import speech_stream
    from arietta_voice.turns import CancellationToken
    from arietta_voice.worker import InferenceWorker
    from scripts.hardware_validation import apply_trial_overrides, collect_environment
    root = Path(__file__).resolve().parents[1]
    directory = args.output.resolve()
    if not directory.is_relative_to(root/'tmp') or directory.exists():
        raise ValueError('Output must be a fresh directory under repository tmp')
    source = load_config()
    if read_runtime_status(source)['is_running']:
        raise ValueError('Stop the normal voice runtime before this probe')
    protected = protected_fingerprints(root)
    directory.mkdir(parents=True, mode=0o700)
    config = isolated_config(source, directory)
    apply_trial_overrides(config, 'mlx', args.model)
    output = resolve_device(config.audio.output_device, 'output')['index']
    worker = InferenceWorker(config)
    report = dict(status='running', audible=args.audible, microphone=False,
                  private_context_loaded=False, recordings_saved=False, rows=[],
                  environment=collect_environment(config, root))
    path = directory/'report.json'
    write_report(path, report)
    try:
        worker.call('warmup', timeout_s=config.runtime.startup_timeout_s)
        # Reproducible quiet tone, no synthesis variability in scheduling controls.
        pcm = (.01*np.sin(2*np.pi*440*np.arange(14400)/24000)).astype(np.float32)
        if not args.audible:
            pcm.fill(0)
        report['control_pcm_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
        for repetition in range(args.repetitions):
            for case in ('continuous', 'producer_gap', 'foreground_stall', 'count', 'sentences'):
                token = CancellationToken()
                started = time.monotonic()
                deadline = started + config.runtime.turn_timeout_s
                operations, result = {}, {}
                def infer(operation, payload=None, *, cancellation):
                    begin = time.monotonic()
                    value = worker.call(operation, payload, cancellation=cancellation,
                                        timeout_s=max(.001, deadline-time.monotonic()))
                    elapsed = (time.monotonic()-begin)*1000
                    row = operations.setdefault(operation, dict(calls=0, total_ms=0., max_ms=0.))
                    row.update(calls=row['calls']+1, total_ms=row['total_ms']+elapsed,
                               max_ms=max(row['max_ms'], elapsed))
                    return value
                if case in ('count', 'sentences'):
                    request = dict(message=PROMPTS[case == 'sentences'], history=[], use_tools=False,
                                   enabled_tools=(), enabled_model_tools=())
                    stream = speech_stream(infer=infer, worker=worker, cancellation=token,
                                           deadline=deadline, request=request, on_result=result.update,
                                           playback_events=True)
                    if not args.audible:
                        async def muted(original):
                            try:
                                async for samples, rate in original:
                                    yield (np.zeros_like(samples), rate) if isinstance(samples, np.ndarray) else (samples, rate)
                            finally:
                                await original.aclose()
                        stream = muted(stream)
                else:
                    stream = scheduled_pcm(pcm, gap_s=.6 if case == 'producer_gap' else 0.)
                polls = 0
                def health():
                    nonlocal polls
                    polls += 1
                    if case == 'foreground_stall' and polls == 8:
                        time.sleep(.35)  # Deliberate foreground starvation, not a producer wait.
                playback = play_tts_stream_passive(tts_stream=stream, chime_player=ChimePlayer(False),
                    output_device=output, cancellation=token, deadline=deadline, health_check=health)
                row = asdict(playback)
                row.pop('buffered_chunks')
                row.update(case=case, repetition=repetition, elapsed_s=time.monotonic()-started,
                           operations=operations)
                if result:
                    row['response_sha256'] = hashlib.sha256(result['response'].encode()).hexdigest()
                report['rows'].append(row)
                write_report(path, report)
                print(json.dumps({k:row[k] for k in ('case','repetition','output_underflows','output_timing')}),flush=True)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        worker.close()
        report['protected_files_unchanged'] = protected == protected_fingerprints(root)
        if not report['protected_files_unchanged']:
            report['status'] = 'failed'
        write_report(path, report)
    return 0 if report['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, choices=range(1, 4), default=2)
    parser.add_argument('--audible', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    with validation_lock(Path(__file__).resolve().parents[1]/'tmp/hardware-validation'):
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
