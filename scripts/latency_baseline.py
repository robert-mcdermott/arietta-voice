"""Opt-in paced synthetic speech benchmark on the actual Mac; no private context.

uv run --locked scripts/latency_baseline.py --turns 100 --audible
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import random
import re
import sys
import time
from types import SimpleNamespace

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arietta_voice.config import load_config
from arietta_voice.runtime_status import read_runtime_status
from scripts.hardware_validation import (
    isolated_config, collect_environment, protected_fingerprints, validation_lock,
    write_report, rss_sample, summarize_resources, summarize_events, stop_process, _synthesize,
)

# Ten fixed, short, no-tool questions; repeat in seeded balanced order.
QUESTIONS = (
    'What is two plus two?',
    'What is five times three?',
    'How many days are in a week?',
    'What is the opposite of hot?',
    'Name the three primary colors of light.',
    'What do bees make?',
    'What is the capital of France?',
    'How many sides does a triangle have?',
    'Why does ice float on water? Answer in one short sentence.',
    'What is photosynthesis? Answer in one short sentence.',
)
SEED = 1729


class BenchmarkStop(BaseException):
    """Escape the runtime's recoverable Exception handler on a harness failure."""


class BenchmarkComplete(BaseException):
    pass


def case_order(turns):
    order = [i % len(QUESTIONS) for i in range(turns)]
    random.Random(SEED).shuffle(order)
    return [0] + order  # First turn is reported separately, never in warm quantiles.


def quantiles(values):
    import numpy as np
    return {k: round(float(np.percentile(values, p)), 3)
            for k, p in [('p50', 50), ('p95', 95), ('max', 100)]} if values else {}


def summarize_turns(rows):
    warm = [r for r in rows if not r['first_turn'] and r['valid']]
    metrics = ('speech_end_to_first_write_ms', 'speech_end_to_first_signal_write_ms',
               'speech_end_to_tts_ready_ms', 'speech_end_to_processing_ms',
               'processing_to_first_write_ms', 'turn_processing_ms', 'output_open_ms',
               'first_tts_chunk_ms')
    return {'warm_turns': len(warm),
            'metrics_ms': {key: quantiles([r[key] for r in warm]) for key in metrics},
            'stages_ms': {stage: quantiles([r['stages_ms'].get(stage, 0) for r in warm])
                          for stage in ('endpoint', 'transcribe', 'chat', 'tts_start', 'tts_next')},
            'by_case': {str(i): {'n': sum(r['case_id'] == i for r in warm),
                                  **quantiles([r['speech_end_to_first_write_ms'] for r in warm if r['case_id'] == i])}
                        for i in range(len(QUESTIONS))}}


class ReplayMicrophone:
    """Paced array input to the unmodified listening loop, no physical mic.

    During output there is no competing speech. drain/get_nowait intentionally
    return no queued input; this benchmark does not evaluate capture or barge-in.
    """
    dropped_chunks = input_overflows = 0

    def __init__(self, runtime, fixtures, order, clock=time.monotonic, sleep=time.sleep):
        self.runtime, self.fixtures, self.order = runtime, fixtures, order
        self.clock, self.sleep = clock, sleep
        self.index = 0
        self.offset = 0
        self.started = None
        self.row = None

    def get(self, timeout=None):
        import numpy as np
        rate = self.runtime.config.audio.sample_rate
        width = self.runtime.config.audio.chunk_samples
        source = self.fixtures[self.order[self.index]]
        if self.started is None:
            self.runtime.reset_session()
            self.runtime._open_session(None)
            self.runtime.vad.reset()
            self.started = self.clock()
            self.row = {'case_id': self.order[self.index], 'first_turn': self.index == 0,
                        'speech_end': self.started + len(source) / rate, 'calls': []}
        deadline = self.started + (self.offset + width) / rate
        self.sleep(max(0, deadline - self.clock()))
        self.row['max_replay_lag_ms'] = max(self.row.get('max_replay_lag_ms', 0),
                                           (self.clock() - deadline) * 1000)
        if self.clock() > self.row['speech_end'] + 15:
            raise BenchmarkStop('No completed utterance within 15 seconds of fixture end')
        chunk = np.zeros(width, dtype=np.float32)
        piece = source[self.offset:self.offset + width]
        chunk[:len(piece)] = piece
        self.offset += width
        return chunk

    def get_nowait(self):
        raise queue.Empty

    def drain(self):
        pass

    def statistics(self):
        return {'queued_chunks': 0, 'queue_max_chunks': self.runtime.config.audio.queue_max_chunks,
                'dropped_chunks': 0, 'input_overflows': 0, 'synthetic_replay': True}

    def advance(self):
        self.index += 1
        self.offset, self.started, self.row = 0, None, None
        if self.index == len(self.order):
            raise BenchmarkComplete


def finish_row(row, finished, failed):
    stages = defaultdict(float)
    for name, start, end in row['calls']:
        stages[name] += (end - start) * 1000
    required = ('processing', 'first_write', 'first_signal_write', 'tts_ready', 'output_open_ms')
    valid = (not failed and all(k in row for k in required)
             and row.get('first_write', 0) >= row['speech_end']
             and stages['transcribe'] > 0 and stages['chat'] > 0)
    result = {'case_id': row['case_id'], 'first_turn': row['first_turn'], 'valid': valid,
              'stages_ms': dict(stages), 'output_underflows': row.get('output_underflows', 0),
              'endpoint_calls': sum(c[0] == 'endpoint' for c in row['calls']),
              'endpoint_rejections': row.get('endpoint_rejections', 0),
              'max_replay_lag_ms': row.get('max_replay_lag_ms', 0)}
    if valid:
        for name, stamp in [('first_write', 'first_write'), ('first_signal_write', 'first_signal_write'),
                            ('tts_ready', 'tts_ready'), ('processing', 'processing')]:
            result[f'speech_end_to_{name}_ms'] = (row[stamp] - row['speech_end']) * 1000
        result.update(processing_to_first_write_ms=(row['first_write'] - row['processing']) * 1000,
                      turn_processing_ms=(finished - row['processing']) * 1000,
                      output_open_ms=row['output_open_ms'],
                      output_reported_latency_ms=row.get('output_reported_latency_ms'),
                      response_chars=row.get('response_chars'),
                      input_samples=row.get('input_samples'),
                      first_tts_chunk_ms=row.get('first_tts_chunk_ms', 0),
                      transcript_matches_fixture=row.get('transcript_matches_fixture'))
    return result


def run_child(config, directory, turns, audible, idle_seconds):
    os.setsid()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import sounddevice as sd
    import arietta_voice.audio as audio_module
    from arietta_voice.runtime import AriettaVoiceRuntime
    path = directory / 'report.json'
    report = {'status': 'running', 'phase': 'fixtures', 'requested_warm_turns': turns,
              'audible': audible, 'rows': [], 'recordings_saved': False,
              'private_context_loaded': False}
    runtime = None
    original_audio_sd = audio_module.sd
    try:
        write_report(path, report)
        runtime = AriettaVoiceRuntime(config)
        runtime.warmup()
        fixtures = []
        fixture_hashes = []
        for text in QUESTIONS:
            pcm, sr = _synthesize(runtime, text)
            pcm = np.interp(np.arange(int(len(pcm) * config.audio.sample_rate / sr)) * sr / config.audio.sample_rate,
                            np.arange(len(pcm)), pcm).astype(np.float32)
            # Reproducible waveform-end definition: last sample > -60 dB of peak.
            nonquiet = np.flatnonzero(np.abs(pcm) > max(float(np.max(np.abs(pcm))) * .001, 1e-6))
            if not len(nonquiet):
                raise ValueError('Empty synthetic fixture')
            pcm = pcm[:nonquiet[-1] + 1]
            fixtures.append(pcm)
            fixture_hashes.append(hashlib.sha256(pcm.tobytes()).hexdigest())
        runtime.worker.close()
        runtime.chime_player.stop()
        report.update(phase='startup', fixture_sha256=fixture_hashes,
                      fixture_durations_s=[len(a) / config.audio.sample_rate for a in fixtures])
        write_report(path, report)
        # Fresh inference process; OS/model file caches remain warm from preparation.
        runtime = AriettaVoiceRuntime(config)
        runtime.status_publisher.start(runtime)
        started = time.monotonic()
        from arietta_voice.devices import resolve_device
        output = resolve_device(config.audio.output_device, 'output')
        runtime.output_device = runtime.chime_player.device = output['index']
        runtime.warmup()
        report['cached_process_startup_s'] = time.monotonic() - started
        report.update(phase='idle', ready=True)
        write_report(path, report)
        time.sleep(idle_seconds)
        microphone = ReplayMicrophone(runtime, fixtures, case_order(turns))
        runtime.microphone = microphone
        original_call = runtime.worker.call

        def measured_call(operation, payload=None, **kwargs):
            start = time.monotonic()
            value = original_call(operation, payload, **kwargs)
            end = time.monotonic()
            row = microphone.row
            if row is not None:
                # Preserve the baseline's chat-stage total across incremental
                # calls; it is now interleaved with TTS and playback.
                stage = 'chat' if operation in {'chat_start', 'chat_next'} else operation
                row['calls'].append((stage, start, end))
                if operation == 'tts_next' and value is not None:
                    row.setdefault('tts_ready', end)
                    row.setdefault('first_tts_chunk_ms', (end - start) * 1000)
                if operation == 'transcribe':
                    normalize = lambda text: re.sub(r'\W+', ' ', text.casefold()).strip()
                    row['transcript_matches_fixture'] = normalize(value) == normalize(QUESTIONS[row['case_id']])
                if operation == 'chat':
                    row['response_chars'] = len(value['response'])
                if operation == 'chat_next' and value['kind'] == 'done':
                    row['response_chars'] = len(value['result']['response'])
                if operation == 'endpoint' and value < .5:
                    row['endpoint_rejections'] = row.get('endpoint_rejections', 0) + 1
            return value

        runtime.worker.call = measured_call
        original_output = sd.OutputStream

        class MeasuredOutput:
            def __init__(self, *args, **kwargs):
                self.started = time.monotonic()
                self.stream = original_output(*args, **kwargs)

            @property
            def latency(self):
                return self.stream.latency

            def start(self):
                self.stream.start()
                microphone.row['output_open_ms'] = (time.monotonic() - self.started) * 1000
                microphone.row['output_reported_latency_ms'] = float(self.stream.latency) * 1000

            def write(self, data):
                row = microphone.row
                now = time.monotonic()
                row.setdefault('first_write', now)
                if np.any(np.abs(data) > 1e-4):
                    row.setdefault('first_signal_write', now)
                underflow = self.stream.write(data if audible else np.zeros_like(data))
                row['output_underflows'] = row.get('output_underflows', 0) + int(underflow)
                return underflow

            def stop(self):
                return self.stream.stop()

            def abort(self):
                return self.stream.abort()

            def close(self):
                return self.stream.close()

        # Keep sounddevice.play's internal callback stream untouched (chimes).
        class AudioProxy:
            OutputStream = MeasuredOutput

            def __getattr__(self, name):
                if name == 'play' and not audible:
                    return lambda data, *args, **kwargs: sd.play(np.zeros_like(data), *args, **kwargs)
                return getattr(sd, name)

        audio_module.sd = AudioProxy()
        original_process = runtime.process_utterance

        def measured_process(pcm, mic, keys):
            row = microphone.row
            row['processing'] = time.monotonic()
            row['input_samples'] = len(pcm)
            result = original_process(pcm, mic, keys)
            record = finish_row(row, time.monotonic(), runtime.last_error is not None)
            report['rows'].append(record)
            report['summary'] = summarize_turns(report['rows'])
            report['reliability'] = runtime.reliability_snapshot()
            write_report(path, report)
            if not record['valid'] or result.buffered_chunks or result.end_session_requested:
                raise BenchmarkStop('Invalid turn; partial measurements retained')
            microphone.advance()
            return result

        runtime.process_utterance = measured_process
        report['phase'] = 'turns'
        write_report(path, report)
        runtime._listen(microphone, SimpleNamespace(pressed=lambda: False))
    except BenchmarkComplete:
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        import traceback
        traceback.print_exc()
    finally:
        audio_module.sd = original_audio_sd
        if runtime is not None:
            runtime.worker.close()
            runtime.chime_player.stop()
            runtime.microphone = None
            runtime.turn_stage = 'stopped'
            runtime.status_publisher.stop(runtime)
        report.update(phase='stopped', finished_at=datetime.now(timezone.utc).isoformat())
        write_report(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--turns', type=int, default=100)
    parser.add_argument('--audible', action='store_true')
    parser.add_argument('--idle-seconds', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.turns <= 200 or not 0 <= args.idle_seconds <= 60:
        parser.error('turns must be 1..200; idle-seconds must be 0..60')
    root = Path(__file__).resolve().parents[1]
    os.umask(0o077)
    with validation_lock(root / 'tmp/hardware-validation'):
        source = load_config(args.config)
        if read_runtime_status(source)['is_running']:
            parser.error('Stop the normal voice runtime before benchmarking')
        before = protected_fingerprints(root)
        directory = root / 'tmp/latency-baseline' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        directory.mkdir(parents=True, mode=0o700)
        config = isolated_config(source, directory)
        if not config.audio.tts_enabled:
            parser.error('This baseline requires TTS enabled')
        metadata = collect_environment(config, root)
        metadata.update(benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        corpus_sha256=hashlib.sha256(json.dumps(QUESTIONS).encode()).hexdigest(),
                        seed=SEED, silence_ms=config.audio.silence_ms,
                        chime_enabled=config.audio.chime_enabled, temperature=config.models.temperature,
                        protocol='paced synthetic input; fresh active session per turn; first output write, not acoustic onset')
        write_report(directory / 'environment.json', metadata)
        print(f'Baseline directory: {directory}', flush=True)
        process = multiprocessing.get_context('spawn').Process(target=run_child,
            args=(config, directory, args.turns, args.audible, args.idle_seconds))
        samples = []
        error = None
        started = time.monotonic()
        process.start()
        try:
            while process.is_alive():
                path = directory / 'report.json'
                report = json.loads(path.read_text()) if path.exists() else {}
                sample = {'elapsed_seconds': round(time.monotonic() - started, 2),
                          'phase': report.get('phase', 'starting'), 'ready': report.get('ready', False),
                          'heartbeat_age_seconds': read_runtime_status(config).get('heartbeat_age_seconds'),
                          **rss_sample(process.pid)}
                samples.append(sample)
                write_report(directory / 'resources.json', {'samples': samples})
                if len(samples) % 6 == 0:
                    print(json.dumps({'phase': sample['phase'], 'turns': len(report.get('rows', [])),
                                      'rss_mib': sample['rss_mib']}), flush=True)
                if time.monotonic() - started > 1800:
                    raise TimeoutError('Benchmark exceeded 30-minute outer deadline')
                process.join(timeout=5)
        except (Exception, KeyboardInterrupt) as exc:
            error = type(exc).__name__
        finally:
            stop_process(process)
        path = directory / 'report.json'
        report = json.loads(path.read_text()) if path.exists() else {'status': 'failed'}
        report['resources_by_phase'] = {phase: {
            **summarize_resources([s for s in samples if s['phase'] == phase]),
            'sampled_cpu_percent': quantiles([s['cpu_percent'] for s in samples if s['phase'] == phase])}
            for phase in sorted({s['phase'] for s in samples})}
        report['events'] = summarize_events(config.logging.directory)
        try:
            status = json.loads((config.logging.directory / 'runtime_status.json').read_text())
        except (OSError, ValueError):
            status = {}
        report['protected_files_unchanged'] = before == protected_fingerprints(root)
        report['elapsed_s'] = time.monotonic() - started
        report['gates'] = {
            'all_turns': len(report.get('rows', [])) == args.turns + 1,
            'valid_turns': all(r['valid'] for r in report.get('rows', [])),
            'no_output_underflows': not any(r['output_underflows'] for r in report.get('rows', [])),
            'no_content_logged': not report['events']['content_fields_present'],
            'valid_event_log': report['events']['invalid_records'] == 0,
            'redacted_stopped_status': status.get('lifecycle') == 'stopped'
            and not status.get('last_user_transcript') and not status.get('last_assistant_text'),
            'private_files_unchanged': report['protected_files_unchanged'],
            'clean_child_exit': process.exitcode == 0 and error is None,
            'no_failures': report.get('reliability', {}).get('turn_failures') == 0,
            'no_worker_restarts': report.get('reliability', {}).get('worker_restarts') == 0,
        }
        if not all(report['gates'].values()):
            report['status'] = 'failed'
        if error:
            report['error_type'] = error
        write_report(path, report)
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
        return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
