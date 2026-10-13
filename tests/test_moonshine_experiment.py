"""Synthetic experiment contracts; no Moonshine imports or audio devices."""
import hashlib
import json
import multiprocessing
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from scripts.moonshine_experiment import (
    TranscriptCollector, load_fixture, run_bounded, stream_once, word_errors,
)


def line(text, ident=1, complete=False, start=0):
    return SimpleNamespace(text=text, line_id=ident, is_complete=complete, start_time=start)


def event(name, **values):
    return type(name, (), values)()


def test_partial_revisions_replace_text_and_final_snapshot_is_authoritative():
    now = [0.]
    collector = TranscriptCollector(clock=lambda: now[0])
    now[0] = .25
    collector(event('LineStarted', line=line('Good')))
    collector(event('LineTextChanged', line=line('Goodbye')))
    collector(event('LineCompleted', line=line('Goodbye.', complete=True)))
    collector(event('LineCompleted', line=line('Goodbye.', complete=True)))
    assert collector.lines == {1: 'Goodbye.'}
    assert collector.first_partial_ms == 250
    assert collector.partial_changes == 1
    assert collector.completed == {1}
    final = SimpleNamespace(lines=[line('Next.', 2, True, 2), line('Goodbye.', 1, True),
                                  line('Goodbye.', 1, True)])
    assert collector.finish(final) == 'Goodbye. Next.'


@pytest.mark.parametrize('failure', ['callback', 'missing', 'incomplete', 'oversized'])
def test_finalization_rejects_native_errors_and_unfinished_text(failure):
    collector = TranscriptCollector()
    transcript = SimpleNamespace(lines=[line('Ready.', complete=True)])
    if failure == 'callback':
        collector(event('Error', error=ValueError('synthetic')))
    elif failure == 'missing':
        transcript = None
    elif failure == 'incomplete':
        transcript.lines[0].is_complete = False
    else:
        transcript.lines[0].text = 'x'*12001
    with pytest.raises(RuntimeError):
        collector.finish(transcript)


def test_silence_is_valid_and_event_history_is_bounded():
    collector = TranscriptCollector()
    assert collector.finish(SimpleNamespace(lines=[])) == ''
    for i in range(5000):
        collector(event('LineUpdated', line=line('provisional', ident=i)))
    assert len(collector.lines) == 128
    assert sum(collector.events.values()) == 4096
    with pytest.raises(RuntimeError):
        collector.finish(SimpleNamespace(lines=[]))


@pytest.mark.parametrize('failure', ['listener', 'feed', 'stop'])
def test_stream_always_closes_on_native_or_listener_failure(failure):
    class Stream:
        closed = False

        def add_listener(self, listener):
            if failure == 'listener':
                raise RuntimeError('synthetic registration failure')
            self.listener = listener

        def start(self):
            pass

        def add_audio(self, chunk, rate):
            assert rate == 16000 and len(chunk) <= 512
            if failure == 'feed':
                self.listener(event('Error', error=RuntimeError('synthetic')))

        def stop(self):
            return None

        def close(self):
            self.closed = True

    stream = Stream()
    transcriber = SimpleNamespace(create_stream=lambda **kw: stream)
    with pytest.raises(RuntimeError):
        stream_once(transcriber, np.zeros(600), paced=False)
    assert stream.closed


def test_fixture_rejects_escape_tampering_and_excessive_size(tmp_path):
    path = tmp_path/'fixture.wav'
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b'\0\0'*512)
    case = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert len(load_fixture(tmp_path, case)) == 512
    with pytest.raises(ValueError, match='inside'):
        load_fixture(tmp_path, {**case, 'file': '../outside.wav'})
    with pytest.raises(ValueError, match='hash'):
        load_fixture(tmp_path, {**case, 'sha256': 'invalid'})
    path.write_bytes(b'\0'*(16000*30*2+4097))
    with pytest.raises(ValueError, match='size'):
        load_fixture(tmp_path, case)


def test_word_error_metric_exposes_number_formatting_and_silence_hallucinations():
    assert word_errors('Bridget.', 'BRIDGET') == {'edits': 0, 'words': 1, 'exact': True}
    assert word_errors('two plus two', '2 plus 2')['edits'] == 2
    assert word_errors('', 'hallucination')['edits'] == 1
    assert word_errors('one two', 'one')['edits'] == 1


def stuck_worker(args):
    time.sleep(30)


def failed_worker(args):
    raise RuntimeError('synthetic decoder failure')


@pytest.mark.parametrize('target,timeout,error,reason', [
    (stuck_worker, .1, TimeoutError, 'process_timeout'),
    (failed_worker, 10, RuntimeError, 'worker_failure'),
])
def test_process_deadline_and_crash_leave_failure_report_without_orphan(
        tmp_path, target, timeout, error, reason):
    before = {p.pid for p in multiprocessing.active_children()}
    args = SimpleNamespace(output=tmp_path/'result.json', timeout=timeout)
    with pytest.raises(error):
        run_bounded(args, target=target)
    assert json.loads(args.output.read_text()) == {'status': 'failed', 'reason': reason}
    assert {p.pid for p in multiprocessing.active_children()} == before
