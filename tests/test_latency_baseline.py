"""Pure timing/protocol contracts; no live model or audio access."""
from types import SimpleNamespace
import queue

import numpy as np
import pytest

from scripts.latency_baseline import (
    ReplayMicrophone, BenchmarkComplete, BenchmarkStop, case_order,
    finish_row, summarize_turns,
)


def test_balanced_reproducible_order_excludes_first_turn():
    order = case_order(100)
    assert order == case_order(100)
    assert order[0] == 0
    assert all(order[1:].count(i) == 10 for i in range(10))


def test_conversation_history_is_retained_within_groups_only():
    now = [0.]
    session = SimpleNamespace(turn_count=0)
    reset = lambda: setattr(session, 'turn_count', 0)
    runtime = SimpleNamespace(session=session,
        config=SimpleNamespace(audio=SimpleNamespace(sample_rate=10, chunk_samples=4)),
        reset_session=reset, _open_session=lambda _: None, vad=SimpleNamespace(reset=lambda: None))
    mic = ReplayMicrophone(runtime, [np.ones(4)], [0]*5, conversation_turns=2,
        clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0,now[0]+delay))
    histories = []
    for i in range(5):
        mic.get()
        histories.append(mic.row['history_turns'])
        session.turn_count += 1
        if i < 4: mic.advance()
    assert histories == [0, 0, 1, 0, 1]


def test_shared_fixtures_require_matching_corpus_and_hash(tmp_path):
    import hashlib
    import json
    from scripts.latency_baseline import QUESTIONS, load_shared_fixtures
    path = tmp_path/'fixtures.npz'
    np.savez(path, **{f'q{i}':np.ones(512,dtype=np.float32) for i in range(len(QUESTIONS))})
    meta = {'questions':list(QUESTIONS), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix('.json').write_text(json.dumps(meta))
    assert len(load_shared_fixtures(path)) == 10
    meta['questions'][0] = 'Mismatched text'
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='mismatch'):
        load_shared_fixtures(path)


def test_stage_timings_use_waveform_end_and_exclude_initial_turn():
    row = {'case_id': 0, 'first_turn': False, 'speech_end': 10.,
           'processing': 11., 'tts_ready': 12.5, 'first_write': 13.,
           'first_signal_write': 13.1, 'output_open_ms': 25.,
           'calls': [('endpoint', 10.8, 11.), ('transcribe', 11., 11.1),
                     ('chat', 11.1, 12.), ('tts_next', 12., 12.5)]}
    result = finish_row(row, 14., False)
    assert result['valid']
    assert result['speech_end_to_first_write_ms'] == 3000
    assert result['speech_end_to_processing_ms'] == 1000
    assert result['processing_to_first_write_ms'] == 2000
    first = {**result, 'first_turn': True, 'speech_end_to_first_write_ms': 99999}
    invalid = {**result, 'valid': False}
    summary = summarize_turns([first, invalid, result])
    assert summary['warm_turns'] == 1
    assert summary['metrics_ms']['speech_end_to_first_write_ms']['p95'] == 3000
    assert 'calls' not in result and 'speech_end' not in result
    assert not finish_row(row, 14., True)['valid']
    assert not finish_row({k: v for k, v in row.items() if k != 'first_write'}, 14., False)['valid']


def test_replay_is_paced_zero_padded_and_not_dropped_by_output_drain():
    now = [100.]
    runtime = SimpleNamespace(config=SimpleNamespace(audio=SimpleNamespace(sample_rate=10, chunk_samples=4)),
                              reset_session=lambda: None, _open_session=lambda _: None,
                              vad=SimpleNamespace(reset=lambda: None))
    mic = ReplayMicrophone(runtime, [np.ones(6)], [0], clock=lambda: now[0],
                           sleep=lambda s: now.__setitem__(0, now[0] + s))
    np.testing.assert_array_equal(mic.get(), [1, 1, 1, 1])
    assert now[0] == 100.4
    assert mic.row['speech_end'] == 100.6
    mic.drain()
    with pytest.raises(queue.Empty):
        mic.get_nowait()
    np.testing.assert_array_equal(mic.get(), [1, 1, 0, 0])
    assert now[0] == 100.8
    with pytest.raises(BenchmarkComplete):
        mic.advance()


def test_unrecognized_fixture_cannot_leave_listening_loop_running_forever():
    now = [0.]
    runtime = SimpleNamespace(config=SimpleNamespace(audio=SimpleNamespace(sample_rate=10, chunk_samples=4)),
                              reset_session=lambda: None, _open_session=lambda _: None,
                              vad=SimpleNamespace(reset=lambda: None))
    mic = ReplayMicrophone(runtime, [np.ones(6)], [0], clock=lambda: now[0],
                           sleep=lambda s: now.__setitem__(0, now[0] + s))
    mic.get()
    now[0] = 16
    with pytest.raises(BenchmarkStop):
        mic.get()
