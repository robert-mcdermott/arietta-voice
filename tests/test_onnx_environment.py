"""Environment comparisons must measure the same work; no native inference here."""
from copy import deepcopy

import pytest

from scripts.latency_baseline import finish_row
from scripts.onnx_environment import StageTimer, compare_blocks


def blocks():
    row = finish_row(dict(case_id=0, first_turn=False, history_turns=0, speech_end=0,
        processing=1, first_write=2, first_signal_write=2, tts_ready=1.9, output_open_ms=10,
        response_sha256='synthetic', calls=[('transcribe',1,1.1),('chat',1.1,1.5)]), 3, False)
    report = dict(status='passed', gates={'complete':True}, tts_backend='onnx',
        requested_warm_turns=1, rows=[dict(row, first_turn=True), row],
        fixture_sha256=['synthetic'], conversation_turns=5)
    metadata = dict(models={'chat':'synthetic'}, audio={}, chime_enabled=False, temperature=0,
        seed=1729, cached_revisions={}, asset_sha256={}, stt_asset_sha256={}, benchmark_sha256='synthetic')
    return [dict(environment=label, report=deepcopy(report), metadata=deepcopy(metadata))
            for label in ('production','trial')]


def test_comparison_preserves_per_block_values_and_excludes_initial_turn():
    data = blocks()
    data[1]['report']['rows'][0]['speech_end_to_first_write_ms'] = 99999
    data[1]['report']['rows'][1]['speech_end_to_first_write_ms'] = 2200
    result = compare_blocks(data)
    assert result['trial']['warm_turns'] == 1
    assert result['trial']['block_first_write_p50_ms'] == [2200]
    assert result['paired_trial_minus_production_ms']['p50'] == 200


@pytest.mark.parametrize('change,message', [
    (lambda b: b.pop(), 'required'),
    (lambda b: b.append(deepcopy(b[0])), 'Unbalanced'),
    (lambda b: b[1]['report'].update(status='failed'), 'Failed'),
    (lambda b: b[1]['report']['gates'].update(complete=False), 'Failed'),
    (lambda b: b[1]['report']['rows'].pop(), 'Failed'),
    (lambda b: b[1]['report']['rows'][1].update(response_sha256='different'), 'responses'),
    (lambda b: b[1]['report']['rows'][1].update(response_sha256=None), 'Failed'),
    (lambda b: b[1]['report'].update(fixture_sha256=['different']), 'protocol'),
    (lambda b: b[1]['report']['rows'][1].update(history_turns=4), 'protocol'),
    (lambda b: b[1]['metadata']['models'].update(chat='different'), 'metadata'),
])
def test_comparison_rejects_invalid_or_different_work(change, message):
    data = blocks()
    change(data)
    with pytest.raises(ValueError, match=message):
        compare_blocks(data)


def test_probe_stage_wrapper_preserves_results_and_errors():
    timer = StageTimer()
    value = timer.wrap('phonemize', lambda _: 'synthetic phonemes')('synthetic text')
    assert value == 'synthetic phonemes' and len(timer.phonemes) == 1
    def fail():
        raise RuntimeError('synthetic')
    with pytest.raises(RuntimeError, match='synthetic'):
        timer.wrap('onnx_run', fail)()
    assert timer.calls == {'phonemize':1, 'onnx_run':1}
    assert timer.wall['onnx_run'] >= 0 and timer.cpu['onnx_run'] >= 0
    timer.reset()
    assert not timer.calls and not timer.phonemes
