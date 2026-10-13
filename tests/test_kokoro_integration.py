"""Comparison validity checks without native models or hardware."""
from copy import deepcopy

import pytest

from scripts.kokoro_integration import summarize
from scripts.latency_baseline import finish_row


def reports():
    resource = dict(stt_loaded=True, chat_loaded=True, tts_loaded=True, endpoint_loaded=True,
                    peak_rss_mib=100, mlx_peak_mib=200, mlx_active_mib=90)
    row = finish_row(dict(case_id=0, first_turn=False, history_turns=0, speech_end=0,
        processing=1, first_write=2, first_signal_write=2, tts_ready=1.9, output_open_ms=10,
        calls=[('transcribe',1,1.1), ('chat',1.1,1.5)]), 3, False)
    row['model_resources'] = resource
    base = dict(status='passed', gates={'complete': True}, fixture_sha256=['synthetic'],
                conversation_turns=5, requested_warm_turns=1, cached_process_startup_s=1,
                rows=[dict(row, first_turn=True), row])
    return [dict(deepcopy(base), tts_backend=backend) for backend in ('onnx', 'mlx')]


def test_summary_keeps_allocator_and_rss_separate():
    result = summarize(reports())
    assert result['mlx']['worker_peak_rss_mib'] == 100
    assert result['mlx']['worker_mlx_peak_mib'] == 200


@pytest.mark.parametrize('change,message', [
    (lambda r: r[1].update(status='failed'), 'successful'),
    (lambda r: r[1].update(fixture_sha256=['different']), 'fixture'),
    (lambda r: r[1]['rows'].pop(), 'Incomplete'),
    (lambda r: r[1]['rows'][0].update(history_turns=1), 'conversation cases'),
    (lambda r: r[1]['rows'][0]['model_resources'].update(chat_loaded=False), 'resident'),
    (lambda r: r.pop(), 'Both backends'),
    (lambda r: r.append(deepcopy(r[0])), 'Unbalanced'),
])
def test_summary_rejects_invalid_comparisons(change, message):
    data = reports()
    change(data)
    with pytest.raises(ValueError, match=message):
        summarize(data)
