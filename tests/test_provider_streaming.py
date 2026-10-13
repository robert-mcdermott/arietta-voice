import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arietta_voice.llm import BedrockChatModel, GemmaChatModel, ModelEvent


class Events:
    def __init__(self, values):
        self.values = iter(values)
        self.closed = False
    def __iter__(self):
        return self.values
    def close(self):
        self.closed = True


def test_mlx_deltas_and_final_flush_are_not_duplicated_and_close_on_cancel(monkeypatch):
    closed = []
    def chunks(*args, **kwargs):
        assert kwargs['max_tokens'] == 17
        assert kwargs['temperature'] == 0
        try:
            for text in ['Hello', ' world', '.']:
                yield SimpleNamespace(text=text)
        finally:
            closed.append(True)
    monkeypatch.setitem(sys.modules, 'mlx_vlm', SimpleNamespace(stream_generate=chunks))
    model = GemmaChatModel('synthetic')
    model.ensure_loaded = lambda: None
    model._model, model._processor = object(), Mock()
    events = list(model.stream([], max_tokens=17, temperature=0))
    assert ''.join(e.text for e in events) == 'Hello world.'
    assert events[-1].kind == 'done'
    partial = model.stream([], max_tokens=17, temperature=0)
    assert next(partial).text == 'Hello'
    partial.close()
    assert closed == [True, True]


def test_bedrock_stream_maps_text_ignores_reasoning_and_closes():
    events = Events([
        {'contentBlockDelta': {'delta': {'reasoningContent': {'text': 'PRIVATE reasoning'}}}},
        {'contentBlockDelta': {'delta': {'text': 'Hello '}}},
        {'contentBlockDelta': {'delta': {'text': 'world.'}}},
        {'messageStop': {'stopReason': 'end_turn'}},
        {'metadata': {'usage': {}}},
    ])
    client = Mock(converse_stream=Mock(return_value={'stream': events}))
    model = BedrockChatModel('synthetic', client_factory=lambda: client)
    result = list(model.stream([{'role':'system','content':'Synthetic'}, {'role':'user','content':'Hi'}], max_tokens=17, temperature=0))
    assert result == [ModelEvent('text','Hello '), ModelEvent('text','world.'), ModelEvent('done',stop_reason='end_turn')]
    assert client.converse_stream.call_args.kwargs['system'] == [{'text':'Synthetic'}]
    assert events.closed


@pytest.mark.parametrize('tail', [[], [{'modelStreamErrorException': {}}],
    [{'messageStop': {'stopReason': 'guardrail_intervened'}}],
    [{'contentBlockDelta': {'delta': {'toolUse': {'input': '{}'}}}}]])
def test_bedrock_partial_or_failed_stream_never_completes(tail):
    events = Events([{'contentBlockDelta': {'delta': {'text': 'Partial'}}}, *tail])
    model = BedrockChatModel('synthetic', client_factory=lambda: Mock(converse_stream=lambda **kw: {'stream':events}))
    stream = model.stream([])
    assert next(stream).text == 'Partial'
    with pytest.raises((ValueError, RuntimeError)):
        list(stream)
    assert events.closed


def test_bedrock_cancel_closes_transport():
    events = Events([{'contentBlockDelta': {'delta': {'text': 'Partial'}}}])
    model = BedrockChatModel('synthetic', client_factory=lambda: Mock(converse_stream=lambda **kw: {'stream':events}))
    stream = model.stream([])
    next(stream)
    stream.close()
    assert events.closed
