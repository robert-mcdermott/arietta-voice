from unittest.mock import Mock

from arietta_voice.config import default_config
from arietta_voice.conversation import ConversationService
from arietta_voice.inference import InferenceEngine
from arietta_voice.models import ToolResponse


def test_voice_and_console_share_policy_and_request_is_not_mutated(tmp_path):
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    generated = []
    def generate(messages, **options):
        generated.append((messages, options))
        return '**Synthetic** answer'
    engine = InferenceEngine(config)
    engine._service = ConversationService(config, generate=generate)
    payload = dict(message='  Next  ', history=[{'role':'user','content':' Previous '}],
                   use_tools=False, max_tokens=17, temperature=0., enabled_tools=[], enabled_model_tools=[])
    try:
        console = engine.handle('chat', payload)
        voice = engine.handle('chat', {**payload, 'voice_turn': True})
        assert console == voice
        assert generated[0] == generated[1]
        assert generated[0][1] == {'max_tokens':17, 'temperature':0.}
        assert generated[0][0][-1]['content'] == 'Next'
        assert 'enabled_tools' in payload
    finally:
        engine._loop.close()


def test_shared_service_retains_tools_but_drops_disabled_registry(tmp_path, monkeypatch):
    from arietta_voice import tools
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ('synthetic',)
    config.model_tools.enabled = ()
    active = Mock(maybe_handle=Mock(return_value=ToolResponse('synthetic', 'Tool answer')))
    empty = Mock(maybe_handle=Mock(return_value=None))
    factory = Mock(side_effect=[active, empty])
    monkeypatch.setattr(tools, 'ToolRegistry', factory)
    service = ConversationService(config, generate=lambda *a, **kw: 'Model answer')
    assert service.chat(message='First').source == 'tool'
    assert service.chat(message='Second').source == 'tool'
    assert factory.call_count == 1
    config.tools.enabled = ()
    assert service.chat(message='Third').source == 'model'
    assert factory.call_count == 2
    assert active.maybe_handle.call_count == 2


def test_stream_yields_clause_before_model_finishes_and_closes_on_cancel(tmp_path):
    from arietta_voice.llm import ModelEvent
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    progress = []
    def tokens(*a, **kw):
        try:
            yield ModelEvent('text', 'First sentence. ')
            progress.append('continued')
            yield ModelEvent('text', 'Second sentence.')
            yield ModelEvent('done')
        finally:
            progress.append('closed')
    service = ConversationService(config)
    service._llm = Mock(stream=tokens)
    stream = service.stream(message='Synthetic', use_tools=False)
    assert next(stream) == {'kind': 'clause', 'text': 'First sentence.'}
    assert not progress
    stream.close()
    assert progress == ['closed']


def test_tool_planning_is_withheld_and_only_final_answer_streams(tmp_path, monkeypatch):
    from arietta_voice import model_tools
    from arietta_voice.llm import ModelEvent
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ()
    registry = Mock(instruction_text=lambda: 'Plan privately', execute=Mock(return_value=Mock(followup_context='Synthetic tool result')))
    monkeypatch.setattr(model_tools, 'ModelToolRegistry', lambda *a, **kw: registry)
    service = ConversationService(config)
    service._llm = Mock(generate=lambda *a, **kw: '{"tool_call":{"name":"synthetic","arguments":{}}}',
                        stream=lambda *a, **kw: iter([ModelEvent('text', 'Final answer. More.'), ModelEvent('done')]))
    events = list(service.stream(message='Synthetic'))
    assert [e['text'] for e in events if e['kind'] == 'clause'] == ['Final answer.', 'More.']
    assert events[-1]['result']['model_tool'] == 'synthetic'
    registry.execute.assert_called_once_with('synthetic', {})


def test_invalid_tool_payload_never_reaches_speech(tmp_path, monkeypatch):
    import pytest
    from arietta_voice import model_tools
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ()
    monkeypatch.setattr(model_tools, 'ModelToolRegistry', lambda *a, **kw: Mock(instruction_text=lambda: 'Plan'))
    service = ConversationService(config, generate=lambda *a, **kw: '{"tool_call": broken')
    with pytest.raises(ValueError, match='planning'):
        next(service.stream(message='Synthetic'))
