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
