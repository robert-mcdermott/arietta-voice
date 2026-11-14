from __future__ import annotations

from arietta_voice.config import ModelConfig
from arietta_voice.llm import BedrockChatModel, GemmaChatModel, _bedrock_messages, build_chat_model


def test_build_chat_model_selects_provider() -> None:
    local = build_chat_model(ModelConfig(chat_provider="local", chat_model="mlx-community/gemma-4"))
    bedrock = build_chat_model(
        ModelConfig(
            chat_provider="bedrock",
            chat_model="us.anthropic.claude-sonnet-4-6",
            bedrock_region="us-west-2",
        )
    )

    assert isinstance(local, GemmaChatModel)
    assert isinstance(bedrock, BedrockChatModel)


def test_bedrock_messages_split_system_and_chat_roles() -> None:
    system, messages = _bedrock_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    )

    assert system == [{"text": "You are helpful."}]
    assert messages == [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]


def test_bedrock_chat_model_generates_text_from_converse() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.request = None

        def converse(self, **kwargs):
            self.request = kwargs
            return {
                "output": {
                    "message": {
                        "content": [
                            {"text": "Hello from Bedrock."},
                        ]
                    }
                }
            }

    fake_client = FakeClient()
    model = BedrockChatModel(
        "us.anthropic.claude-sonnet-4-6",
        region_name="us-west-2",
        client_factory=lambda: fake_client,
    )

    response = model.generate(
        [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=111,
        temperature=0.25,
    )

    assert response == "Hello from Bedrock."
    assert fake_client.request is not None
    assert fake_client.request["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert fake_client.request["system"] == [{"text": "System prompt"}]
    assert fake_client.request["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]
    assert fake_client.request["inferenceConfig"] == {"temperature": 0.25, "maxTokens": 111}


def _fake_mlx(monkeypatch, chunks, calls):
    """Stand in for mlx-vlm: record options and yield scripted text segments."""
    import sys
    import types
    from types import SimpleNamespace

    class PromptCacheState:
        pass

    def stream_generate(model, processor, prompt, **options):
        calls.append(options)
        for text in chunks:
            yield SimpleNamespace(text=text)

    class KVCache:
        pass

    class RotatingKVCache:
        pass

    package = types.ModuleType("mlx_vlm")
    package.stream_generate = stream_generate
    package.__path__ = []
    generate = types.ModuleType("mlx_vlm.generate")
    generate.PromptCacheState = PromptCacheState
    models = types.ModuleType("mlx_vlm.models")
    models.__path__ = []
    cache = types.ModuleType("mlx_vlm.models.cache")
    cache.KVCache, cache.RotatingKVCache = KVCache, RotatingKVCache
    # A model with one full-attention and one sliding-window layer.
    cache.make_prompt_cache = lambda language_model: [KVCache(), RotatingKVCache()]
    for name, module in (
        ("mlx_vlm", package),
        ("mlx_vlm.generate", generate),
        ("mlx_vlm.models", models),
        ("mlx_vlm.models.cache", cache),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    PromptCacheState.KVCache = KVCache
    return PromptCacheState


def _loaded_gemma(**options):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from arietta_voice.llm import GemmaChatModel

    model = GemmaChatModel("synthetic", **options)
    model.ensure_loaded = lambda: None  # The fake below stands in for the loaded model.
    model._model = SimpleNamespace(language_model=object())
    model._processor = Mock(apply_chat_template=Mock(return_value="prompt"))
    return model


def test_local_stream_reuses_one_prefix_cache_across_conversation_turns(monkeypatch):
    calls = []
    state_type = _fake_mlx(monkeypatch, ["Four", "."], calls)
    model = _loaded_gemma()
    messages = [{"role": "user", "content": "Synthetic"}]

    events = list(model.stream(messages, reuse_prefix=True))
    assert [(event.kind, event.text) for event in events] == [
        ("text", "Four"),
        ("text", "."),
        ("done", ""),
    ]
    list(model.stream(messages, reuse_prefix=True))
    assert isinstance(calls[0]["prompt_cache_state"], state_type)
    assert calls[0]["prompt_cache_state"] is calls[1]["prompt_cache_state"]
    # Sliding-window ring buffers are replaced by plain caches, which can be
    # rewound to a shared prefix at any length.
    assert [type(cache) for cache in calls[0]["prompt_cache"]] == [state_type.KVCache] * 2

    # Unrelated prompts, such as memory updates, must not touch the cache.
    list(model.stream(messages))
    assert "prompt_cache_state" not in calls[2] and "prompt_cache" not in calls[2]
    assert model._prefix_cache is calls[0]["prompt_cache_state"]


def test_stopped_local_generation_discards_the_prefix_cache(monkeypatch):
    calls = []
    _fake_mlx(monkeypatch, ["First", " second"], calls)
    model = _loaded_gemma()
    messages = [{"role": "user", "content": "Synthetic"}]

    stream = model.stream(messages, reuse_prefix=True)
    assert next(stream).text == "First"
    stream.close()  # Cancelled mid-answer: the cache no longer matches its record.
    assert model._prefix_cache is None

    list(model.stream(messages, reuse_prefix=True))
    assert calls[1]["prompt_cache_state"] is not calls[0]["prompt_cache_state"]
    assert model._prefix_cache is calls[1]["prompt_cache_state"]


def test_prefix_reuse_can_be_disabled_in_configuration(monkeypatch, tmp_path):
    from arietta_voice.config import default_config
    from arietta_voice.llm import build_chat_model

    calls = []
    _fake_mlx(monkeypatch, ["Answer."], calls)
    model = _loaded_gemma(reuse_prompt_prefix=False)
    list(model.stream([{"role": "user", "content": "Synthetic"}], reuse_prefix=True))
    assert "prompt_cache_state" not in calls[0]

    config = default_config(tmp_path)
    assert build_chat_model(config.models).reuse_prompt_prefix is True
    config.models.reuse_prompt_prefix = False
    assert build_chat_model(config.models).reuse_prompt_prefix is False
