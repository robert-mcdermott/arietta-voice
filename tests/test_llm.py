from __future__ import annotations

import pytest

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
        # As in mlx-vlm: how many leading tokens match the last prompt and answer.
        token_ids = None

        def find_prefix_length(self, new_ids):
            if self.token_ids is None:
                return 0
            shared = 0
            for cached, new in zip(self.token_ids, new_ids, strict=False):
                if cached != new:
                    break
                shared += 1
            return shared

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


def test_local_stream_declares_tools_through_the_chat_template(monkeypatch):
    calls = []
    _fake_mlx(monkeypatch, ["<|tool_call>call:t{}<tool_call|>"], calls)
    model = _loaded_gemma()
    tools = [{"type": "function", "function": {"name": "t", "description": "d"}}]
    events = list(model.stream([{"role": "user", "content": "Hi"}], tools=tools))
    assert events[0].text == "<|tool_call>call:t{}<tool_call|>"
    assert model._processor.apply_chat_template.call_args.kwargs["tools"] == tools
    list(model.stream([{"role": "user", "content": "Hi"}]))
    assert "tools" not in model._processor.apply_chat_template.call_args.kwargs
    assert model.native_tool_calls is True


def test_a_repeated_prompt_still_reuses_the_cache(monkeypatch):
    # mlx-vlm reuses a prefix only when the new prompt is longer than it; an
    # exact repeat must keep one token back so it is not processed from scratch.
    _fake_mlx(monkeypatch, ["Answer."], [])
    from arietta_voice.llm import _repeat_safe_prefix_state

    state = _repeat_safe_prefix_state()
    assert state.find_prefix_length([1, 2, 3]) == 0  # Nothing cached yet.
    state.token_ids = [1, 2, 3, 4, 5, 6]  # The last prompt, then its answer.
    assert state.find_prefix_length([1, 2, 3, 4]) == 3  # The same prompt again.
    assert state.find_prefix_length([1, 2, 3, 4, 5, 6]) == 5
    assert state.find_prefix_length([1, 2, 9, 9]) == 2  # Diverges: unchanged.
    assert state.find_prefix_length([1, 2, 3, 4, 5, 6, 7, 8]) == 6  # Longer: unchanged.
    assert state.find_prefix_length([]) == 0


def test_gemma_per_layer_inputs_follow_each_forward_pass():
    # mlx-vlm sliced precomputed per-layer inputs at the cache's absolute offset,
    # which is wrong once a cached prefix is reused; the model must compute them
    # from each pass's own tokens instead.
    from types import SimpleNamespace

    from arietta_voice.llm import _align_per_layer_inputs

    calls = []

    def get_input_embeddings(input_ids, pixel_values=None, **kwargs):
        calls.append(input_ids)
        return SimpleNamespace(inputs_embeds="embeds", per_layer_inputs="precomputed")

    model = SimpleNamespace(get_input_embeddings=get_input_embeddings)
    _align_per_layer_inputs(model)
    features = model.get_input_embeddings("ids", None, mask=None)
    assert features.inputs_embeds == "embeds" and features.per_layer_inputs is None
    assert calls == ["ids"]
    wrapped = model.get_input_embeddings
    _align_per_layer_inputs(model)  # Loading twice wraps once.
    assert model.get_input_embeddings is wrapped
    _align_per_layer_inputs(SimpleNamespace())  # Models without the hook are left alone.


def test_repetition_penalty_is_a_setting(monkeypatch, tmp_path):
    from arietta_voice.config import default_config, validate_config
    from arietta_voice.llm import build_chat_model

    calls = []
    _fake_mlx(monkeypatch, ["Answer."], calls)
    model = _loaded_gemma(repetition_penalty=1.05)
    list(model.stream([{"role": "user", "content": "Synthetic"}]))
    assert calls[0]["repetition_penalty"] == 1.05
    config = default_config(tmp_path)
    assert config.models.repetition_penalty == 1.2
    config.models.repetition_penalty = 1.1
    assert build_chat_model(config.models).repetition_penalty == 1.1
    config.models.repetition_penalty = 0.5
    with pytest.raises(ValueError, match="repetition_penalty"):
        validate_config(config)
