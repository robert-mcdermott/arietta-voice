from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from arietta_voice.config import ModelConfig


@dataclass(frozen=True, slots=True)
class ModelEvent:
    # text or done; reasoning/tool payloads are never text events. The
    # conversation adds "filler", spoken while a slow tool works.
    kind: str
    text: str = ""
    stop_reason: str | None = None


class ChatModel(Protocol):
    model_name: str
    provider: str

    def ensure_loaded(self) -> None: ...

    def is_loaded(self) -> bool: ...

    # True when ``stream`` accepts ``tools`` and the model writes native tool calls.
    native_tool_calls: bool

    def stream(
        self, messages, *, max_tokens=200, temperature=0.7, reuse_prefix=False, tools=None
    ): ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str: ...


# Gemma 4 opens a hidden reasoning channel with this token.
REASONING_START = "<|channel>"


class GemmaChatModel:
    provider = "local"
    # Gemma 4's chat template declares tools and the model answers with
    # <|tool_call>call:name{...}<tool_call|> (see model_tooling).
    native_tool_calls = True

    def __init__(
        self, model_name: str, *, reuse_prompt_prefix: bool = True, repetition_penalty: float = 1.2
    ) -> None:
        self.model_name = model_name
        self.reuse_prompt_prefix = reuse_prompt_prefix
        self.repetition_penalty = repetition_penalty
        self._model = None
        self._processor = None
        self._prefix_cache = None
        self._logger = logging.getLogger("arietta_voice.llm")

    def ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        from mlx_vlm import load

        self._logger.info("Loading local chat model %s", self.model_name)
        self._model, self._processor = load(self.model_name)
        _align_per_layer_inputs(self._model)

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        self.ensure_loaded()
        from mlx_vlm import generate

        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        result = generate(
            self._model,
            self._processor,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=self.repetition_penalty,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)

    def stream(self, messages, *, max_tokens=200, temperature=0.7, reuse_prefix=False, tools=None):
        """Yield text events; with reuse_prefix, skip prompt tokens shared with the last call.

        Persona, tool instructions and earlier turns repeat at the start of every
        conversation prompt. Reusing their cached attention state leaves only the
        new tokens to process before the first word. Callers with unrelated
        prompts, such as memory updates, must leave reuse off so they do not
        replace the conversation's cache.
        """
        self.ensure_loaded()
        from mlx_vlm import stream_generate

        template_options = {"tools": tools} if tools else {}
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_options
        )
        reuse_prefix = reuse_prefix and self.reuse_prompt_prefix
        options = {}
        if reuse_prefix:
            if self._prefix_cache is None:
                self._prefix_cache = _repeat_safe_prefix_state()
            options["prompt_cache_state"] = self._prefix_cache
            # Used when nothing can be reused; mlx-vlm swaps in the kept cache
            # when the prompt shares a prefix with the last one.
            options["prompt_cache"] = self._rewindable_cache()
        if tools:
            # With tools declared, Gemma 4 often reasons in a hidden channel before
            # it answers, which took 10 to 20 seconds. Ruling out the token that
            # opens the channel makes it answer or call a tool directly. (Starting
            # its turn with an empty reasoning block did not work: it reasoned
            # anyway, then closed the block.)
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            channel = tokenizer.convert_tokens_to_ids(REASONING_START)
            if isinstance(channel, int) and channel >= 0:
                options["logit_bias"] = {channel: -1e9}
        chunks = stream_generate(
            self._model,
            self._processor,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=self.repetition_penalty,
            **options,
        )
        if reuse_prefix:
            self._prefix_cache.prompt_ids = None
        finished = False
        last = None
        try:
            for chunk in chunks:
                last = chunk
                # mlx-vlm emits incremental detokenizer segments, including its
                # final flush. These are not cumulative response strings.
                if chunk.text:
                    yield ModelEvent("text", chunk.text)
            finished = True
            yield ModelEvent("done")
        finally:
            chunks.close()
            if reuse_prefix and not finished:
                self._keep_stopped_cache(last, options["prompt_cache"])

    def _keep_stopped_cache(self, last, fresh_cache) -> None:
        """Keep the cache of a generation stopped part way (an interruption, a resume).

        mlx-vlm records a cache's tokens only when a generation finishes, so a
        stopped one left the cache ahead of its record, and it was dropped: the
        next turn read the whole prompt again, about 2.5 s instead of 0.5 s.
        The cache holds this prompt and some of its answer. Recording just the
        prompt is enough, since the next generation trims the cache back to the
        part it shares, by the cache's own length.
        """
        state = self._prefix_cache
        prompt_ids = getattr(state, "prompt_ids", None)
        if state is None or last is None or prompt_ids is None:
            self._prefix_cache = None  # Nothing known about the cache: start clean.
            return
        # The kept cache was used when part of the prompt was reused; otherwise
        # the prompt went into the fresh cache passed in.
        used = state.cache if getattr(last, "cached_tokens", 0) else fresh_cache
        state.update(prompt_ids, used)

    def _rewindable_cache(self) -> list:
        """The model's cache, with plain caches in place of sliding-window ring buffers.

        Gemma's sliding-attention layers keep only the last 512 tokens in a ring
        buffer. Once a prompt and its answer pass 512 tokens the ring has
        wrapped, mlx-vlm cannot rewind it to the shared prefix, and every later
        turn reprocessed the whole prompt (about a second). A plain cache keeps
        every token and the model applies the same 512-token window through the
        attention mask, so attention is unchanged and the prefix can always be
        reused. The extra memory is small at conversation lengths.
        """
        from mlx_vlm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

        return [
            KVCache() if isinstance(cache, RotatingKVCache) else cache
            for cache in make_prompt_cache(self._model.language_model)
        ]


def _align_per_layer_inputs(model) -> None:
    """Keep Gemma's per-layer inputs in step with the tokens of each forward pass.

    Gemma 4 E2B/E4B add a per-layer embedding for every token. mlx-vlm 0.7.12
    computes them once for all the tokens a generation is given and, for each
    prefill chunk, slices them at the cache's absolute offset. When a cached
    prompt prefix is reused, only the new tokens are given, yet the offset
    counts the reused prefix, so the slice is shifted and every new token gets
    another token's embedding. Answers then garbled what they copied from the
    prompt, such as the first letters of an email address. Without the precomputed
    copy, the language model computes them from each pass's own token ids,
    which is always aligned and gives identical results otherwise. Text only:
    the precomputed copy differs only for image and audio tokens.
    """
    original = getattr(model, "get_input_embeddings", None)
    if not callable(original) or getattr(original, "_arietta_aligned", False):
        return

    def text_input_embeddings(*args, **kwargs):
        features = original(*args, **kwargs)
        if getattr(features, "per_layer_inputs", None) is not None:
            features.per_layer_inputs = None
        return features

    text_input_embeddings._arietta_aligned = True
    model.get_input_embeddings = text_input_embeddings


def _repeat_safe_prefix_state():
    """mlx-vlm's prompt cache state, made to reuse a prompt that repeats the last one.

    mlx-vlm reuses the cached prefix only when the new prompt is longer than the
    part it shares with the last one. A prompt asked again word for word is all
    shared, so it was processed from scratch: about 4 seconds before the first
    word with the full tool list, against about half a second. Holding back the
    last token keeps the repeat on the reuse path: everything but that token
    comes from the cache.
    """
    from mlx_vlm.generate import PromptCacheState

    class RepeatSafePromptCacheState(PromptCacheState):
        prompt_ids: list | None = None  # This generation's prompt, if it is stopped.

        def find_prefix_length(self, new_ids: list) -> int:
            self.prompt_ids = list(new_ids)
            return min(super().find_prefix_length(new_ids), max(len(new_ids) - 1, 0))

    return RepeatSafePromptCacheState()


class BedrockChatModel:
    provider = "bedrock"
    # Tools are described in the prompt and requested as JSON (model_tooling).
    native_tool_calls = False

    def __init__(
        self,
        model_name: str,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.region_name = region_name
        self.profile_name = profile_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self._client_factory = client_factory
        self._client = None
        self._logger = logging.getLogger("arietta_voice.llm")

    def ensure_loaded(self) -> None:
        if self._client is not None:
            return
        if self._client_factory is not None:
            self._client = self._client_factory()
            return

        import boto3

        session_kwargs: dict[str, str] = {}
        if self.region_name:
            session_kwargs["region_name"] = self.region_name
        if self.profile_name:
            session_kwargs["profile_name"] = self.profile_name
        if self.aws_access_key_id:
            session_kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            session_kwargs["aws_session_token"] = self.aws_session_token

        self._logger.info(
            "Initializing Bedrock chat model %s in region %s",
            self.model_name,
            self.region_name or "<default>",
        )
        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")

    def is_loaded(self) -> bool:
        return self._client is not None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        self.ensure_loaded()
        system_blocks, bedrock_messages = _bedrock_messages(messages)

        request: dict[str, Any] = {
            "modelId": self.model_name,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system_blocks:
            request["system"] = system_blocks

        response = self._client.converse(**request)
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        text_parts = [block["text"] for block in content_blocks if "text" in block]
        return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()

    def stream(self, messages, *, max_tokens=200, temperature=0.7, reuse_prefix=False, tools=None):
        # Prefix reuse is a local-model optimisation; the service manages its own caching.
        self.ensure_loaded()
        system, conversation = _bedrock_messages(messages)
        request = {
            "modelId": self.model_name,
            "messages": conversation,
            "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens},
        }
        if system:
            request["system"] = system
        events = self._client.converse_stream(**request)["stream"]
        reason = None
        try:
            for event in events:
                if any(key.endswith("Exception") for key in event):
                    raise RuntimeError("Bedrock streaming failure")
                start = event.get("contentBlockStart", {}).get("start", {})
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "toolUse" in start or "toolUse" in delta:
                    raise ValueError("Unexpected native tool call in text generation")
                if "text" in delta:
                    if reason is not None:
                        raise ValueError("Text after stream completion")
                    yield ModelEvent("text", delta["text"])
                if "messageStop" in event:
                    reason = event["messageStop"]["stopReason"]
                    if reason not in {"end_turn", "stop_sequence", "max_tokens"}:
                        raise RuntimeError("Bedrock did not finish a text answer")
            if reason is None:
                raise RuntimeError("Incomplete Bedrock stream")
            yield ModelEvent("done", stop_reason=reason)
        finally:
            events.close()


class OllamaChatModel:
    """A model served by Ollama: a larger local model, or one of its cloud models.

    Streams Ollama's /api/chat. Tools are described in the prompt and requested
    as JSON, as with Bedrock. Whether anything leaves the Mac depends on the
    model: an Ollama ":cloud" model runs on Ollama's servers.
    """

    provider = "ollama"
    native_tool_calls = False

    def __init__(self, model_name: str, *, url: str = "http://localhost:11434", timeout_s=60.0):
        self.model_name = model_name
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def ensure_loaded(self) -> None:
        return None  # Ollama loads the model on the first request.

    def is_loaded(self) -> bool:
        return True

    def generate(
        self, messages, *, max_tokens=200, temperature=0.7, reuse_prefix=False, tools=None
    ):
        return "".join(
            event.text
            for event in self.stream(messages, max_tokens=max_tokens, temperature=temperature)
            if event.kind == "text"
        )

    def stream(self, messages, *, max_tokens=200, temperature=0.7, reuse_prefix=False, tools=None):
        import json
        import urllib.request

        body = {
            "model": self.model_name,
            "messages": _plain_messages(messages),
            "stream": True,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        request = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        finished = False
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            for line in response:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(f"Ollama: {data['error']}")
                text = (data.get("message") or {}).get("content") or ""
                if text:
                    yield ModelEvent("text", text)
                if data.get("done"):
                    finished = True
                    break
        if not finished:
            raise RuntimeError("Incomplete Ollama stream")
        yield ModelEvent("done")


def _plain_messages(messages) -> list[dict[str, str]]:
    """Messages as plain role and text: tool calls dropped, tool results as notes."""
    plain = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role == "tool":
            plain.append({"role": "system", "content": f"Tool result:\n{content}"})
        elif role in ("system", "user", "assistant") and isinstance(content, str) and content:
            plain.append({"role": role, "content": content})
    return plain


def build_chat_model(config: ModelConfig) -> ChatModel:
    provider = (config.chat_provider or "local").strip().lower()
    if provider == "ollama":
        return OllamaChatModel(config.chat_model, url=config.ollama_url)
    if provider == "local":
        return GemmaChatModel(
            config.chat_model,
            reuse_prompt_prefix=config.reuse_prompt_prefix,
            repetition_penalty=config.repetition_penalty,
        )
    if provider == "bedrock":
        return BedrockChatModel(
            config.chat_model,
            region_name=config.bedrock_region,
            profile_name=config.bedrock_profile,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            aws_session_token=config.aws_session_token,
        )
    raise ValueError(f"Unsupported chat provider: {config.chat_provider}")


def _bedrock_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    system_blocks: list[dict[str, str]] = []
    bedrock_messages: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "").strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_blocks.append({"text": content})
            continue
        if role not in {"user", "assistant"}:
            continue
        bedrock_messages.append(
            {
                "role": role,
                "content": [{"text": content}],
            }
        )

    return system_blocks, bedrock_messages
