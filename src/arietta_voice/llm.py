from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from arietta_voice.config import ModelConfig


@dataclass(frozen=True, slots=True)
class ModelEvent:
    kind: str  # text or done; reasoning/tool payloads are never text events.
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

    def __init__(self, model_name: str, *, reuse_prompt_prefix: bool = True) -> None:
        self.model_name = model_name
        self.reuse_prompt_prefix = reuse_prompt_prefix
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
            repetition_penalty=1.2,
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
                from mlx_vlm.generate import PromptCacheState

                self._prefix_cache = PromptCacheState()
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
            repetition_penalty=1.2,
            **options,
        )
        finished = False
        try:
            for chunk in chunks:
                # mlx-vlm emits incremental detokenizer segments, including its
                # final flush. These are not cumulative response strings.
                if chunk.text:
                    yield ModelEvent("text", chunk.text)
            finished = True
            yield ModelEvent("done")
        finally:
            chunks.close()
            if reuse_prefix and not finished:
                # A stopped generation has advanced the cache past the tokens
                # recorded for it. Start clean rather than reuse a mismatch.
                self._prefix_cache = None

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


def build_chat_model(config: ModelConfig) -> ChatModel:
    provider = (config.chat_provider or "local").strip().lower()
    if provider == "local":
        return GemmaChatModel(config.chat_model, reuse_prompt_prefix=config.reuse_prompt_prefix)
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
