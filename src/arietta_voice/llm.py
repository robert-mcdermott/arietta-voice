from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
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

    def stream(self, messages, *, max_tokens=200, temperature=0.7): ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str: ...


class GemmaChatModel:
    provider = "local"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
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

    def stream(self, messages, *, max_tokens=200, temperature=0.7):
        self.ensure_loaded()
        from mlx_vlm import stream_generate

        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        chunks = stream_generate(self._model, self._processor, prompt,
                                 max_tokens=max_tokens, temperature=temperature,
                                 repetition_penalty=1.2)
        try:
            for chunk in chunks:
                # mlx-vlm emits incremental detokenizer segments, including its
                # final flush. These are not cumulative response strings.
                if chunk.text:
                    yield ModelEvent("text", chunk.text)
            yield ModelEvent("done")
        finally:
            chunks.close()

    def generate_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        max_tokens: int = 120,
        temperature: float = 0.0,
    ) -> str:
        self.ensure_loaded()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted_prompt = apply_chat_template(
            self._processor,
            self._model.config,
            prompt,
            num_images=1,
        )

        result = generate(
            self._model,
            self._processor,
            formatted_prompt,
            image=str(image_path),
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.1,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)


class BedrockChatModel:
    provider = "bedrock"

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


    def stream(self, messages, *, max_tokens=200, temperature=0.7):
        self.ensure_loaded()
        system, conversation = _bedrock_messages(messages)
        request = {"modelId": self.model_name, "messages": conversation,
                   "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens}}
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
        return GemmaChatModel(config.chat_model)
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


def _bedrock_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
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
