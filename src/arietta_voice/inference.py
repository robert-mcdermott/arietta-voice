"""Heavy models and tool execution live only in the inference child process."""
from __future__ import annotations

import asyncio
from dataclasses import asdict

import numpy as np


class InferenceEngine:
    def __init__(self, config) -> None:
        self.config = config
        self._service = None
        self._stt = None
        self._endpoint = None
        self._speaker = None
        self._tts = None
        self._tts_pending = None
        self._voice_tools = None
        self._voice_model_tools = None
        self._loop = asyncio.new_event_loop()

    def service(self):
        if self._service is None:
            from arietta_voice.api import AriettaApiService
            self._service = AriettaApiService(config=self.config)
        return self._service

    def stt(self):
        if self._stt is None:
            from arietta_voice.stt import MoonshineSpeechToText
            self._stt = MoonshineSpeechToText(self.config.models.stt_language)
        return self._stt

    def endpoint(self):
        if self._endpoint is None:
            from arietta_voice.turn_detection import SmartTurnEndpointDetector
            self._endpoint = SmartTurnEndpointDetector(self.config.audio.sample_rate)
        return self._endpoint

    def speaker(self):
        if self._speaker is None:
            from arietta_voice.tts import KokoroTextToSpeech
            self._speaker = KokoroTextToSpeech(self.config.audio.voice, self.config.audio.tts_speed)
        return self._speaker

    def handle(self, operation, payload):
        if operation == "warmup":
            from arietta_voice.llm import build_chat_model
            self.stt().ensure_loaded()
            service = self.service()
            service._llm = build_chat_model(self.config.models)
            service._llm.ensure_loaded()
            service.knowledge.preload()
            if self.config.audio.smart_turn_enabled:
                self.endpoint().ensure_loaded()
            if self.config.audio.tts_enabled:
                self.speaker().ensure_loaded()
            return None
        if operation == "transcribe":
            return self.stt().transcribe(payload, self.config.audio.sample_rate)
        if operation == "endpoint":
            return self.endpoint().probability(payload)
        if operation == "chat":
            from arietta_voice.api import ChatRequest
            service = self.service()
            service.config.tools.enabled = tuple(payload.pop("enabled_tools"))
            service.config.model_tools.enabled = tuple(payload.pop("enabled_model_tools"))
            if payload.pop("voice_turn", False):
                result = self._voice_chat(payload)
            else:
                result = service._chat_in_process(ChatRequest(**payload), record=False)
            if len(result.response) > 12_000:
                raise ValueError("Response exceeds the speech limit")
            return {**asdict(result), "model_loaded": service._llm is not None and service._llm.is_loaded()}
        if operation == "memory":
            from arietta_voice.llm import build_chat_model
            from arietta_voice.memory import MemoryStore
            service = self.service()
            if service._llm is None:
                service._llm = build_chat_model(self.config.models)
            memory = MemoryStore(self.config.prompts.memory_path, self.config.memory.enabled,
                                 self.config.memory.consolidate_every_turns)
            memory.ensure_exists()
            added = memory.update_from_turn(service._llm, payload["user"], payload["assistant"])
            consolidated = memory.maybe_consolidate(service._llm, payload["turn_count"])
            return {"facts_added": added, "consolidated": consolidated}
        if operation == "tts_start":
            self._tts_pending = None
            self._tts = self.speaker().create_stream(payload).__aiter__()
            return None
        if operation == "tts_next":
            if self._tts_pending is None:
                try:
                    samples, rate = self._loop.run_until_complete(anext(self._tts))
                except StopAsyncIteration:
                    self._tts = None
                    return None
                self._tts_pending = (np.asarray(samples, dtype=np.float32), rate)
            samples, rate = self._tts_pending
            piece = samples[:4096].copy()
            self._tts_pending = (samples[4096:], rate) if len(samples) > 4096 else None
            return piece, rate
        if operation == "tts_close":
            if self._tts is not None and hasattr(self._tts, "aclose"):
                self._loop.run_until_complete(self._tts.aclose())
            self._tts = self._tts_pending = None
            return None
        raise ValueError("Unknown inference operation")

    def _voice_chat(self, payload):
        """Keep voice plugin instances alive across turns, as before isolation."""
        from arietta_voice.api import ChatResult
        from arietta_voice.model_tooling import maybe_generate_with_model_tools
        from arietta_voice.model_tools import ModelToolRegistry
        from arietta_voice.models import ConversationTurn
        from arietta_voice.prompts import sanitize_for_speech
        from arietta_voice.tools import ToolRegistry

        service = self.service()
        knowledge = service.knowledge if self.config.knowledge.enabled else None
        if self._voice_tools is None:
            self._voice_tools = ToolRegistry(self.config, knowledge_retriever=knowledge)
        tool_response = self._voice_tools.maybe_handle(payload["message"])
        used_tool = tool_response.name if tool_response else None
        if tool_response and tool_response.bypass_model:
            return ChatResult(sanitize_for_speech(tool_response.content), "tool", used_tool, None)
        messages = service.prompt_manager.build_messages(
            [ConversationTurn(**turn) for turn in payload["history"]], payload["message"], tool_response)
        if self._voice_model_tools is None:
            self._voice_model_tools = ModelToolRegistry(self.config, knowledge_retriever=knowledge)
        response, model_tool = maybe_generate_with_model_tools(
            messages, model_tool_registry=self._voice_model_tools,
            llm_generate=lambda messages: service._llm_generate(messages, max_tokens=None, temperature=None))
        return ChatResult(sanitize_for_speech(response), "model", used_tool,
                          model_tool.name if model_tool else None)
