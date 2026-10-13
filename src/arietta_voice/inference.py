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
        self._chat = None
        self._memory = self._memory_store = self._memory_proposal = None
        self._loop = asyncio.new_event_loop()

    def service(self):
        if self._service is None:
            from arietta_voice.conversation import ConversationService
            self._service = ConversationService(self.config)
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
        if operation in {"chat", "chat_start"}:
            payload = dict(payload)
            service = self.service()
            service.config.tools.enabled = tuple(payload.pop("enabled_tools"))
            service.config.model_tools.enabled = tuple(payload.pop("enabled_model_tools"))
            payload.pop("voice_turn", None)  # Compatibility with older runtime callers.
            if operation == "chat_start":
                self.handle("chat_close", None)
                self._chat = service.stream(**payload)
                return None
            result = service.chat(**payload)
            if len(result.response) > 12_000:
                raise ValueError("Response exceeds the speech limit")
            return {**asdict(result), "model_loaded": service._llm is not None and service._llm.is_loaded()}
        if operation == "chat_next":
            if self._chat is None:
                raise ValueError("No active conversation stream")
            return next(self._chat)
        if operation == "chat_close":
            if self._chat is not None:
                self._chat.close()
            self._chat = None
            return None
        if operation == "speech_close":
            try:
                self.handle("tts_close", None)
            finally:
                self.handle("chat_close", None)
            return None
        if operation == "memory_start":
            from arietta_voice.llm import build_chat_model
            from arietta_voice.memory import MemoryStore
            self.handle("memory_close", None)
            self._memory_store = MemoryStore(self.config.prompts.memory_path, self.config.memory.enabled,
                                             self.config.memory.consolidate_every_turns)
            model = None
            if self.config.memory.enabled:
                service = self.service()
                if service._llm is None:
                    service._llm = build_chat_model(self.config.models)
                model = service._llm
            self._memory = self._memory_store.prepare_update(model, payload["user"], payload["assistant"], payload["turn_count"])
            return None
        if operation == "memory_next":
            if self._memory is None:
                raise ValueError("No active memory job")
            try:
                next(self._memory)
                return {"kind": "progress"}
            except StopIteration as done:
                self._memory_proposal = done.value
                self._memory = None
                return {"kind": "prepared"}
        if operation == "memory_commit":
            if self._memory_store is None or self._memory is not None:
                raise ValueError("Memory job is not prepared")
            try:
                return self._memory_store.commit_update(self._memory_proposal)
            finally:
                self.handle("memory_close", None)
        if operation == "memory_close":
            try:
                if self._memory is not None:
                    self._memory.close()
            finally:
                self._memory = self._memory_store = self._memory_proposal = None
            return None
        if operation == "tts_start":
            if not isinstance(payload, str) or len(payload) > 12_000:
                raise ValueError("Speech text exceeds its limit")
            self.handle("tts_close", None)
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
                samples = np.asarray(samples, dtype=np.float32)
                # The backend may produce a whole sentence before slicing IPC
                # frames; cap retained backend output as well as the wire chunk.
                if (samples.ndim != 1 or not 8000 <= rate <= 96000
                        or len(samples) > rate * 30 or not np.isfinite(samples).all()):
                    raise ValueError("Invalid or oversized TTS output")
                self._tts_pending = (samples, rate)
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
