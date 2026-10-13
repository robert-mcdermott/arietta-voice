"""Shared conversation policy for voice and the web console, without HTTP state."""
from dataclasses import dataclass

from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.llm import build_chat_model
from arietta_voice.model_tooling import maybe_generate_with_model_tools
from arietta_voice.models import ConversationTurn
from arietta_voice.prompts import PromptManager, sanitize_for_speech


@dataclass(slots=True)
class ChatResult:
    response: str
    source: str
    used_tool: str | None
    model_tool: str | None


class ConversationService:
    def __init__(self, config, *, prompt_manager=None, knowledge=None, generate=None):
        self.config = config
        self.prompt_manager = prompt_manager or PromptManager(config)
        self.knowledge = knowledge or KnowledgeRetriever(config.knowledge)
        self._llm = None
        self._generate_override = generate
        self._tools = self._model_tools = None
        self._tools_key = self._model_tools_key = None

    def _llm_generate(self, messages, *, max_tokens=None, temperature=None):
        if self._generate_override is not None:
            return self._generate_override(messages, max_tokens=max_tokens, temperature=temperature)
        if self._llm is None:
            self._llm = build_chat_model(self.config.models)
        return self._llm.generate(messages, max_tokens=max_tokens or self.config.models.max_tokens,
                                  temperature=self.config.models.temperature if temperature is None else temperature)

    def chat(self, *, message, history=(), use_tools=True, max_tokens=None, temperature=None):
        message = message.strip()
        history = [ConversationTurn(role=t['role'], content=t['content'].strip())
                   for t in history if t['content'].strip()]
        knowledge = self.knowledge if self.config.knowledge.enabled else None
        response = None
        if use_tools:
            from arietta_voice.tools import ToolRegistry
            key = (tuple(self.config.tools.enabled), self.config.knowledge.enabled)
            if self._tools is None or key != self._tools_key:
                self._tools = ToolRegistry(self.config, knowledge_retriever=knowledge)
                self._tools_key = key
            response = self._tools.maybe_handle(message)
        used_tool = response.name if response else None
        if response and response.bypass_model:
            return ChatResult(sanitize_for_speech(response.content), 'tool', used_tool, None)
        messages = self.prompt_manager.build_messages(history, message, response)
        def generate(messages):
            return self._llm_generate(messages, max_tokens=max_tokens, temperature=temperature)
        if use_tools:
            from arietta_voice.model_tools import ModelToolRegistry
            key = (tuple(self.config.model_tools.enabled), self.config.knowledge.enabled)
            if self._model_tools is None or key != self._model_tools_key:
                self._model_tools = ModelToolRegistry(self.config, knowledge_retriever=knowledge)
                self._model_tools_key = key
            answer, model_tool = maybe_generate_with_model_tools(
                messages, model_tool_registry=self._model_tools, llm_generate=generate)
        else:
            answer, model_tool = generate(messages), None
        return ChatResult(sanitize_for_speech(answer), 'model', used_tool,
                          model_tool.name if model_tool else None)
