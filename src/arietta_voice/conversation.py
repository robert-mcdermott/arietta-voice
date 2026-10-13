"""Shared conversation policy for voice and the web console, without HTTP state."""
from dataclasses import dataclass

from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.llm import build_chat_model, ModelEvent
from arietta_voice.model_tooling import parse_model_tool_request
from arietta_voice.clauses import SpeechClauses, RESPONSE_CHARS
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

    def chat(self, **request):
        # HTTP/text callers collect the same events used by voice playback.
        stream = self.stream(**request)
        try:
            for event in stream:
                if event['kind'] == 'done':
                    return ChatResult(**event['result'])
            raise RuntimeError('Conversation ended without a result')
        finally:
            stream.close()

    def _llm_stream(self, messages, *, max_tokens=None, temperature=None):
        if self._generate_override is not None:
            yield ModelEvent('text', self._llm_generate(messages, max_tokens=max_tokens, temperature=temperature))
            yield ModelEvent('done')
            return
        if self._llm is None:
            self._llm = build_chat_model(self.config.models)
        yield from self._llm.stream(messages, max_tokens=max_tokens or self.config.models.max_tokens,
                                   temperature=self.config.models.temperature if temperature is None else temperature)

    def stream(self, *, message, history=(), use_tools=True, max_tokens=None, temperature=None):
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
            yield from self._speech_events(iter([ModelEvent('text', response.content), ModelEvent('done')]),
                                           source='tool', used_tool=used_tool, model_tool=None)
            return
        messages = self.prompt_manager.build_messages(history, message, response)
        def generate(messages):
            return self._llm_generate(messages, max_tokens=max_tokens, temperature=temperature)
        if use_tools:
            from arietta_voice.model_tools import ModelToolRegistry
            key = (tuple(self.config.model_tools.enabled), self.config.knowledge.enabled)
            if self._model_tools is None or key != self._model_tools_key:
                self._model_tools = ModelToolRegistry(self.config, knowledge_retriever=knowledge)
                self._model_tools_key = key
            instruction = self._model_tools.instruction_text()
            if instruction:
                # A planning response may be JSON. Never expose it incrementally.
                first = generate([{'role': 'system', 'content': instruction}, *messages])
                if len(first) > RESPONSE_CHARS:
                    raise ValueError('Planning response exceeds its limit')
                request = parse_model_tool_request(first)
                if request is None:
                    if 'tool_call' in first or first.lstrip().startswith(('{', '```')):
                        raise ValueError('Invalid tool planning response')
                    yield from self._speech_events(iter([ModelEvent('text', first), ModelEvent('done')]),
                                                   source='model', used_tool=used_tool, model_tool=None)
                    return
                execution = self._model_tools.execute(request.name, request.arguments)
                if execution is None:
                    raise ValueError('Model requested an unavailable tool')
                messages = [*messages, {'role': 'system', 'content': execution.followup_context}]
                model_tool = request.name
            else:
                model_tool = None
        else:
            model_tool = None
        yield from self._speech_events(self._llm_stream(messages, max_tokens=max_tokens, temperature=temperature),
                                       source='model', used_tool=used_tool, model_tool=model_tool)

    def _speech_events(self, events, **metadata):
        clauses = SpeechClauses()
        spoken = []
        response_parts = []
        complete = False
        try:
            for event in events:
                if complete:
                    raise ValueError('Event after completion')
                if event.kind == 'text':
                    for clause in clauses.feed(event.text):
                        spoken.append(clause)
                        yield {'kind': 'clause', 'text': clause}
                    if event.text:
                        response_parts.append(event.text)
                elif event.kind == 'done':
                    complete = True
                else:
                    raise ValueError('Unexpected generation event')
            if not complete:
                raise RuntimeError('Incomplete generation stream')
            tail = clauses.finish()
            if tail:
                spoken.append(tail)
                yield {'kind': 'clause', 'text': tail}
            if not spoken:
                spoken.append("I don't have a good answer for that yet.")
                yield {'kind': 'clause', 'text': spoken[0]}
            response = sanitize_for_speech(''.join(response_parts)) or spoken[0]
            yield {'kind': 'done', 'result': {'response': response, **metadata}}
        finally:
            if hasattr(events, 'close'):
                events.close()
