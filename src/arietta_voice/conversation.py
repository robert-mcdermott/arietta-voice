"""Shared conversation policy for voice and the web console, without HTTP state."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from arietta_voice.clauses import RESPONSE_CHARS, SpeechClauses
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.llm import ModelEvent, build_chat_model
from arietta_voice.model_tooling import (
    NATIVE_CALL_START,
    parse_model_tool_request,
    parse_native_tool_calls,
    tool_specs,
)
from arietta_voice.models import ConversationTurn, ToolResponse
from arietta_voice.prompts import PromptManager, sanitize_for_speech

# Tool calls the model may make in one turn before it must answer.
MAX_TOOL_STEPS = 3
NATIVE_TOOL_RULES = (
    "Use a tool only when you need one, and answer general knowledge yourself. "
    "Never say you set, changed or cancelled something unless a tool did it."
)
GAVE_UP = "Sorry, I couldn't finish that."
# Said once per turn when a tool is still working after this long, so a slow
# tool (the weather service, say) is not met with silence.
FILLER = "One moment."
FILLER_AFTER_S = 0.8


@dataclass(slots=True)
class ChatResult:
    response: str
    source: str
    used_tool: str | None
    model_tool: str | None


class ConversationService:
    def __init__(
        self, config, *, prompt_manager=None, knowledge=None, generate=None, native_tools=False
    ):
        """``generate`` replaces the model (tests, the console); ``native_tools``
        says whether that replacement takes ``tools`` and writes native calls."""
        self.config = config
        self._native_override = native_tools
        self.prompt_manager = prompt_manager or PromptManager(config)
        self.knowledge = knowledge or KnowledgeRetriever(config.knowledge)
        self._llm = None
        self._generate_override = generate
        self._tools = None
        self._tools_key = None

    def _llm_generate(self, messages, *, max_tokens=None, temperature=None):
        if self._generate_override is not None:
            return self._generate_override(messages, max_tokens=max_tokens, temperature=temperature)
        if self._llm is None:
            self._llm = build_chat_model(self.config.models)
        return self._llm.generate(
            messages,
            max_tokens=max_tokens or self.config.models.max_tokens,
            temperature=self.config.models.temperature if temperature is None else temperature,
        )

    def chat(self, **request):
        # HTTP/text callers collect the same events used by voice playback.
        stream = self.stream(**request)
        try:
            for event in stream:
                if event["kind"] == "done":
                    return ChatResult(**event["result"])
            raise RuntimeError("Conversation ended without a result")
        finally:
            stream.close()

    def _llm_stream(self, messages, *, max_tokens=None, temperature=None, tools=None):
        if self._generate_override is not None:
            options = {"tools": tools} if tools else {}
            text = self._generate_override(
                messages, max_tokens=max_tokens, temperature=temperature, **options
            )
            yield ModelEvent("text", text)
            yield ModelEvent("done")
            return
        if self._llm is None:
            self._llm = build_chat_model(self.config.models)
        options = {"tools": tools} if tools else {}
        yield from self._llm.stream(
            messages,
            max_tokens=max_tokens or self.config.models.max_tokens,
            temperature=self.config.models.temperature if temperature is None else temperature,
            reuse_prefix=True,
            **options,
        )

    def _native_tools(self) -> bool:
        """Whether the model declares tools natively (Gemma 4) instead of prompted JSON."""
        if not getattr(self.config.models, "native_tool_calls", True):
            return False
        if self._generate_override is not None:
            return self._native_override
        if self._llm is None:
            self._llm = build_chat_model(self.config.models)
        return getattr(self._llm, "native_tool_calls", False) is True

    def stream(self, *, message, history=(), use_tools=True, max_tokens=None, temperature=None):
        message = message.strip()
        history = [
            ConversationTurn(role=t["role"], content=t["content"].strip())
            for t in history
            if t["content"].strip()
        ]
        tools = self._registry() if use_tools else None
        filler = _Filler()
        hit = None
        if tools is not None:
            hit = yield from filler.wait(
                lambda: tools.fast_path(message), {"kind": "clause", "text": FILLER}
            )
        used_tool = hit.name if hit else None
        if hit and hit.result.speak:
            yield from self._speech_events(
                iter([ModelEvent("text", hit.result.speak), ModelEvent("done")]),
                source="tool",
                used_tool=used_tool,
                model_tool=None,
            )
            return
        context = None
        if hit:
            context = ToolResponse(hit.name, "", bypass_model=False, context=hit.result.content)
        messages = self.prompt_manager.build_messages(history, message, context)

        if tools is not None and tools.model_tools and self._native_tools():
            called: list[str] = []
            yield from self._speech_events(
                self._native_turn(tools, messages, called, max_tokens, temperature, filler),
                source="model",
                used_tool=used_tool,
                finalize=lambda: {"model_tool": ", ".join(called) or None},
            )
            return

        model_tool = None
        instruction = tools.instruction_text() if tools is not None else None
        if instruction:
            planning = self._llm_stream(
                [{"role": "system", "content": instruction}, *messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            tool_text, answer = _split_planning(planning)
            if answer is not None:
                # The model chose to answer directly: speak it as it arrives.
                yield from self._speech_events(
                    answer, source="model", used_tool=used_tool, model_tool=None
                )
                return
            request = parse_model_tool_request(tool_text)
            if request is None:
                raise ValueError("Invalid tool planning response")
            model_tool = request.name
            content, result = _run_for_model(tools, request.name, request.arguments)
            if result is not None and result.speak and _speaks_for_itself(tools, request.name):
                yield from self._speech_events(
                    iter([ModelEvent("text", result.speak), ModelEvent("done")]),
                    source="model",
                    used_tool=used_tool,
                    model_tool=model_tool,
                )
                return
            if result is not None and result.content.strip():
                content = f"Result from the {request.name} tool:\n{content}"
            messages = [*messages, {"role": "system", "content": content}]
        yield from self._speech_events(
            self._llm_stream(messages, max_tokens=max_tokens, temperature=temperature),
            source="model",
            used_tool=used_tool,
            model_tool=model_tool,
        )

    def _registry(self):
        from arietta_voice.tools import ToolRegistry, enabled_tool_names

        key = (
            tuple(enabled_tool_names(self.config)),
            self.config.knowledge.enabled,
            str(self.config.tools.directory),
        )
        if self._tools is None or key != self._tools_key:
            knowledge = self.knowledge if self.config.knowledge.enabled else None
            self._tools = ToolRegistry(self.config, knowledge_retriever=knowledge)
            self._tools_key = key
        return self._tools

    def _native_turn(self, tools, messages, called, max_tokens, temperature, filler=None):
        """Text events for one turn with native tool calls, running tools between steps.

        The model may answer, call a tool, or start answering and then call one.
        Tool calls are never spoken. After each call the result is added as a
        tool message and the model continues; after MAX_TOOL_STEPS calls it must
        answer, or the turn ends with an apology.
        """
        specs = tool_specs(tools.model_tools)
        messages = _with_rules(messages, NATIVE_TOOL_RULES)
        fallback = None  # A tool's own answer, if the model then says nothing.
        for step in range(MAX_TOOL_STEPS + 1):
            spoken: list[str] = []
            call_text = None
            hidden = _HiddenReasoning()
            events = self._llm_stream(
                messages, max_tokens=max_tokens, temperature=temperature, tools=specs
            )
            try:
                for event in events:
                    if call_text is not None:
                        if event.kind == "text":
                            call_text += event.text
                        continue
                    if event.kind != "text":
                        continue
                    text = hidden.visible(event.text)
                    head, marker, tail = text.partition(NATIVE_CALL_START)
                    if head:
                        spoken.append(head)
                        yield ModelEvent("text", head)
                    if marker:
                        call_text = marker + tail
            finally:
                _close(events)
            if call_text is None:
                if not "".join(spoken).strip() and fallback:
                    yield ModelEvent("text", fallback)
                yield ModelEvent("done")
                return
            calls = parse_native_tool_calls(call_text)
            if calls is None:
                raise ValueError("Invalid tool call")
            if step == MAX_TOOL_STEPS:
                break
            call = calls[0]  # One at a time; the model can ask again for more.
            called.append(call.name)
            filler = filler or _Filler()
            content, result = yield from filler.wait(
                lambda: _run_for_model(tools, call.name, call.arguments),
                ModelEvent("filler", FILLER),
            )
            if result is not None and result.speak and _speaks_for_itself(tools, call.name):
                yield ModelEvent("text", result.speak)
                yield ModelEvent("done")
                return
            fallback = result.speak if result is not None else None
            call_id = f"call_{step + 1}"
            messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": "".join(spoken),
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": call.name,
                    "content": content,
                },
            ]
        yield ModelEvent("text", GAVE_UP)
        yield ModelEvent("done")

    def _speech_events(self, events, finalize=None, **metadata):
        clauses = SpeechClauses()
        spoken = []
        response_parts = []
        complete = False
        try:
            for event in events:
                if complete:
                    raise ValueError("Event after completion")
                if event.kind == "filler":
                    # Spoken while a tool works; not part of the answer.
                    yield {"kind": "clause", "text": event.text}
                elif event.kind == "text":
                    for clause in clauses.feed(event.text):
                        spoken.append(clause)
                        yield {"kind": "clause", "text": clause}
                    if event.text:
                        response_parts.append(event.text)
                elif event.kind == "done":
                    complete = True
                else:
                    raise ValueError("Unexpected generation event")
            if not complete:
                raise RuntimeError("Incomplete generation stream")
            tail = clauses.finish()
            if tail:
                spoken.append(tail)
                yield {"kind": "clause", "text": tail}
            if not spoken:
                spoken.append("I don't have a good answer for that yet.")
                yield {"kind": "clause", "text": spoken[0]}
            response = sanitize_for_speech("".join(response_parts)) or spoken[0]
            if finalize is not None:
                metadata.update(finalize())
            yield {"kind": "done", "result": {"response": response, **metadata}}
        finally:
            if hasattr(events, "close"):
                events.close()


def _split_planning(events):
    """Tell a tool request from a spoken answer after the first visible character.

    A planning response is either JSON (optionally fenced) asking for a tool, or
    the answer itself. Returns ``(text, None)`` with the complete response for a
    tool request, which must never be spoken, or ``(None, events)`` replaying a
    direct answer from its first event so speech can start while it generates.
    """
    seen = []
    try:
        for event in events:
            seen.append(event)
            if event.kind == "done":
                return None, iter(seen)  # Finished without visible text.
            if event.kind != "text":
                raise ValueError("Unexpected generation event")
            head = "".join(item.text for item in seen).lstrip()
            if head.startswith(("{", "`")):
                return _read_tool_request(head, events), None
            if head:
                return None, _direct_answer(seen, events)
        raise RuntimeError("Incomplete generation stream")
    except BaseException:
        _close(events)
        raise


def _read_tool_request(head, events):
    parts, size, complete = [head], len(head), False
    try:
        if size > RESPONSE_CHARS:
            raise ValueError("Planning response exceeds its limit")
        for event in events:
            if complete:
                raise ValueError("Event after completion")
            if event.kind == "done":
                complete = True
            elif event.kind == "text":
                size += len(event.text)
                if size > RESPONSE_CHARS:
                    raise ValueError("Planning response exceeds its limit")
                parts.append(event.text)
            else:
                raise ValueError("Unexpected generation event")
    finally:
        _close(events)
    if not complete:
        raise RuntimeError("Incomplete generation stream")
    return "".join(parts)


def _direct_answer(seen, events):
    # Speech has started, so a tool request appearing later cannot be honoured
    # or spoken. Stop the turn rather than read JSON aloud.
    text = ""
    try:
        for source in (seen, events):
            for event in source:
                if event.kind == "text":
                    text += event.text
                    if "tool_call" in text:
                        raise ValueError("Invalid tool planning response")
                yield event
    finally:
        _close(events)


def _close(events):
    if hasattr(events, "close"):
        events.close()


def _run_for_model(tools, name, arguments):
    """Run a tool the model asked for: (text for the model, result or None on failure)."""
    from arietta_voice.tools import ToolError

    try:
        result = tools.run(name, arguments)
    except ToolError as error:
        return (
            f"The {name} tool could not be used ({error.kind}: {error}). Answer as well as "
            "you can without it, and say briefly that you could not check.",
            None,
        )
    if not result.content.strip():
        return (
            f"The {name} tool found nothing for that request. If the answer depends on it, "
            "say you are not sure instead of inventing details.",
            result,
        )
    return result.content, result


def _speaks_for_itself(tools, name) -> bool:
    return getattr(tools.tools.get(name), "speaks_for_itself", False) is True


class _Filler:
    """Runs slow work in the background and says FILLER once if it takes a while."""

    def __init__(self) -> None:
        self.said = False

    def wait(self, work, event):
        """Generator: yields ``event`` at most once per turn; returns ``work()``."""
        if self.said:
            return work()
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arietta-tool-wait")
        try:
            future = pool.submit(work)
            try:
                return future.result(timeout=FILLER_AFTER_S)
            except FutureTimeout:
                self.said = True
                yield event
                return future.result()
        finally:
            pool.shutdown(wait=False)


class _HiddenReasoning:
    """Drops ``<|channel>...<channel|>`` reasoning from streamed text, across chunks."""

    START, END = "<|channel>", "<channel|>"

    def __init__(self) -> None:
        self.inside = False

    def visible(self, text: str) -> str:
        shown = []
        while text:
            if self.inside:
                _, found, text = text.partition(self.END)
                self.inside = not found
            else:
                before, found, text = text.partition(self.START)
                shown.append(before)
                self.inside = bool(found)
        return "".join(shown)


def _with_rules(messages, rules):
    """Add tool rules to the first system message, which holds the tool declarations."""
    if messages and messages[0]["role"] == "system":
        first = {**messages[0], "content": f"{messages[0]['content'].rstrip()}\n\n{rules}"}
        return [first, *messages[1:]]
    return [{"role": "system", "content": rules}, *messages]
