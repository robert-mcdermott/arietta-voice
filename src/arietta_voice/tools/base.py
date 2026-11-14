"""One tool interface for instant answers and for tools the model calls.

A tool has a name, a description of when to use it, a JSON-schema description of
its arguments, a time limit, and a ``run`` method that returns a ``ToolResult``.
Optional ``triggers`` (regular expressions) let a request bypass the model: when
one matches, its named groups become the arguments and the tool runs at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from arietta_voice.config import AppConfig
    from arietta_voice.knowledge import KnowledgeRetriever

EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(slots=True, frozen=True)
class ToolContext:
    config: AppConfig
    knowledge_retriever: KnowledgeRetriever | None = None


@dataclass(slots=True, frozen=True)
class ToolResult:
    """What a tool produced.

    ``content`` is information for the model to use in its answer. ``speak`` is a
    finished answer for the user; when a trigger runs a tool and it returns
    ``speak``, that is said directly without asking the model. ``data`` holds
    structured details for logs and the console, never for the model.
    """

    content: str = ""
    speak: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.content.strip() and not (self.speak or "").strip()


class ToolError(Exception):
    """A tool could not run: unknown, invalid arguments, failure or time limit."""

    def __init__(self, tool: str, kind: str, message: str) -> None:
        super().__init__(message)
        self.tool, self.kind = tool, kind


class BaseTool:
    """Convenient base class; a tool only needs these attributes and ``run``."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict[str, Any]] = EMPTY_PARAMETERS
    triggers: ClassVar[tuple[str, ...]] = ()
    model_callable: ClassVar[bool] = True
    timeout_s: ClassVar[float] = 10.0
    # Other names a model may use for an argument, such as {"note": "text"}.
    # Small models often guess a near miss; this accepts it instead of failing.
    aliases: ClassVar[dict[str, str]] = {}
    # When the model calls this tool and the result has ``speak``, say that and
    # do not ask the model again: faster, and it cannot misstate what happened.
    speaks_for_itself: ClassVar[bool] = False
    # Changes something outside the assistant (a light, a message, a purchase).
    # The assistant asks before running these; ``confirm`` says when: "ask"
    # (every time, unless you answer "always"), "first_use" or "never".
    consequential: ClassVar[bool] = False
    confirm: ClassVar[str] = "ask"

    def __init__(self, context: ToolContext | None = None) -> None:
        self.context = context

    def match(self, text: str) -> dict[str, Any] | None:
        """Arguments for an instant answer, or None when no trigger matches."""
        for pattern in self.triggers:
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                return {key: value for key, value in found.groupdict().items() if value}
        return None

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
