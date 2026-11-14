from __future__ import annotations

from datetime import date, datetime

from arietta_voice.calendar_math import answer
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult


def local_today() -> date:
    return datetime.now().astimezone().date()


class DateMathTool(BaseTool):
    """Days until, days since, days between, and what day of the week a date is."""

    name = "date_math"
    description = (
        "the user asks how many days until or since a date, what day of the week a date "
        "is, or what the date will be after some days or weeks"
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "the question, such as how many days until Christmas",
                "maxLength": 200,
            }
        },
        "required": ["question"],
    }
    aliases = {"query": "question", "text": "question", "date": "question"}
    speaks_for_itself = True
    timeout_s = 2.0

    def __init__(self, context=None, clock=local_today) -> None:
        super().__init__(context)
        self.clock = clock

    def match(self, text):
        return {"question": text} if answer(text, self.clock()) else None

    def run(self, arguments):
        reply = answer(arguments["question"], self.clock())
        if reply is None:
            return ToolResult(
                content="That date question was not understood.",
                speak="Sorry, I couldn't work out that date.",
            )
        return ToolResult(content=reply, speak=reply)


def create_tool(context: ToolContext) -> DateMathTool:
    return DateMathTool(context)
