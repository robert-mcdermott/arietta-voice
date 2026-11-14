from __future__ import annotations

from arietta_voice.spoken_math import (
    MathError,
    evaluate,
    parse_conversion,
    parse_expression,
    say_number,
)
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult


class CalculateTool(BaseTool):
    """Arithmetic and unit conversion, worked out exactly instead of by the model."""

    name = "calculate"
    description = "the user asks for arithmetic, a percentage or a unit conversion"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "what to work out, such as 17 * 23, 15% of 80 or 5 miles to km",
                "maxLength": 200,
            }
        },
        "required": ["expression"],
    }
    aliases = {"query": "expression", "math": "expression", "question": "expression"}
    speaks_for_itself = True
    timeout_s = 2.0

    def match(self, text):
        if parse_conversion(text) or parse_expression(text):
            return {"expression": text}
        return None

    def run(self, arguments):
        text = arguments["expression"].strip()
        conversion = parse_conversion(text)
        if conversion is not None:
            answer = conversion.spoken()
            return ToolResult(content=answer, speak=answer, data={"result": conversion.result})
        parsed = parse_expression(text)
        try:
            value = evaluate(parsed.python if parsed else text)
        except MathError as error:
            return ToolResult(content=f"Could not work that out: {error}.", speak=f"{error}.")
        result = say_number(value)
        answer = f"{parsed.spoken} is {result}." if parsed else f"That's {result}."
        answer = answer[0].upper() + answer[1:]
        return ToolResult(content=answer, speak=answer, data={"result": value})


def create_tool(context: ToolContext) -> CalculateTool:
    return CalculateTool(context)
