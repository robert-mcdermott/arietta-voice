"""Example tool: roll dice. Enable it by adding "example_dice" to [tools].enabled."""

import random

from arietta_voice.tools import BaseTool, ToolResult


class ExampleDiceTool(BaseTool):
    name = "example_dice"
    description = "the user asks you to roll a die or dice"
    # "roll a 20 sided die" answers instantly; other phrasings go to the model.
    triggers = (r"\broll (?:a|one) (?P<sides>\d+)[ -]sided (?:die|dice)\b",)
    parameters = {
        "type": "object",
        "properties": {
            "sides": {"type": "integer", "minimum": 2, "maximum": 100, "description": "sides"},
            "count": {"type": "integer", "minimum": 1, "maximum": 10, "description": "dice"},
        },
        "required": ["sides"],
    }

    def run(self, arguments):
        sides, count = arguments["sides"], arguments.get("count", 1)
        rolls = [random.randint(1, sides) for _ in range(count)]
        spoken = ", ".join(map(str, rolls))
        return ToolResult(
            content=f"Rolled {count} {sides}-sided dice: {spoken} (total {sum(rolls)}).",
            speak=f"You rolled {spoken}.",
            data={"rolls": rolls},
        )


def create_tool(context):
    return ExampleDiceTool(context)
