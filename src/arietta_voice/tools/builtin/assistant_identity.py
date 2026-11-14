from __future__ import annotations

from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult


class AssistantIdentityTool(BaseTool):
    name = "assistant_identity"
    description = "the user asks who or what the assistant is"
    triggers = (
        r"\bwho are you\b",
        r"\bwhat(?:'s| is) your name\b",
        r"\bwhat do you do\b",
        r"\bwhat(?:'s| is) your purpose\b",
    )
    # The persona already tells the model who it is.
    model_callable = False

    def run(self, arguments):
        identity = self.context.config.assistant.identity
        return ToolResult(content=identity, speak=identity)


def create_tool(context: ToolContext) -> AssistantIdentityTool:
    return AssistantIdentityTool(context)
