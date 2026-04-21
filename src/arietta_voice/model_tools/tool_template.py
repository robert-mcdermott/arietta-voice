from __future__ import annotations

from arietta_voice.model_tools.base import ModelToolContext, ModelToolExecution


# Copy this file to a new module in this directory, then add the module name to
# [model_tools].enabled in config/arietta_voice.toml.
# For normal user-added tools, you do not need to edit config.py. Only add the
# module to DEFAULT_ENABLED_MODEL_TOOLS there if you want it to be a repo-level
# built-in default.
class ExampleModelTool:
    name = "example_model_tool"
    description = "you need a short follow-up lookup that the model cannot answer reliably on its own"
    arguments_example = {"topic": "<short topic>"}

    def invoke(self, arguments: dict[str, object]) -> ModelToolExecution | None:
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return None

        return ModelToolExecution(
            followup_context=(
                "Example model tool requested by the assistant.\n"
                f"Tool result for topic '{topic}': replace this with your real tool output.\n"
                "Use this tool result when composing the final answer."
            )
        )


def create_tool(context: ModelToolContext) -> ExampleModelTool:
    del context
    return ExampleModelTool()
