from __future__ import annotations

from arietta_voice.models import ToolResponse
from arietta_voice.tools.base import ToolContext


class KnowledgeRetrievalTool:
    name = "retrieve_knowledge"

    def __init__(self, context: ToolContext) -> None:
        self.knowledge_retriever = context.knowledge_retriever

    def maybe_handle(self, user_text: str) -> ToolResponse | None:
        if self.knowledge_retriever is None:
            return None

        result = self.knowledge_retriever.search(user_text)
        if result is None:
            return None

        return ToolResponse(
            name=self.name,
            content="",
            bypass_model=False,
            context=result.prompt_context(),
        )


def create_tool(context: ToolContext) -> KnowledgeRetrievalTool:
    return KnowledgeRetrievalTool(context)
