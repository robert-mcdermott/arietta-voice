from __future__ import annotations

from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult


class SearchKnowledgeTool(BaseTool):
    name = "search_knowledge"
    description = (
        "the user asks about this organization, home or its people (not general knowledge) "
        "and the notes already in the prompt do not answer it"
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "short search query"}},
        "required": ["query"],
    }

    def match(self, text):
        # Every request is looked up, so relevant notes reach the model without a
        # tool call. Nothing found adds nothing.
        return {"query": text} if text.strip() else None

    def run(self, arguments):
        query = arguments["query"].strip()
        result = self.context.knowledge_retriever.search(query) if query else None
        if result is None:
            return ToolResult(data={"query": query, "matches": 0})
        return ToolResult(
            content=result.prompt_context(),
            knowledge=result,
            data={"query": query, "matches": len(result.matches)},
        )


def create_tool(context: ToolContext) -> SearchKnowledgeTool | None:
    if context.knowledge_retriever is None:
        return None  # Knowledge is turned off.
    return SearchKnowledgeTool(context)
