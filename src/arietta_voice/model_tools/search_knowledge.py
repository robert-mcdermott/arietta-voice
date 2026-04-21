from __future__ import annotations

from arietta_voice.knowledge import KnowledgeRetriever, KnowledgeSearchResult
from arietta_voice.model_tools.base import ModelToolContext, ModelToolExecution


class SearchKnowledgeTool:
    name = "search_knowledge"
    description = (
        "the user's question depends on local organization or environment knowledge and the "
        "existing snippets in the prompt are not enough"
    )
    arguments_example = {"query": "<short search query>"}

    def __init__(self, knowledge_retriever: KnowledgeRetriever) -> None:
        self.knowledge_retriever = knowledge_retriever

    def invoke(self, arguments: dict[str, object]) -> ModelToolExecution | None:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return None

        result = self.knowledge_retriever.search(query)
        return ModelToolExecution(
            followup_context=_knowledge_tool_context(query, result),
            details={"query": query, "matches": 0 if result is None else len(result.matches)},
        )


def _knowledge_tool_context(query: str, result: KnowledgeSearchResult | None) -> str:
    if result is None:
        return (
            f"Additional local knowledge search requested by the assistant for query: {query}\n"
            "No matching local knowledge snippets were found. If the answer depends on unavailable "
            "local knowledge, say you are not sure instead of inventing details."
        )
    return (
        f"Additional local knowledge search requested by the assistant for query: {query}\n"
        f"{result.prompt_context()}"
    )


def create_tool(context: ModelToolContext) -> SearchKnowledgeTool | None:
    if context.knowledge_retriever is None:
        return None
    return SearchKnowledgeTool(context.knowledge_retriever)
