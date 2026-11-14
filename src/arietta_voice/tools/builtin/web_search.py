from __future__ import annotations

import re

from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult
from arietta_voice.web_search import SearchClient, results_for_model

# "Look up who Ada Lovelace was", "search the web for the Eiffel Tower", "google X".
LOOKUP = re.compile(
    r"^(?:(?:hey|ok|okay),? )?(?:(?:please|can you|could you|would you|will you) )?"
    r"(?:search(?: the web| online| the internet| wikipedia)?(?: for)?|look up|google)\s+"
    r"(?P<query>.+?)(?:\s+(?:online|on the web|on the internet|on wikipedia|for me|please))*"
    r"\s*[?.!]*$",
    re.IGNORECASE,
)
# Your own notes are the knowledge tool's job.
NOT_WEB = re.compile(r"^(?:my|our|the) (?:notes|files|knowledge|documents)\b", re.IGNORECASE)
QUESTION_WORDS = re.compile(r"^(?:who|what|where|when|which|how)\s+(?:is|was|are|were\s+)?", re.I)


def lookup_request(text: str) -> dict | None:
    """The search words in a clear request to look something up, or None."""
    found = LOOKUP.match(" ".join(text.split()))
    if found is None:
        return None
    query = found["query"].strip(" ,.")
    if NOT_WEB.match(query):
        return None
    query = QUESTION_WORDS.sub("", query)
    query = re.sub(r"\s+(?:is|was|are|were)$", "", query).strip(" ,.")
    return {"query": query} if query else None


class WebSearchTool(BaseTool):
    """Searches the web (opt-in) and credits the source when the answer is spoken."""

    name = "web_search"
    description = (
        "the user asks about news, recent events or a specific current fact you are not sure "
        "of, or asks you to look something up online (not for the time, weather, timers or "
        "this home)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "what to search for, such as 2026 Nobel Prize in Physics",
                "maxLength": 200,
            }
        },
        "required": ["query"],
    }
    aliases = {"q": "query", "search": "query", "text": "query", "question": "query"}
    timeout_s = 12.0

    def __init__(self, context=None, client=None) -> None:
        super().__init__(context)
        self.client = client or SearchClient(context.config.web_search)

    def match(self, text):
        # A clear "look up ..." always searches; the results go to the model,
        # which answers and says where the answer came from.
        return lookup_request(text)

    def run(self, arguments):
        query = arguments["query"].strip()
        results = self.client.search(query)  # A failure becomes "could not check".
        return ToolResult(
            content=results_for_model(query, results),
            data={
                "query": query,
                "sources": [{"title": r.title, "url": r.url} for r in results],
            },
        )


def create_tool(context: ToolContext) -> WebSearchTool | None:
    if not context.config.web_search.enabled:
        return None  # Opt-in: searches leave the Mac.
    return WebSearchTool(context)
