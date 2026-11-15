"""Retrieval recall on the made-up notes, with the keyword search CI can run.

Nine in ten questions the notes answer must find a note containing the answer
among the three attached, and general questions should rarely attach any.
"""

import json
import re
from pathlib import Path

from arietta_voice.config import KnowledgeConfig
from arietta_voice.knowledge import KnowledgeRetriever

EVALS = Path(__file__).resolve().parent
QUESTIONS = json.loads((EVALS / "notes_questions.json").read_text())


def _retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(KnowledgeConfig(backend="keyword", directory=EVALS / "fixture_notes"))


def test_questions_about_the_notes_find_the_answer_in_the_top_three():
    retriever = _retriever()
    about = [item for item in QUESTIONS if item.get("notes")]
    found = 0
    for item in about:
        result = retriever.search(item["q"])
        snippets = [match.snippet for match in result.matches] if result else []
        found += any(re.search(item["a"], text, re.IGNORECASE) for text in snippets)
    assert len(about) >= 14
    assert found / len(about) >= 0.9, f"recall@3 {found}/{len(about)}"


def test_general_questions_mostly_attach_no_notes():
    # Keyword search shares words loosely ("boiling point of water" and the hot
    # water notes); hybrid search filters those by meaning (min_similarity).
    retriever = _retriever()
    general = [item for item in QUESTIONS if not item.get("notes")]
    attached = sum(retriever.search(item["q"]) is not None for item in general)
    assert attached <= 1, f"{attached} of {len(general)} general questions got notes"
