"""Rank the shipped keyword retriever against labeled synthetic note sections."""

import json
import re
from pathlib import Path

from arietta_voice.config import KnowledgeConfig
from arietta_voice.knowledge import KnowledgeRetriever, _load_sections

EVALS = Path(__file__).resolve().parent
NOTES = EVALS / "fixture_notes"
QUESTIONS = json.loads((EVALS / "notes_questions.json").read_text())


def section_name(match):
    return f"{match.source_path.as_posix()}#{match.section_title}"


def test_question_set_has_valid_evidence():
    assert len(QUESTIONS) == len({item["q"] for item in QUESTIONS}) == 50
    sections = _load_sections(NOTES, snippet_char_limit=KnowledgeConfig().snippet_char_limit)
    for item in QUESTIONS:
        assert bool(item.get("sections")) == bool(item.get("notes")), item["q"]
        for name in item.get("sections", []):
            assert any(
                section_name(section) == name and re.search(item["a"], section.snippet, re.I)
                for section in sections
            ), (item["q"], name)


def test_questions_about_the_notes_find_the_right_section_and_answer():
    retriever = KnowledgeRetriever(KnowledgeConfig(backend="keyword", directory=NOTES))
    about = [item for item in QUESTIONS if item.get("sections")]
    misses = []
    passage_misses = []
    first = 0
    for item in about:
        result = retriever.search(item["q"])
        matches = result.matches[:3] if result else ()
        relevant = [match for match in matches if section_name(match) in item["sections"]]
        if matches and section_name(matches[0]) in item["sections"]:
            first += 1
        if not relevant:
            misses.append(item["q"])
        if not any(re.search(item["a"], match.snippet, re.I) for match in relevant):
            passage_misses.append(item["q"])
    total = len(about)
    print(f"First result: {first}/{total}. Section recall@3: {total - len(misses)}/{total}.")
    print(f"Answer-bearing passage@3: {total - len(passage_misses)}/{total}.")
    print("Section misses:", misses, "Passage misses:", passage_misses)
    assert total == 42
    assert (total - len(misses)) / total >= 0.9, misses
    assert (total - len(passage_misses)) / total >= 0.9, passage_misses


def test_questions_without_an_answer_rarely_attach_notes():
    retriever = KnowledgeRetriever(KnowledgeConfig(backend="keyword", directory=NOTES))
    controls = [item for item in QUESTIONS if not item.get("sections")]
    attached = [item["q"] for item in controls if retriever.search(item["q"])]
    print(f"Unrelated attachments: {len(attached)}/{len(controls)}.", attached)
    assert len(controls) == 8
    assert len(attached) <= 1, attached
