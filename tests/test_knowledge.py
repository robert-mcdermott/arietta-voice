from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arietta_voice.config import KnowledgeConfig
from arietta_voice.knowledge import BgeTextEmbedder, KnowledgeRetriever, build_knowledge_index


def test_keyword_retriever_matches_team_and_food_queries(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "project-overview.md").write_text(
        """\
---
title: Project Overview
tags: project, overview, team, support
---

## Team Overview
The project supports local voice-assistant experiences and user-defined tools.
""",
        encoding="utf-8",
    )
    (knowledge_dir / "food-and-dining.md").write_text(
        """\
---
title: Food and Dining
tags: food, eat, lunch, cafe, cafeteria
aliases: where can I eat
---

## On-Site Options
There is a cafeteria and coffee bar nearby.
""",
        encoding="utf-8",
    )

    retriever = KnowledgeRetriever(
        KnowledgeConfig(
            enabled=True,
            backend="keyword",
            directory=knowledge_dir,
            index_directory=tmp_path / ".knowledge_index",
            max_matches=3,
            snippet_char_limit=240,
            min_score=6.0,
        )
    )

    team_result = retriever.search("project overview")
    food_result = retriever.search("Where can I get something to eat?")

    assert team_result is not None
    assert team_result.matches[0].title == "Project Overview"
    assert food_result is not None
    assert food_result.matches[0].title == "Food and Dining"


def test_keyword_retriever_ignores_readme_and_templates(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "README.md").write_text("# ignored", encoding="utf-8")
    (knowledge_dir / "_template.md").write_text("# ignored", encoding="utf-8")
    (knowledge_dir / "leadership.md").write_text(
        """\
---
title: IT Leadership Team
tags: it, leadership, managers
---

## Leadership Roster
List names and roles here.
""",
        encoding="utf-8",
    )

    retriever = KnowledgeRetriever(
        KnowledgeConfig(
            backend="keyword",
            directory=knowledge_dir,
            index_directory=tmp_path / ".knowledge_index",
        )
    )
    result = retriever.search("IT leadership team")

    assert result is not None
    assert len(result.matches) == 1
    assert result.matches[0].title == "IT Leadership Team"


class FakeEmbedder:
    def __init__(self) -> None:
        self.model_name = "fake-bge-small"
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return self._embed(text)

    def _embed(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.zeros(3, dtype=np.float32)
        if any(
            token in lowered for token in ("credential", "password", "reset", "okta", "sign in")
        ):
            vector[0] = 1.0
        if any(token in lowered for token in ("cafeteria", "coffee", "lunch", "food")):
            vector[1] = 1.0
        if not np.any(vector):
            vector[2] = 1.0
        vector /= np.linalg.norm(vector)
        return vector


def test_bge_uses_normalized_cls_for_documents_and_queries() -> None:
    import torch
    import torch.nn.functional as F

    embedder = BgeTextEmbedder.__new__(BgeTextEmbedder)
    embedder.batch_size = 1
    embedder._device = torch.device("cpu")
    embedder._torch = torch
    embedder._F = F
    embedder._tokenizer = lambda batch, **kwargs: {
        "attention_mask": torch.ones((len(batch), 2), dtype=torch.long),
    }
    embedder._model = lambda **kwargs: SimpleNamespace(
        last_hidden_state=torch.tensor([[[3.0, 4.0], [100.0, 0.0]]])
    )
    np.testing.assert_allclose(embedder.embed_documents(["one", "two"]), [[0.6, 0.8], [0.6, 0.8]])
    np.testing.assert_allclose(embedder.embed_query("question"), [0.6, 0.8])


@pytest.mark.parametrize("outdated", [{"version": 1}, {"embedding_pooling": "mean"}])
def test_old_embedding_pipeline_is_rejected_and_rebuilt(tmp_path, monkeypatch, outdated) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "access.md").write_text("# Credential recovery\nUse password reset.")
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda config: FakeEmbedder()
    )
    config = KnowledgeConfig(
        backend="vector",
        directory=knowledge_dir,
        index_directory=tmp_path / "index",
        embedding_model="fake-bge-small",
    )
    build_knowledge_index(config)
    # Older installations have three flat files. They remain readable until
    # a build needs to replace them.
    archive_path = config.index_directory / "current.zip"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(config.index_directory)
    archive_path.unlink()
    assert KnowledgeRetriever(config).search("password reset") is not None
    manifest_path = config.index_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(outdated)
    manifest_path.write_text(json.dumps(manifest))
    assert KnowledgeRetriever(config).search("password reset") is None
    assert build_knowledge_index(config)["updated"]
    assert KnowledgeRetriever(config).search("password reset") is not None
    assert not build_knowledge_index(config)["updated"]


def test_retrieval_and_index_exclude_external_links_and_hidden_directories(
    tmp_path, monkeypatch
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    outside = tmp_path / "private.md"
    outside.write_text("# Credential recovery\nPRIVATE OUTSIDE password reset.")
    (knowledge_dir / "link.md").symlink_to(outside)
    for name in (".private", "_drafts"):
        folder = knowledge_dir / name
        folder.mkdir()
        (folder / "article.md").write_text("# Credential recovery\nPRIVATE HIDDEN password reset.")
    (knowledge_dir / "access.md").write_text("# Credential recovery\nPublic password reset.")
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda config: FakeEmbedder()
    )
    config = KnowledgeConfig(
        directory=knowledge_dir,
        index_directory=tmp_path / "index",
        embedding_model="fake-bge-small",
    )
    result = KnowledgeRetriever(config).search("password reset")
    assert result is not None
    assert len(result.matches) == 1
    assert result.matches[0].source_path == Path("access.md")
    assert build_knowledge_index(config)["section_count"] == 1
    with zipfile.ZipFile(config.index_directory / "current.zip") as archive:
        assert b"PRIVATE" not in archive.read("sections.jsonl")


def test_build_knowledge_index_and_hybrid_search_use_semantic_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    index_dir = tmp_path / ".knowledge_index"
    (knowledge_dir / "access.md").write_text(
        """\
---
title: Credential Recovery
tags: account, recovery
---

## Recovery Steps
Use Okta self-service password reset to regain access.
""",
        encoding="utf-8",
    )
    (knowledge_dir / "food.md").write_text(
        """\
---
title: Food and Dining
tags: food, lunch
---

## Cafe
There is a cafeteria and coffee bar nearby.
""",
        encoding="utf-8",
    )

    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder",
        lambda config: fake_embedder,
    )
    config = KnowledgeConfig(
        backend="hybrid",
        directory=knowledge_dir,
        index_directory=index_dir,
        embedding_model="fake-bge-small",
        min_vector_score=0.2,
    )

    payload = build_knowledge_index(config)
    retriever = KnowledgeRetriever(config)
    result = retriever.search("I can't sign in to my account")

    assert payload["updated"] is True
    with zipfile.ZipFile(index_dir / "current.zip") as archive:
        assert set(archive.namelist()) == {"manifest.json", "sections.jsonl", "vectors.npy"}
    assert result is not None
    assert result.backend == "hybrid"
    assert result.matches[0].title == "Credential Recovery"
    assert fake_embedder.query_calls == 1


def test_hybrid_retriever_falls_back_to_keyword_when_vector_index_is_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    index_dir = tmp_path / ".knowledge_index"
    article = knowledge_dir / "rooms.md"
    article.write_text(
        """\
---
title: Conference Rooms
tags: rooms, meetings
---

## Locations
Conference rooms are on the second floor.
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder",
        lambda config: FakeEmbedder(),
    )
    config = KnowledgeConfig(
        backend="hybrid",
        directory=knowledge_dir,
        index_directory=index_dir,
        embedding_model="fake-bge-small",
        min_vector_score=0.2,
    )
    build_knowledge_index(config)

    article.write_text(
        """\
---
title: Meeting Spaces
tags: rooms, meetings
---

## Locations
Meeting spaces are on the second floor.
""",
        encoding="utf-8",
    )

    retriever = KnowledgeRetriever(config)
    result = retriever.search("meeting spaces")

    assert result is not None
    assert result.matches[0].title == "Meeting Spaces"


def test_hybrid_retriever_skips_semantic_embedding_for_small_talk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    index_dir = tmp_path / ".knowledge_index"
    (knowledge_dir / "access.md").write_text(
        """\
---
title: Credential Recovery
tags: account, recovery
---

## Recovery Steps
Use Okta self-service password reset to regain access.
""",
        encoding="utf-8",
    )

    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder",
        lambda config: fake_embedder,
    )
    config = KnowledgeConfig(
        backend="hybrid",
        directory=knowledge_dir,
        index_directory=index_dir,
        embedding_model="fake-bge-small",
        min_vector_score=0.2,
    )
    build_knowledge_index(config)

    retriever = KnowledgeRetriever(config)
    result = retriever.search("hello")

    assert result is None
    assert fake_embedder.query_calls == 0


class AngleEmbedder:
    """Notes about the gym point one way; anything else points 60 degrees off (cosine 0.5)."""

    model_name = "fake-angle"

    def embed_documents(self, texts):
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        if "gym" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.5, np.sqrt(0.75)], dtype=np.float32)


def rooms_config(tmp_path: Path, monkeypatch, **options) -> KnowledgeConfig:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rooms.md").write_text(
        "# Conference rooms\nThe conference rooms are on the second floor, by the lifts."
    )
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda config: AngleEmbedder()
    )
    return KnowledgeConfig(
        backend="hybrid",
        directory=knowledge_dir,
        index_directory=tmp_path / "index",
        embedding_model="fake-angle",
        **options,
    )


def test_hybrid_search_attaches_only_notes_close_in_meaning(tmp_path, monkeypatch) -> None:
    # "Which floor is the gym on?" shares "floor" with the rooms note but is about
    # something else; reading that note would only slow the answer.
    config = rooms_config(tmp_path, monkeypatch)
    build_knowledge_index(config)
    retriever = KnowledgeRetriever(config)
    assert retriever.search("Which floor is the gym on?") is None  # Cosine 0.5.
    config.min_similarity = 0.4
    assert KnowledgeRetriever(config).search("Which floor is the gym on?")


def test_a_question_opening_with_a_greeting_still_gets_keyword_notes(tmp_path, monkeypatch):
    # Greetings skip the search by meaning; the keyword bar must still apply.
    config = rooms_config(tmp_path, monkeypatch, min_score=1.0)
    build_knowledge_index(config)
    retriever = KnowledgeRetriever(config)
    retriever.preload()  # As at startup: the index is loaded.
    result = retriever.search("Hi, where are the conference rooms?")
    assert result is not None and result.matches[0].title == "Conference rooms"


def test_hybrid_search_without_an_index_uses_the_keyword_bar(tmp_path, monkeypatch) -> None:
    config = rooms_config(tmp_path, monkeypatch, min_score=1000.0)
    assert KnowledgeRetriever(config).search("conference rooms second floor") is None
    config.min_score = 1.0
    assert KnowledgeRetriever(config).search("conference rooms second floor")


def test_a_missing_or_outdated_index_is_built_at_startup(tmp_path, monkeypatch) -> None:
    config = rooms_config(tmp_path, monkeypatch, min_similarity=0.4)
    retriever = KnowledgeRetriever(config)
    assert retriever.build_index_if_needed()
    assert not retriever.build_index_if_needed()  # Up to date.
    (config.directory / "rooms.md").write_text("# Conference rooms\nNow on the third floor.")
    assert retriever.build_index_if_needed()
    keyword = KnowledgeConfig(backend="keyword", directory=config.directory)
    assert not KnowledgeRetriever(keyword).build_index_if_needed()


def test_an_index_that_cannot_be_built_leaves_keyword_search(tmp_path, monkeypatch) -> None:
    config = rooms_config(tmp_path, monkeypatch, min_score=1.0)

    def no_model(config):
        raise OSError("model not downloaded")

    monkeypatch.setattr("arietta_voice.knowledge._create_text_embedder", no_model)
    retriever = KnowledgeRetriever(config)
    assert not retriever.build_index_if_needed()
    assert retriever.search("conference rooms second floor")


def test_notes_reach_the_model_as_one_short_line_each(tmp_path) -> None:
    # The note's own text, whole; only the labels around it are short.
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "it-service-desk-and-support.md").write_text(
        "---\ntitle: Service desk\n---\n\n## Hours\nOpen weekdays from 8 to 6, by phone or chat.\n"
    )
    config = KnowledgeConfig(backend="keyword", directory=knowledge_dir, min_score=1.0)
    text = KnowledgeRetriever(config).search("service desk hours").prompt_context()
    lines = text.splitlines()
    assert lines[1] == "[1] Service desk — Hours: Open weekdays from 8 to 6, by phone or chat."
    assert "it-service-desk-and-support.md" not in text and "Source:" not in text


def test_a_section_that_fits_stays_whole() -> None:
    from arietta_voice.knowledge import _chunk_text

    assert _chunk_text("  Short   section\ntext.  ", 420) == ["Short section text."]
    assert _chunk_text("   ", 420) == []


def test_long_sections_are_cut_between_sentences_and_list_items() -> None:
    from arietta_voice.knowledge import _chunk_text, _clean_text

    prose = " ".join(f"Sentence {n} talks to Dr. Okafor about item {n}." for n in range(30))
    chunks = _chunk_text(prose, 120)
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert all(chunk.endswith(".") and not chunk.endswith("Dr.") for chunk in chunks)
    assert " ".join(chunks) == _clean_text(prose)  # Nothing lost or reordered.

    listing = "Contacts:\n" + "\n".join(
        f"- Person {n}, desk {n}\n  on floor {n}" for n in range(20)
    )
    chunks = _chunk_text(listing, 80)
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert any("- Person 7, desk 7 on floor 7" in chunk for chunk in chunks)  # Items stay whole.

    word_soup = " ".join(["word"] * 200)  # One sentence longer than a chunk.
    assert all(len(chunk) <= 50 for chunk in _chunk_text(word_soup, 50))


def test_a_fact_past_the_first_chunk_can_be_found(tmp_path) -> None:
    # Before, only the first 420 characters of a section were ever attached.
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    filler = " ".join("The hallway lights are on a timer." for _ in range(15))
    (knowledge_dir / "house.md").write_text(
        f"# House\n\n## Utilities\n{filler} The water meter is under the porch steps.\n"
    )
    config = KnowledgeConfig(backend="keyword", directory=knowledge_dir, min_score=1.0)
    result = KnowledgeRetriever(config).search("where is the water meter")
    assert result is not None
    assert "under the porch steps" in result.matches[0].snippet
    assert all(len(match.snippet) <= config.snippet_char_limit for match in result.matches)


def test_chunk_ids_are_stable_and_unique(tmp_path) -> None:
    from arietta_voice.knowledge import _load_sections

    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "same.md").write_text("# Same\n\n" + "Repeat this line. " * 60)
    first = [s.section_id for s in _load_sections(knowledge_dir, snippet_char_limit=100)]
    again = [s.section_id for s in _load_sections(knowledge_dir, snippet_char_limit=100)]
    assert first == again and len(set(first)) == len(first) > 1
