from __future__ import annotations

from pathlib import Path
import json
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
        if any(token in lowered for token in ("credential", "password", "reset", "okta", "sign in")):
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
    monkeypatch.setattr("arietta_voice.knowledge._create_text_embedder", lambda config: FakeEmbedder())
    config = KnowledgeConfig(backend="vector", directory=knowledge_dir,
                             index_directory=tmp_path / "index", embedding_model="fake-bge-small")
    build_knowledge_index(config)
    manifest_path = config.index_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(outdated)
    manifest_path.write_text(json.dumps(manifest))
    assert KnowledgeRetriever(config).search("password reset") is None
    assert build_knowledge_index(config)["updated"]
    assert KnowledgeRetriever(config).search("password reset") is not None
    assert not build_knowledge_index(config)["updated"]


def test_retrieval_and_index_exclude_external_links_and_hidden_directories(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr("arietta_voice.knowledge._create_text_embedder", lambda config: FakeEmbedder())
    config = KnowledgeConfig(directory=knowledge_dir, index_directory=tmp_path / "index",
                             embedding_model="fake-bge-small")
    result = KnowledgeRetriever(config).search("password reset")
    assert result is not None
    assert len(result.matches) == 1
    assert result.matches[0].source_path == Path("access.md")
    assert build_knowledge_index(config)["section_count"] == 1
    assert "PRIVATE" not in (config.index_directory / "sections.jsonl").read_text()


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
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "sections.jsonl").exists()
    assert (index_dir / "vectors.npy").exists()
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
