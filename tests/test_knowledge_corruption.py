"""A broken derived index must not take down keyword retrieval or source files."""

import json
import zipfile

import numpy as np
import pytest

from arietta_voice.config import default_config
from arietta_voice.knowledge import KnowledgeRetriever, build_knowledge_index


class SyntheticEmbedder:
    model_name = "synthetic-2d"

    def embed_documents(self, texts):
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))

    def embed_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def corrupt(directory, damage):
    manifest_path = directory / "manifest.json"
    sections_path = directory / "sections.jsonl"
    vectors_path = directory / "vectors.npy"
    if damage == "manifest-json":
        manifest_path.write_text("{unfinished")
    elif damage == "manifest-list":
        manifest_path.write_text("[]")
    elif damage == "manifest-encoding":
        manifest_path.write_bytes(b"\xff\xfe")
    elif damage.startswith("manifest-field:"):
        field = damage.split(":")[1]
        manifest = json.loads(manifest_path.read_text())
        manifest[field] = {"invalid": "synthetic"}
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "sections-json":
        sections_path.write_text("{unfinished")
    elif damage == "sections-list":
        sections_path.write_text("[]")
    elif damage == "sections-missing-field":
        sections_path.write_text('{"title":"Synthetic"}')
    elif damage in {"sections-external-path", "sections-wrong-type", "sections-duplicate"}:
        record = json.loads(sections_path.read_text())
        if damage == "sections-external-path":
            record["source_path"] = "../private.md"
        elif damage == "sections-wrong-type":
            record["section_title"] = []
        text = json.dumps(record)
        sections_path.write_text(text + ("\n" + text if damage == "sections-duplicate" else ""))
    elif damage == "vectors-truncated":
        vectors_path.write_bytes(b"\x93NUMPY")
    elif damage == "vectors-truncated-zip":
        vectors_path.write_bytes(b"PK\x03\x04unfinished")
    elif damage == "vectors-disguised-archive":
        with vectors_path.open("wb") as handle:
            np.savez(handle, vectors=np.array([[1.0, 0.0]]))
    elif damage == "vectors-nan":
        np.save(vectors_path, np.array([[float("nan"), 0.0]]))
    elif damage == "vectors-infinity":
        np.save(vectors_path, np.array([[float("inf"), 0.0]]))
    elif damage == "vectors-rank":
        np.save(vectors_path, np.array([1.0, 0.0]))
    elif damage == "vectors-dimension":
        np.save(vectors_path, np.array([[1.0, 0.0, 0.0]]))
    elif damage == "vectors-rows":
        np.save(vectors_path, np.zeros((2, 2), dtype=np.float32))
    elif damage == "vectors-strings":
        np.save(vectors_path, np.array([["one", "zero"]]))
    elif damage == "vectors-pickle":
        np.save(vectors_path, np.array([[{}, {}]], dtype=object))
    elif damage == "missing-vectors":
        vectors_path.unlink()
    else:
        raise AssertionError(damage)


@pytest.mark.parametrize("backend", ["vector", "hybrid"])
@pytest.mark.parametrize(
    "damage",
    [
        "truncated-archive",
        "manifest-json",
        "manifest-list",
        "manifest-encoding",
        "manifest-field:source_signature",
        "manifest-field:snippet_char_limit",
        "manifest-field:section_count",
        "manifest-field:vector_dim",
        "sections-json",
        "sections-list",
        "sections-missing-field",
        "sections-external-path",
        "sections-wrong-type",
        "sections-duplicate",
        "vectors-truncated",
        "vectors-truncated-zip",
        "vectors-disguised-archive",
        "vectors-nan",
        "vectors-infinity",
        "vectors-rank",
        "vectors-dimension",
        "vectors-rows",
        "vectors-strings",
        "vectors-pickle",
        "missing-vectors",
    ],
)
def test_corrupt_cache_falls_back_and_rebuilds(tmp_path, monkeypatch, backend, damage):
    config = default_config(tmp_path).knowledge
    config.backend = backend
    config.embedding_model = SyntheticEmbedder.model_name
    config.directory.mkdir()
    source = config.directory / "telescope.md"
    source.write_text("# Synthetic telescope\nThe synthetic telescope opens at noon.")
    original = source.read_bytes()
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda config: SyntheticEmbedder()
    )
    assert build_knowledge_index(config)["updated"]
    archive_path = config.index_directory / "current.zip"
    if damage == "truncated-archive":
        archive_path.write_bytes(b"PK")
    else:
        files = tmp_path / "broken"
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(files)
        corrupt(files, damage)
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in files.iterdir():
                archive.write(path, path.name)
    # A corrupt cache must be rejected before attempting a model load.
    with monkeypatch.context() as guard:

        def unexpected_load(config):
            pytest.fail("Corrupt cache attempted to load an embedding model")

        guard.setattr("arietta_voice.knowledge._create_text_embedder", unexpected_load)
        retriever = KnowledgeRetriever(config)
        retriever.preload()
        result = retriever.search("synthetic telescope")
        assert (result is not None) == (backend == "hybrid")
        if result:
            assert result.matches[0].source_path.name == "telescope.md"
    assert build_knowledge_index(config)["updated"]
    assert KnowledgeRetriever(config).search("synthetic telescope") is not None
    assert not build_knowledge_index(config)["updated"]
    assert source.read_bytes() == original


def test_empty_index_is_valid_without_loading_models(tmp_path):
    config = default_config(tmp_path).knowledge
    config.backend = "vector"
    config.directory.mkdir()
    assert build_knowledge_index(config)["updated"]
    assert not build_knowledge_index(config)["updated"]
    retriever = KnowledgeRetriever(config)
    retriever.preload()
    assert retriever.search("synthetic telescope") is None


def _stop_before_publish(config, ready, resume):
    from unittest.mock import patch

    from arietta_voice import knowledge

    validate = knowledge._read_vector_index

    def pause_staging(source, config, signature):
        result = validate(source, config, signature)
        if source.name == ".building.zip":
            ready.set()
            resume.wait(45)
        return result

    with (
        patch.object(knowledge, "_create_text_embedder", return_value=SyntheticEmbedder()),
        patch.object(knowledge, "_read_vector_index", side_effect=pause_staging),
    ):
        build_knowledge_index(config)


def test_killed_builder_preserves_generation_and_releases_writer_lock(tmp_path, monkeypatch):
    import multiprocessing

    config = default_config(tmp_path).knowledge
    config.backend = "vector"
    config.embedding_model = SyntheticEmbedder.model_name
    config.directory.mkdir()
    note = config.directory / "key.md"
    note.write_text("# Spare key\nThe spare key is in the blue drawer.")
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda _: SyntheticEmbedder()
    )
    assert build_knowledge_index(config)["updated"]
    retriever = KnowledgeRetriever(config)
    assert "blue drawer" in retriever.search("spare key").matches[0].snippet
    archive = config.index_directory / "current.zip"
    previous = archive.read_bytes()
    note.write_text("# Spare key\nThe spare key is in the hall closet.")
    ctx = multiprocessing.get_context("spawn")
    ready, resume = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_stop_before_publish, args=(config, ready, resume))
    child.start()
    try:
        assert ready.wait(20)
        assert retriever.search("spare key") is None  # Never serve the out-of-date vector.
    finally:
        child.terminate()
        child.join(10)
    assert not child.is_alive()
    assert archive.read_bytes() == previous
    assert build_knowledge_index(config)["updated"]
    assert "hall closet" in retriever.search("spare key").matches[0].snippet
    assert not (config.index_directory / ".building.zip").exists()


def test_rejected_vectors_leave_published_index_intact(tmp_path, monkeypatch):
    config = default_config(tmp_path).knowledge
    config.backend = "vector"
    config.directory.mkdir()
    note = config.directory / "key.md"
    note.write_text("# Key\nThe key is in the drawer.")
    config.embedding_model = SyntheticEmbedder.model_name
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda _: SyntheticEmbedder()
    )
    build_knowledge_index(config)
    archive = config.index_directory / "current.zip"
    previous = archive.read_bytes()

    class InvalidEmbedder(SyntheticEmbedder):
        def embed_documents(self, texts):
            return np.full((len(texts), 2), np.nan)

    note.write_text("# Key\nThe key is in the closet.")
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda _: InvalidEmbedder()
    )
    with pytest.raises(ValueError, match="Non-finite"):
        build_knowledge_index(config)
    assert archive.read_bytes() == previous
    assert not (config.index_directory / ".building.zip").exists()
    assert KnowledgeRetriever(config).search("where is the key") is None


def test_overlapping_builds_embed_once_and_publish_one_generation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    config = default_config(tmp_path).knowledge
    config.embedding_model = SyntheticEmbedder.model_name
    config.directory.mkdir()
    (config.directory / "key.md").write_text("# Key\nThe key is in the drawer.")
    embedding, resume, second_started = Event(), Event(), Event()

    class WaitingEmbedder(SyntheticEmbedder):
        def embed_documents(self, texts):
            assert not embedding.is_set(), "the second builder embedded the same notes"
            embedding.set()
            assert resume.wait(10)
            return super().embed_documents(texts)

    def second_build():
        second_started.set()
        return build_knowledge_index(config)

    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda _: WaitingEmbedder()
    )
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(build_knowledge_index, config)
        try:
            assert embedding.wait(5)
            second = pool.submit(second_build)
            assert second_started.wait(5)
        finally:
            resume.set()
        assert first.result()["updated"]
        assert not second.result()["updated"]
