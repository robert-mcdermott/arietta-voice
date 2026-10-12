"""A broken derived index must not take down keyword retrieval or source files."""
import json

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
@pytest.mark.parametrize("damage", [
    "manifest-json", "manifest-list", "manifest-encoding",
    "manifest-field:source_signature", "manifest-field:snippet_char_limit",
    "manifest-field:section_count", "manifest-field:vector_dim",
    "sections-json", "sections-list", "sections-missing-field",
    "sections-external-path", "sections-wrong-type", "sections-duplicate",
    "vectors-truncated", "vectors-truncated-zip", "vectors-disguised-archive",
    "vectors-nan", "vectors-infinity", "vectors-rank",
    "vectors-dimension", "vectors-rows", "vectors-strings", "vectors-pickle", "missing-vectors",
])
def test_corrupt_cache_falls_back_and_rebuilds(tmp_path, monkeypatch, backend, damage):
    config = default_config(tmp_path).knowledge
    config.backend = backend
    config.embedding_model = SyntheticEmbedder.model_name
    config.directory.mkdir()
    source = config.directory / "telescope.md"
    source.write_text("# Synthetic telescope\nThe synthetic telescope opens at noon.")
    original = source.read_bytes()
    monkeypatch.setattr("arietta_voice.knowledge._create_text_embedder", lambda config: SyntheticEmbedder())
    assert build_knowledge_index(config)["updated"]
    corrupt(config.index_directory, damage)
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
