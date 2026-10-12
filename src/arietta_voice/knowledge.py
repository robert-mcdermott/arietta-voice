from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol

import numpy as np

from arietta_voice.config import KnowledgeConfig


TOKEN_RE = re.compile(r"[a-z0-9]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
WHITESPACE_RE = re.compile(r"\s+")
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
IGNORED_FILENAMES = {"README.md"}
INDEX_VERSION = 2
EMBEDDING_POOLING = "cls-normalized"
INDEX_MANIFEST_FILENAME = "manifest.json"
INDEX_SECTIONS_FILENAME = "sections.jsonl"
INDEX_VECTORS_FILENAME = "vectors.npy"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SMALL_TALK_PATTERNS = (
    re.compile(r"^\s*(?:hello|hi|hey)\b", re.IGNORECASE),
    re.compile(r"^\s*good\s+(?:morning|afternoon|evening)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:thanks|thank you)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:bye|goodbye)\b", re.IGNORECASE),
    re.compile(r"^\s*how are you\b", re.IGNORECASE),
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "can",
    "do",
    "for",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "or",
    "something",
    "tell",
    "the",
    "to",
    "we",
    "what",
    "where",
    "who",
    "you",
    "your",
}


@dataclass(slots=True, frozen=True)
class KnowledgeMatch:
    source_path: Path
    title: str
    section_title: str | None
    snippet: str
    score: float


@dataclass(slots=True, frozen=True)
class KnowledgeSearchResult:
    query: str
    backend: str
    matches: tuple[KnowledgeMatch, ...]

    def prompt_context(self) -> str:
        lines = [
            "Use the following local knowledge snippets only if they are relevant to the user's question.",
            "If the snippets do not answer the question, say you are not sure instead of inventing details.",
        ]
        for index, match in enumerate(self.matches, start=1):
            lines.append(f"[Snippet {index}]")
            lines.append(f"Source: {match.source_path.as_posix()}")
            lines.append(f"Title: {match.title}")
            if match.section_title:
                lines.append(f"Section: {match.section_title}")
            lines.append(f"Content: {match.snippet}")
        return "\n".join(lines)


class KnowledgeBackend(Protocol):
    def search(self, query: str) -> KnowledgeSearchResult | None: ...

    def preload(self) -> None: ...

    def is_loaded(self) -> bool: ...


class TextEmbedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass(slots=True)
class _KnowledgeSection:
    section_id: str
    source_path: Path
    title: str
    section_title: str | None
    summary: str | None
    snippet: str
    content_text: str
    title_tokens: Counter[str]
    tag_tokens: Counter[str]
    heading_tokens: Counter[str]
    path_tokens: Counter[str]
    body_tokens: Counter[str]
    searchable_text: str


@dataclass(slots=True, frozen=True)
class _ScoredKnowledgeSection:
    section_id: str
    source_path: Path
    title: str
    section_title: str | None
    snippet: str
    score: float


@dataclass(slots=True, frozen=True)
class _VectorSectionRecord:
    section_id: str
    source_path: Path
    title: str
    section_title: str | None
    snippet: str


class KeywordKnowledgeBackend:
    def __init__(self, config: KnowledgeConfig) -> None:
        self.config = config
        self._sections: list[_KnowledgeSection] = []
        self._signature: tuple[tuple[str, int, int], ...] | None = None

    def preload(self) -> None:
        self._refresh_index()

    def is_loaded(self) -> bool:
        return self._signature is not None

    def search(self, query: str) -> KnowledgeSearchResult | None:
        if not self.config.enabled:
            return None
        self._refresh_index()
        matches = self.candidate_matches(
            query,
            max_matches=self.config.max_matches,
            min_score=self.config.min_score,
        )
        return _knowledge_result(query, backend="keyword", matches=matches)

    def candidate_matches(
        self,
        query: str,
        *,
        max_matches: int,
        min_score: float,
    ) -> list[_ScoredKnowledgeSection]:
        self._refresh_index()
        if not self._sections:
            return []
        return _keyword_candidates(
            self._sections,
            query,
            max_matches=max_matches,
            min_score=min_score,
        )

    def _refresh_index(self) -> None:
        signature = _directory_signature(self.config.directory)
        if signature == self._signature:
            return
        self._sections = _load_sections(
            self.config.directory,
            snippet_char_limit=self.config.snippet_char_limit,
        )
        self._signature = signature


class VectorKnowledgeBackend:
    def __init__(self, config: KnowledgeConfig) -> None:
        self.config = config
        self._sections: list[_VectorSectionRecord] = []
        self._vectors: np.ndarray | None = None
        self._index_signature: tuple[tuple[str, int, int], ...] | None = None
        self._embedder: TextEmbedder | None = None
        self._manifest_loaded = False

    def preload(self) -> None:
        self._ensure_index_loaded(preload_embedder=True)

    def is_loaded(self) -> bool:
        return self._vectors is not None and self._manifest_loaded

    def search(self, query: str) -> KnowledgeSearchResult | None:
        if not self.config.enabled:
            return None
        matches = self.candidate_matches(
            query,
            max_matches=self.config.max_matches,
        )
        return _knowledge_result(query, backend="vector", matches=matches)

    def candidate_matches(self, query: str, *, max_matches: int) -> list[_ScoredKnowledgeSection]:
        if not self.config.enabled or not _should_attempt_semantic_search(query):
            return []
        if not self._ensure_index_loaded(preload_embedder=False):
            return []
        if self._vectors is None or not self._sections:
            return []

        query_vector = self._get_embedder().embed_query(query)
        scores = np.asarray(self._vectors @ query_vector, dtype=np.float32)
        if scores.size == 0:
            return []

        candidate_limit = min(max(max_matches * 6, max_matches), scores.shape[0])
        if candidate_limit <= 0:
            return []

        partition_index = max(0, scores.shape[0] - candidate_limit)
        top_indices = np.argpartition(scores, partition_index)[partition_index:]
        ordered = sorted(
            (int(index) for index in top_indices),
            key=lambda index: (
                -float(scores[index]),
                self._sections[index].source_path.as_posix(),
                self._sections[index].section_title or "",
            ),
        )

        matches: list[_ScoredKnowledgeSection] = []
        for index in ordered:
            score = float(scores[index])
            if score < self.config.min_vector_score:
                continue
            section = self._sections[index]
            matches.append(
                _ScoredKnowledgeSection(
                    section_id=section.section_id,
                    source_path=section.source_path,
                    title=section.title,
                    section_title=section.section_title,
                    snippet=section.snippet,
                    score=round(score, 3),
                )
            )
            if len(matches) >= candidate_limit:
                break
        return matches

    def _ensure_index_loaded(self, *, preload_embedder: bool) -> bool:
        source_signature = _directory_signature(self.config.directory)
        if (
            self._manifest_loaded
            and self._vectors is not None
            and self._index_signature == source_signature
        ):
            if preload_embedder and self._vectors.size:
                self._get_embedder()
            return True

        self._manifest_loaded = False
        self._sections = []
        self._vectors = None
        self._index_signature = None

        index_dir = self.config.index_directory
        manifest_path = index_dir / INDEX_MANIFEST_FILENAME
        sections_path = index_dir / INDEX_SECTIONS_FILENAME
        vectors_path = index_dir / INDEX_VECTORS_FILENAME
        if not (manifest_path.exists() and sections_path.exists() and vectors_path.exists()):
            return False

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_signature = _deserialize_signature(manifest.get("source_signature"))
        if manifest.get("version") != INDEX_VERSION:
            return False
        if manifest.get("embedding_model") != self.config.embedding_model:
            return False
        if manifest.get("embedding_pooling") != EMBEDDING_POOLING:
            return False
        if int(manifest.get("snippet_char_limit", -1)) != self.config.snippet_char_limit:
            return False
        if manifest_signature != source_signature:
            return False

        vectors = np.load(vectors_path)
        sections = _load_vector_section_records(sections_path)
        if vectors.ndim != 2:
            return False
        if vectors.shape[0] != len(sections):
            return False

        self._sections = sections
        self._vectors = vectors.astype(np.float32, copy=False)
        self._index_signature = manifest_signature
        self._manifest_loaded = True
        if preload_embedder and self._vectors.size:
            self._get_embedder()
        return True

    def _get_embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = _create_text_embedder(self.config)
        return self._embedder


class HybridKnowledgeBackend:
    def __init__(self, config: KnowledgeConfig) -> None:
        self.config = config
        self.keyword_backend = KeywordKnowledgeBackend(config)
        self.vector_backend = VectorKnowledgeBackend(config)

    def preload(self) -> None:
        self.keyword_backend.preload()
        self.vector_backend.preload()

    def is_loaded(self) -> bool:
        return self.keyword_backend.is_loaded() or self.vector_backend.is_loaded()

    def search(self, query: str) -> KnowledgeSearchResult | None:
        if not self.config.enabled:
            return None

        candidate_limit = max(self.config.max_matches * 6, self.config.max_matches)
        keyword_matches = self.keyword_backend.candidate_matches(
            query,
            max_matches=candidate_limit,
            min_score=max(1.0, self.config.min_score * 0.4),
        )
        vector_matches = self.vector_backend.candidate_matches(
            query,
            max_matches=candidate_limit,
        )
        if not keyword_matches and not vector_matches:
            return None

        merged = _merge_hybrid_matches(
            keyword_matches,
            vector_matches,
            hybrid_alpha=self.config.hybrid_alpha,
            min_vector_score=self.config.min_vector_score,
        )
        limited = merged[: self.config.max_matches]
        return _knowledge_result(query, backend="hybrid", matches=limited)


class KnowledgeRetriever:
    def __init__(self, config: KnowledgeConfig) -> None:
        self.config = config
        self.backend: KnowledgeBackend | None
        if not config.enabled:
            self.backend = None
        elif config.backend == "keyword":
            self.backend = KeywordKnowledgeBackend(config)
        elif config.backend == "vector":
            self.backend = VectorKnowledgeBackend(config)
        elif config.backend == "hybrid":
            self.backend = HybridKnowledgeBackend(config)
        else:
            raise ValueError(f"Unsupported knowledge backend: {config.backend}")

    def preload(self) -> None:
        if self.backend is None:
            return
        self.backend.preload()

    def is_loaded(self) -> bool:
        if self.backend is None:
            return False
        return self.backend.is_loaded()

    def search(self, query: str) -> KnowledgeSearchResult | None:
        if self.backend is None:
            return None
        return self.backend.search(query)


def build_knowledge_index(config: KnowledgeConfig) -> dict[str, Any]:
    sections = _load_sections(config.directory, snippet_char_limit=config.snippet_char_limit)
    signature = _directory_signature(config.directory)
    index_dir = config.index_directory
    manifest_path = index_dir / INDEX_MANIFEST_FILENAME
    sections_path = index_dir / INDEX_SECTIONS_FILENAME
    vectors_path = index_dir / INDEX_VECTORS_FILENAME

    current_manifest = _read_manifest(manifest_path)
    if (
        current_manifest is not None
        and manifest_path.exists()
        and sections_path.exists()
        and vectors_path.exists()
        and _manifest_matches(current_manifest, config, signature, len(sections))
    ):
        return {
            "updated": False,
            "backend": config.backend,
            "embedding_model": config.embedding_model,
            "index_directory": str(index_dir),
            "section_count": len(sections),
        }

    index_dir.mkdir(parents=True, exist_ok=True)
    if sections:
        embedder = _create_text_embedder(config)
        vectors = embedder.embed_documents([section.content_text for section in sections])
        embedding_model = embedder.model_name
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)
        embedding_model = config.embedding_model

    np.save(vectors_path, vectors.astype(np.float32, copy=False))
    _write_vector_section_records(sections_path, sections)
    manifest = {
        "version": INDEX_VERSION,
        "embedding_model": embedding_model,
        "embedding_pooling": EMBEDDING_POOLING,
        "source_signature": _serialize_signature(signature),
        "snippet_char_limit": config.snippet_char_limit,
        "section_count": len(sections),
        "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "updated": True,
        "backend": config.backend,
        "embedding_model": embedding_model,
        "index_directory": str(index_dir),
        "section_count": len(sections),
    }


def _directory_signature(directory: Path) -> tuple[tuple[str, int, int], ...] | None:
    if not directory.exists():
        return None
    signature: list[tuple[str, int, int]] = []
    for path in _iter_knowledge_files(directory):
        stat = path.stat()
        signature.append((str(path.relative_to(directory)), int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def _iter_knowledge_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    root = directory.resolve()
    for path in sorted(directory.rglob("*")):
        if any(part.startswith(("_", ".")) for part in path.relative_to(directory).parts):
            continue
        try:
            if not path.resolve().is_relative_to(root):
                continue
        except (OSError, RuntimeError):
            continue
        if not path.is_file():
            continue
        if path.name in IGNORED_FILENAMES:
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)
    return files


def _load_sections(directory: Path, *, snippet_char_limit: int) -> list[_KnowledgeSection]:
    sections: list[_KnowledgeSection] = []
    if directory.exists():
        for path in _iter_knowledge_files(directory):
            sections.extend(
                _parse_document(
                    path,
                    root=directory,
                    snippet_char_limit=snippet_char_limit,
                )
            )
    return sections


def _parse_document(path: Path, *, root: Path, snippet_char_limit: int) -> list[_KnowledgeSection]:
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(root.resolve()):
        return []
    raw = resolved_path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(raw)

    title = metadata.get("title") or _first_heading(body) or _humanize_stem(path.stem)
    tags = _parse_list_value(metadata.get("tags", ""))
    aliases = _parse_list_value(metadata.get("aliases", ""))
    summary = _clean_text(metadata.get("summary", ""))
    relative_path = path.relative_to(root)
    path_tokens = Counter(_tokenize(relative_path.as_posix().replace("/", " ")))
    tag_tokens = Counter(_tokenize(" ".join((*tags, *aliases))))
    title_tokens = Counter(_tokenize(title))

    sections = _split_sections(body)
    if not sections:
        sections = [(None, body)]

    parsed: list[_KnowledgeSection] = []
    for section_title, section_text in sections:
        cleaned = _clean_text(section_text)
        if not cleaned:
            continue
        heading_tokens = Counter(_tokenize(section_title or ""))
        body_tokens = Counter(_tokenize(cleaned))
        snippet = _truncate(cleaned, snippet_char_limit)
        content_text = "\n".join(
            part for part in (title, section_title or "", summary or "", cleaned) if part
        )
        searchable_text = content_text.lower()
        section_id = hashlib.sha1(
            "\n".join((relative_path.as_posix(), section_title or "", cleaned)).encode("utf-8")
        ).hexdigest()
        parsed.append(
            _KnowledgeSection(
                section_id=section_id,
                source_path=relative_path,
                title=title,
                section_title=section_title,
                summary=summary or None,
                snippet=snippet,
                content_text=content_text,
                title_tokens=title_tokens,
                tag_tokens=tag_tokens,
                heading_tokens=heading_tokens,
                path_tokens=path_tokens,
                body_tokens=body_tokens,
                searchable_text=searchable_text,
            )
        )
    return parsed


def _score_section(section: _KnowledgeSection, query: str, query_tokens: tuple[str, ...]) -> float:
    matched_tokens: set[str] = set()
    score = 0.0

    for token in query_tokens:
        token_score = 0.0
        if section.title_tokens.get(token):
            token_score += 6.0
        if section.tag_tokens.get(token):
            token_score += 5.0
        if section.heading_tokens.get(token):
            token_score += 4.5
        if section.path_tokens.get(token):
            token_score += 4.0
        body_hits = section.body_tokens.get(token, 0)
        if body_hits:
            token_score += min(body_hits, 3) * 1.5
        if token_score > 0:
            matched_tokens.add(token)
        score += token_score

    if not matched_tokens:
        return 0.0

    normalized_query = _clean_text(query).lower()
    if normalized_query and normalized_query in section.searchable_text:
        score += 4.0

    coverage = len(matched_tokens) / max(1, len(query_tokens))
    score *= 0.65 + (coverage * 0.75)
    return score


def _keyword_candidates(
    sections: list[_KnowledgeSection],
    query: str,
    *,
    max_matches: int,
    min_score: float,
) -> list[_ScoredKnowledgeSection]:
    query_tokens = _meaningful_tokens(query)
    if not query_tokens:
        return []

    matches: list[_ScoredKnowledgeSection] = []
    for section in sections:
        score = _score_section(section, query, query_tokens)
        if score < min_score:
            continue
        matches.append(
            _ScoredKnowledgeSection(
                section_id=section.section_id,
                source_path=section.source_path,
                title=section.title,
                section_title=section.section_title,
                snippet=section.snippet,
                score=round(score, 3),
            )
        )

    matches.sort(
        key=lambda item: (-item.score, item.source_path.as_posix(), item.section_title or "")
    )
    return matches[:max_matches]


def _knowledge_result(
    query: str,
    *,
    backend: str,
    matches: list[_ScoredKnowledgeSection],
) -> KnowledgeSearchResult | None:
    if not matches:
        return None
    return KnowledgeSearchResult(
        query=query,
        backend=backend,
        matches=tuple(
            KnowledgeMatch(
                source_path=match.source_path,
                title=match.title,
                section_title=match.section_title,
                snippet=match.snippet,
                score=round(match.score, 3),
            )
            for match in matches
        ),
    )


def _merge_hybrid_matches(
    keyword_matches: list[_ScoredKnowledgeSection],
    vector_matches: list[_ScoredKnowledgeSection],
    *,
    hybrid_alpha: float,
    min_vector_score: float,
) -> list[_ScoredKnowledgeSection]:
    keyword_by_id = {match.section_id: match for match in keyword_matches}
    vector_by_id = {match.section_id: match for match in vector_matches}
    max_keyword = max((match.score for match in keyword_matches), default=0.0)
    max_vector = max((match.score for match in vector_matches), default=min_vector_score)

    merged: list[_ScoredKnowledgeSection] = []
    for section_id in set(keyword_by_id) | set(vector_by_id):
        keyword_match = keyword_by_id.get(section_id)
        vector_match = vector_by_id.get(section_id)
        base = keyword_match or vector_match
        if base is None:
            continue

        keyword_norm = 0.0
        if keyword_match is not None and max_keyword > 0:
            keyword_norm = keyword_match.score / max_keyword

        vector_norm = 0.0
        if vector_match is not None:
            if max_vector <= min_vector_score:
                vector_norm = 1.0
            else:
                vector_norm = max(
                    0.0,
                    min(
                        1.0,
                        (vector_match.score - min_vector_score) / (max_vector - min_vector_score),
                    ),
                )

        combined = ((1.0 - hybrid_alpha) * keyword_norm) + (hybrid_alpha * vector_norm)
        if keyword_match is not None and vector_match is not None:
            combined += 0.1

        merged.append(
            _ScoredKnowledgeSection(
                section_id=base.section_id,
                source_path=base.source_path,
                title=base.title,
                section_title=base.section_title,
                snippet=base.snippet,
                score=round(combined, 3),
            )
        )

    merged.sort(
        key=lambda item: (-item.score, item.source_path.as_posix(), item.section_title or "")
    )
    return merged


def _serialize_signature(signature: tuple[tuple[str, int, int], ...] | None) -> list[list[Any]] | None:
    if signature is None:
        return None
    return [[path, modified_ns, size] for path, modified_ns, size in signature]


def _deserialize_signature(value: Any) -> tuple[tuple[str, int, int], ...] | None:
    if value is None:
        return None
    rows: list[tuple[str, int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        rows.append((str(item[0]), int(item[1]), int(item[2])))
    return tuple(rows)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_matches(
    manifest: dict[str, Any],
    config: KnowledgeConfig,
    source_signature: tuple[tuple[str, int, int], ...] | None,
    section_count: int,
) -> bool:
    return (
        manifest.get("version") == INDEX_VERSION
        and manifest.get("embedding_model") == config.embedding_model
        and manifest.get("embedding_pooling") == EMBEDDING_POOLING
        and int(manifest.get("snippet_char_limit", -1)) == config.snippet_char_limit
        and int(manifest.get("section_count", -1)) == section_count
        and _deserialize_signature(manifest.get("source_signature")) == source_signature
    )


def _write_vector_section_records(path: Path, sections: list[_KnowledgeSection]) -> None:
    lines = []
    for section in sections:
        payload = {
            "section_id": section.section_id,
            "source_path": section.source_path.as_posix(),
            "title": section.title,
            "section_title": section.section_title,
            "snippet": section.snippet,
        }
        lines.append(json.dumps(payload, ensure_ascii=True))
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_vector_section_records(path: Path) -> list[_VectorSectionRecord]:
    records: list[_VectorSectionRecord] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            _VectorSectionRecord(
                section_id=str(payload["section_id"]),
                source_path=Path(str(payload["source_path"])),
                title=str(payload["title"]),
                section_title=None
                if payload.get("section_title") in {None, ""}
                else str(payload["section_title"]),
                snippet=str(payload["snippet"]),
            )
        )
    return records


def _should_attempt_semantic_search(query: str) -> bool:
    if any(pattern.search(query) for pattern in SMALL_TALK_PATTERNS):
        return False
    tokens = _meaningful_tokens(query)
    if not tokens:
        return False
    if len(tokens) == 1 and tokens[0] in {"hello", "hi", "hey", "thanks", "goodbye", "bye"}:
        return False
    return True


def _create_text_embedder(config: KnowledgeConfig) -> TextEmbedder:
    return BgeTextEmbedder(config.embedding_model)


class BgeTextEmbedder:
    def __init__(self, model_name: str, *, batch_size: int = 24) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding support requires torch and transformers to be installed."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self._torch = torch
        self._F = F
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._embed_texts([f"{QUERY_PREFIX}{text}"])
        return vector[0]

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                outputs = self._model(**encoded)
                # BGE v1.5 uses the first token (CLS), then L2 normalization.
                pooled = outputs.last_hidden_state[:, 0]
                normalized = self._F.normalize(pooled, p=2, dim=1)
            batches.append(normalized.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(batches)

def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        return {}, raw

    end_marker = "\n---\n"
    end = raw.find(end_marker, 4)
    if end == -1:
        return {}, raw

    front_matter = raw[4:end]
    body = raw[end + len(end_marker) :]
    metadata: dict[str, str] = {}
    for line in front_matter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata, body


def _parse_list_value(value: str) -> tuple[str, ...]:
    raw = value.strip()
    if not raw:
        return ()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    items = [item.strip().strip("'\"") for item in raw.split(",")]
    return tuple(item for item in items if item)


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if current_heading is not None or any(item.strip() for item in current_lines):
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_heading is not None or any(item.strip() for item in current_lines):
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    return None


def _humanize_stem(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").title()


def _clean_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


def _meaningful_tokens(text: str) -> tuple[str, ...]:
    raw = [token for token in _tokenize(text) if len(token) > 1]
    filtered = [token for token in raw if token not in STOPWORDS]
    tokens = filtered or raw
    deduped = list(dict.fromkeys(tokens))
    return tuple(deduped)


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    shortened = cleaned[:limit].rsplit(" ", 1)[0].strip()
    return f"{shortened}..."
