from __future__ import annotations

import hashlib
import json
import logging
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from arietta_voice.config import KnowledgeConfig

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
WHITESPACE_RE = re.compile(r"\s+")
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
IGNORED_FILENAMES = {"README.md"}
INDEX_VERSION = 3  # 3: long sections are indexed in chunks.
EMBEDDING_POOLING = "cls-normalized"
INDEX_MANIFEST_FILENAME = "manifest.json"
INDEX_SECTIONS_FILENAME = "sections.jsonl"
INDEX_VECTORS_FILENAME = "vectors.npy"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Notes are found for most requests, often unrelated. Telling the model to be
# unsure whenever they did not answer made it refuse half of ordinary questions
# it knew ("how many legs does a spider have?"). The rule is part of the system
# prompt, which is the same every turn and so is reused, not read again with
# every set of notes.
NOTES_RULE = (
    "Notes from the local knowledge base may come just before a question. They are "
    "found by a keyword search, so they may have nothing to do with the question. If "
    "they answer it, use them. If not, ignore them completely and answer from your own "
    "knowledge."
)
NOTES_HEADING = "Knowledge base notes (they may be unrelated to the question):"
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
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "re",
    "should",
    "something",
    "tell",
    "the",
    "to",
    "ve",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
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
    section_id: str = ""

    def prompt_line(self, label: str) -> str:
        heading = self.title
        if self.section_title and self.section_title != self.title:
            heading = f"{self.title} — {self.section_title}"
        return f"[{label}] {heading}: {self.snippet}"


@dataclass(slots=True, frozen=True)
class KnowledgeSearchResult:
    query: str
    backend: str
    matches: tuple[KnowledgeMatch, ...]

    def prompt_context(self) -> str:
        """The notes as the model reads them: one line each, "[1] Title — Section: text".

        The text is the note's own; only the labels around it are short. The
        model reads about 4 ms a token on an M4, and the former four label lines
        per note (with the file path) were about a quarter of the block.
        """
        lines = [NOTES_HEADING]  # How to use them is in the system prompt (NOTES_RULE).
        for index, match in enumerate(self.matches, start=1):
            lines.append(match.prompt_line(str(index)))
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
        matches = [
            match
            for match in self.candidate_matches(query, max_matches=self.config.max_matches)
            if match.score >= self.config.min_similarity
        ]
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

        # An incomplete/corrupt cache is disposable. Keep keyword retrieval
        # available and let an explicit index build repair it. Model failures
        # are deliberately outside this boundary and still reach turn recovery.
        try:
            manifest = _read_manifest(manifest_path)
            if manifest is None:
                return False
            sections = _load_vector_section_records(sections_path)
            if not _manifest_matches(manifest, self.config, source_signature, len(sections)):
                return False
            source_paths = {item[0] for item in source_signature or ()}
            if any(section.source_path.as_posix() not in source_paths for section in sections):
                return False
            if len({section.section_id for section in sections}) != len(sections):
                return False
            vectors = np.load(vectors_path, allow_pickle=False)
            if not isinstance(vectors, np.ndarray):
                vectors.close()  # Reject an npz archive disguised as the npy file.
                return False
            if (
                vectors.ndim != 2
                or vectors.shape != (len(sections), manifest.get("vector_dim"))
                or (bool(sections) and vectors.shape[1] == 0)
                or not np.issubdtype(vectors.dtype, np.floating)
                or not np.isfinite(vectors).all()
            ):
                return False
            vectors = vectors.astype(np.float32, copy=False)
            if not np.isfinite(vectors).all():
                return False
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            EOFError,
            zipfile.BadZipFile,
        ):
            return False

        self._sections = sections
        self._vectors = vectors
        self._index_signature = source_signature
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
        # Every attached note costs the model time to read (about 4 ms a token on
        # an M4, so a second for three notes), and almost every question shares a
        # few words with some note. With an index, a note must be close in
        # meaning; keywords only rank. Without one, the keyword search's own bar
        # applies, as with the keyword backend, and also when the question was
        # not searched by meaning ("Hi, what's the Wi-Fi called?").
        if _should_attempt_semantic_search(query) and self.vector_backend.is_loaded():
            close = {
                match.section_id
                for match in vector_matches
                if match.score >= self.config.min_similarity
            }
            vector_matches = [match for match in vector_matches if match.section_id in close]
            keyword_matches = [match for match in keyword_matches if match.section_id in close]
        else:
            keyword_matches = [
                match for match in keyword_matches if match.score >= self.config.min_score
            ]
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

    def build_index_if_needed(self) -> bool:
        """Build a missing or outdated index for hybrid or vector search.

        Without one, those backends quietly search by keyword only. Called at
        startup, never during a turn. Returns whether an index was built.
        """
        if not isinstance(self.backend, (HybridKnowledgeBackend, VectorKnowledgeBackend)):
            return False
        vector = getattr(self.backend, "vector_backend", self.backend)
        if vector._ensure_index_loaded(preload_embedder=False):
            return False
        try:
            payload = build_knowledge_index(self.config)
        except Exception:  # A model that cannot load leaves keyword search.
            LOGGER.warning("Could not build the knowledge index", exc_info=True)
            return False
        LOGGER.info("Built the knowledge index (%d sections)", payload["section_count"])
        return bool(payload["updated"])

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

    cached = VectorKnowledgeBackend(config)
    if cached._ensure_index_loaded(preload_embedder=False) and len(cached._sections) == len(
        sections
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
        signature.append(
            (str(path.relative_to(directory)), int(stat.st_mtime_ns), int(stat.st_size))
        )
    return tuple(signature)


def _is_searchable_note(relative: Path, resolved_root: Path) -> bool:
    if (
        relative.is_absolute()
        or any(part.startswith(("_", ".")) for part in relative.parts)
        or relative.name in IGNORED_FILENAMES
        or relative.suffix.lower() not in SUPPORTED_EXTENSIONS
    ):
        return False
    try:
        path = (resolved_root / relative).resolve()
        return path.is_relative_to(resolved_root) and path.is_file()
    except (OSError, RuntimeError):
        return False


def _iter_knowledge_files(directory: Path) -> list[Path]:
    root = directory.resolve()
    return [
        path
        for path in sorted(directory.rglob("*"))
        if _is_searchable_note(path.relative_to(directory), root)
    ]


def cited_passage(
    config: KnowledgeConfig, relative: Path, section_id: str
) -> _KnowledgeSection | None:
    """Look up the cited chunk in the current note without trusting a stale index."""
    root = config.directory
    if not config.enabled or not _is_searchable_note(relative, root.resolve()):
        return None
    for section in _parse_document(
        root / relative, root=root, snippet_char_limit=config.snippet_char_limit
    ):
        if section.section_id == section_id:
            return section
    return None


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
        heading_tokens = Counter(_tokenize(section_title or ""))
        # A long section is searched and attached in pieces, so every part of it
        # can reach the model; before, only its first characters ever did.
        for index, chunk in enumerate(_chunk_text(section_text, snippet_char_limit)):
            content_text = "\n".join(
                part for part in (title, section_title or "", summary or "", chunk) if part
            )
            # Stable while the text is unchanged: the file, heading, position and text.
            section_id = hashlib.sha1(
                "\n".join(
                    (relative_path.as_posix(), section_title or "", str(index), chunk)
                ).encode("utf-8")
            ).hexdigest()
            parsed.append(
                _KnowledgeSection(
                    section_id=section_id,
                    source_path=relative_path,
                    title=title,
                    section_title=section_title,
                    summary=summary or None,
                    snippet=chunk,
                    content_text=content_text,
                    title_tokens=title_tokens,
                    tag_tokens=tag_tokens,
                    heading_tokens=heading_tokens,
                    path_tokens=path_tokens,
                    body_tokens=Counter(_tokenize(chunk)),
                    searchable_text=content_text.lower(),
                )
            )
    return parsed


LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\|)")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9])")
ABBREVIATION_RE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|Sr|Jr|vs|etc|e\.g|i\.e|No)\.$")


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split a section into pieces of at most ``limit`` characters.

    A section that fits stays whole, exactly as before. A longer one is cut
    between list items and sentences, never inside one unless a single
    sentence is longer than the limit, which is then cut between words.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= limit:
        return [cleaned]
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if any(LIST_ITEM_RE.match(line) for line in lines):
            # A list: each item, with any wrapped lines, is one unit.
            items: list[str] = []
            for line in lines:
                if LIST_ITEM_RE.match(line) or not items:
                    items.append(line)
                else:
                    items[-1] += " " + line
            units.extend(_clean_text(item) for item in items)
        else:
            units.extend(_sentences(_clean_text(" ".join(lines))))
    chunks: list[str] = []
    current = ""
    for unit in units:
        for piece in _split_long(unit, limit):
            if current and len(current) + 1 + len(piece) > limit:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in SENTENCE_END_RE.split(text):
        if sentences and ABBREVIATION_RE.search(sentences[-1]):
            sentences[-1] += " " + part  # "Dr. Okafor" is one sentence.
        else:
            sentences.append(part)
    return [sentence for sentence in sentences if sentence]


def _split_long(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit + 1)
        cut = cut if cut > 0 else limit
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    return [*pieces, text] if text else pieces


def _score_section(section: _KnowledgeSection, query: str, query_tokens: tuple[str, ...]) -> float:
    matched_tokens: set[str] = set()
    body_terms = set()
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
        body_term = token
        if token not in section.body_tokens and len(token) >= 4 and token.isalpha():
            # Allow a regular final -s in either direction (roller/rollers,
            # leave/leaves). Keep exact matches for titles, names and tags.
            if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
                body_term = token[:-1]
            else:
                body_term = token + "s"
        body_hits = section.body_tokens.get(body_term, 0)
        if body_hits:
            if token not in STOPWORDS:
                body_terms.add(body_term)
            token_score += min(body_hits, 3) * 1.5
        if token_score > 0:
            matched_tokens.add(token)
        score += token_score

    if not matched_tokens:
        return 0.0

    # Several distinct query terms in the same passage are stronger evidence
    # than repeats of one word or a tag shared by every passage in the file.
    if len(body_terms) >= 2:
        score += 4.0

    normalized_query = _clean_text(query).lower()
    if normalized_query and normalized_query in section.searchable_text:
        score += 4.0
    elif len(query_tokens) > 1 and len(matched_tokens) == len(query_tokens):
        # "Where is my spare key?" should find "the spare key" in the body,
        # even when neither word appears in the note title or heading.
        phrase = re.escape(" ".join(query_tokens))
        if re.search(rf"\b{phrase}\b", section.snippet, re.IGNORECASE):
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
                section_id=match.section_id,
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


def _serialize_signature(
    signature: tuple[tuple[str, int, int], ...] | None,
) -> list[list[Any]] | None:
    if signature is None:
        return None
    return [[path, modified_ns, size] for path, modified_ns, size in signature]


def _read_manifest(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


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
        and manifest.get("snippet_char_limit") == config.snippet_char_limit
        and manifest.get("section_count") == section_count
        and type(manifest.get("vector_dim")) is int
        and manifest.get("source_signature") == _serialize_signature(source_signature)
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
        if (
            not isinstance(payload, dict)
            or any(
                not isinstance(payload.get(key), str)
                for key in ("section_id", "source_path", "title", "snippet")
            )
            or not (
                payload.get("section_title") is None
                or isinstance(payload.get("section_title"), str)
            )
        ):
            raise ValueError("Invalid cached section record")
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
