from __future__ import annotations

import fcntl
import hashlib
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from arietta_voice.private_config import write_private_text

# Provenance travels with the fact in the one private memory file. There is no
# second store to reconcile or to purge when the owner forgets something.
ORIGIN = re.compile(
    r" <!-- arietta-memory:(?P<origin>chat|voice|consolidated|file)"
    r"(?::saved=(?P<saved_on>[0-9]{4}-[0-9]{2}-[0-9]{2}))?(?P<edited>:edited)? -->$"
)
SAVE_REQUEST = re.compile(
    r"^(?:please\s+)?(?:remember that|remember this (?:decision|memory)|remember(?: this)?(?=\s*:)"
    r"|save (?:a dated memory|this (?:decision|memory)|this (?:as a dated memory|to memory)))"
    r"(?=\s|[:,.!]|$)\s*[:,]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
FACT = re.compile(r"^[-*] +(.+)$")
EXTRACT = (
    "Record only durable facts or preferences the user explicitly stated about themselves. "
    "Do not infer facts from questions, assistant answers, or hypothetical examples. "
    "Skip facts already in memory. Output one plain '- ' bullet per new fact, or only NONE."
)


class MemoryConflict(ValueError):
    pass


def requested_memory(message: str) -> str | None:
    """Only direct save commands in the current user's message create dated entries."""
    message = message.strip()
    match = SAVE_REQUEST.fullmatch(message)
    if match is None or message.endswith("?"):
        return None
    value = match[1].strip()
    return value if value.strip(".,!:") else ""


@dataclass(frozen=True)
class _Entry:
    id: int
    text: str
    origin: str
    edited: bool
    saved_on: str | None

    @property
    def key(self):
        return self.saved_on, self.text.casefold()


def prompt_memory(text: str, *, include_dated: bool = True) -> str:
    lines = []
    for line in text.splitlines():
        mark = ORIGIN.search(line)
        if mark and mark["saved_on"]:
            if include_dated:
                lines.append(f"{line[: mark.start()]} (saved {mark['saved_on']})")
        else:
            lines.append(ORIGIN.sub("", line))
    return "\n".join(lines)


def _revision(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _fact_lines(text):
    """Editable single-line Markdown bullets; leave prose and complex Markdown intact."""
    lines = text.splitlines()
    fence = ""
    for i, line in enumerate(lines):
        if fence:
            if line.lstrip().startswith(fence):
                fence = ""
            continue
        delimiter = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if delimiter:
            fence = delimiter[1]
            continue
        fact = FACT.fullmatch(line)
        if not fact:
            continue
        origin = ORIGIN.search(fact[1])
        if (
            not (origin and origin["saved_on"])
            and i + 1 < len(lines)
            and lines[i + 1].strip()
            and not lines[i + 1].startswith(("- ", "* ", "#"))
        ):
            continue
        value = ORIGIN.sub("", fact[1]).strip()
        if value:
            yield _Entry(
                i,
                value,
                origin["origin"] if origin else "file",
                bool(origin and origin["edited"]),
                origin["saved_on"] if origin else None,
            )


def _consolidation_prompt(text):
    # The model can consolidate simple facts. Only the owner edits free-form
    # Markdown and facts they have explicitly corrected.
    entries = {entry.id: entry for entry in _fact_lines(text)}
    for i, line in enumerate(text.splitlines()):
        if i not in entries and line.strip() not in ("", "# Memory"):
            return None
    facts = [
        f"- {entry.text}" for entry in entries.values() if not entry.edited and not entry.saved_on
    ]
    if not facts:
        return None
    return (
        "Here is a memory file:\n\n# Memory\n"
        + "\n".join(facts)
        + "\n\nRewrite it to keep only durable facts. Remove duplicates and transient details. "
        "Return only the cleaned file, starting with '# Memory' and bullet facts."
    )


def _consolidated_text(result, previous, *, annotate):
    lines = result.splitlines()
    if not lines or lines[0].strip() != "# Memory":
        return None
    entries = list(_fact_lines(previous))
    origins = {entry.text.casefold(): entry.origin for entry in entries}
    protected = {entry.text.casefold() for entry in entries if entry.edited or entry.saved_on}
    old_lines = previous.splitlines()
    fixed = [old_lines[entry.id] for entry in entries if entry.edited or entry.saved_on]
    cleaned = ["# Memory"]
    for line in lines[1:]:
        if not line.strip():
            continue
        match = FACT.fullmatch(line)
        if not match or "<!--" in line:
            return None
        value = match[1].strip()
        source = origins.get(value.casefold(), "consolidated")
        if value.casefold() not in protected:
            tag = f" <!-- arietta-memory:{source} -->" if annotate else ""
            cleaned.append(f"- {value}{tag}")
    return "\n".join([*cleaned, *fixed]) + "\n"


class TextGenerator(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str: ...


class MemoryStore:
    def __init__(self, path: Path, enabled: bool, consolidate_every_turns: int) -> None:
        self.path = path
        self.enabled = enabled
        self.consolidate_every_turns = consolidate_every_turns

    def ensure_exists(self) -> None:
        if not self.enabled:
            return
        with self._locked():
            if not self.path.exists():
                write_private_text(self.path, "# Memory\n")

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.path.with_name("." + self.path.name + ".lock"),
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "r+") as lock:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield lock

    def save_dated(self, value: str, *, origin: str) -> None:
        if not self.enabled:
            raise ValueError("Memory is off. Turn on memory in Settings to save a dated memory.")
        value = value.strip()
        if not value or len(value.splitlines()) != 1 or "<!--" in value:
            raise ValueError("Tell me the decision to remember in one plain-text statement.")
        if origin not in ("chat", "voice"):
            raise ValueError("Unknown memory source.")
        day = date.today().isoformat()
        with self._locked():
            try:
                with self.path.open(encoding="utf-8", newline="") as handle:
                    current = handle.read(128_001)
            except FileNotFoundError:
                current = "# Memory\n"
            if len(current) > 128_000:
                raise ValueError(
                    "The memory file is full. Review it in Persona before saving more."
                )
            if any(entry.key == (day, value.casefold()) for entry in _fact_lines(current)):
                return
            newline = "\r\n" if "\r\n" in current else "\n"
            record = f"- {value} <!-- arietta-memory:{origin}:saved={day} -->{newline}"
            # Insert outside any free-form Markdown or unclosed fence in an owner's file.
            header = re.match(r"^# Memory(?:\r\n|\n|$)", current)
            start = header.end() if header else 0
            prefix = current[:start]
            if prefix and not prefix.endswith("\n"):
                prefix += newline
            updated = prefix + record + current[start:]
            if len(updated) > 128_000:
                raise ValueError(
                    "The memory file is full. Review it in Persona before saving more."
                )
            write_private_text(self.path, updated)

    def inspect(self):
        text = self.path.read_text(encoding="utf-8") if self.path.is_file() else ""
        entries = {}
        for entry in _fact_lines(text):
            entries.setdefault(entry.key, asdict(entry))
        return dict(
            enabled=self.enabled,
            file=str(self.path),
            content=text,
            revision=_revision(text),
            entries=list(entries.values()),
        )

    def edit(self, content, *, revision=None, entry_id=None):
        with self._locked() as lock:
            # newline="" keeps unrelated CRLF lines intact during per-fact edits.
            try:
                with self.path.open(encoding="utf-8", newline="") as handle:
                    current = handle.read()
            except FileNotFoundError:
                current = ""
            normalized = current.replace("\r\n", "\n").replace("\r", "\n")
            if revision is not None and _revision(normalized) != revision:
                raise MemoryConflict("Memory changed. Reload it and review the latest facts.")
            if entry_id is not None:
                entries = list(_fact_lines(current))
                selected = next((row for row in entries if row.id == entry_id), None)
                if selected is None:
                    raise MemoryConflict("This fact is no longer available. Reload memory.")
                lines = current.splitlines(keepends=True)
                if content is not None:
                    content = content.strip()
                    if not content or len(content.splitlines()) != 1 or "<!--" in content:
                        raise ValueError("Enter a fact on one line.")
                for entry in entries:
                    if entry.key == selected.key:
                        i = entry.id
                        saved = f":saved={entry.saved_on}" if entry.saved_on else ""
                        end = lines[i][len(lines[i].rstrip("\r\n")) :]
                        lines[i] = (
                            f"- {content} <!-- arietta-memory:{entry.origin}{saved}:edited -->{end}"
                            if i == entry_id and content is not None
                            else ""
                        )
                content = "".join(lines)
            # Reject even a queued learning job which only starts after this edit.
            lock.seek(0)
            lock.write(str(time.time_ns()))
            lock.truncate()
            lock.flush()
            write_private_text(self.path, content)

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else "# Memory\n"

    def update_from_turn(
        self, generator: TextGenerator, user_text: str, assistant_text: str
    ) -> int:
        if not self.enabled:
            return 0

        expected = self._background_snapshot()
        prompt = (
            f"Current memory:\n{prompt_memory(expected or '# Memory', include_dated=False)}\n\n"
            f"User said: {user_text}\n"
            f"Assistant said: {assistant_text}\n\n" + EXTRACT
        )
        result = generator.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.2,
        ).strip()
        if not result or result.upper() == "NONE":
            return 0

        lines = [
            line.strip()
            for line in result.splitlines()
            if line.strip().startswith("-") and "<!--" not in line
        ]
        if not lines:
            return 0

        current = expected if expected is not None else "# Memory\n"
        result = self.commit_update(
            dict(
                expected=expected,
                text=current + "\n" + "\n".join(lines) + "\n",
                facts_added=len(lines),
                consolidated=False,
            )
        )
        return result["facts_added"]

    def maybe_consolidate(self, generator: TextGenerator, turn_count: int) -> bool:
        if not self.enabled or not self.path.exists():
            return False
        if turn_count == 0 or turn_count % self.consolidate_every_turns != 0:
            return False

        expected = self._background_snapshot()
        prompt = _consolidation_prompt(expected or "")
        if not prompt:
            return False
        result = generator.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        ).strip()
        text = _consolidated_text(result, expected, annotate="<!-- arietta-memory:" in expected)
        if text is None:
            return False
        return self.commit_update(
            dict(expected=expected, text=text, facts_added=0, consolidated=True)
        )["consolidated"]

    def prepare_update(
        self, generator, user_text, assistant_text, turn_count, *, origin=None, submitted_ns=None
    ):
        """Yield between model events; return a proposal without modifying disk."""
        if not self.enabled:
            return None
        expected = self._background_snapshot()
        current = expected if expected is not None else "# Memory\n"

        def generate(prompt, max_tokens):
            stream = generator.stream(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.2
            )
            parts, chars, complete = [], 0, False
            try:
                for event in stream:
                    if complete:
                        raise ValueError("Memory event after completion")
                    if event.kind == "text":
                        chars += len(event.text)
                        if chars > 12_000:
                            raise ValueError("Memory response exceeds limit")
                        if event.text:
                            parts.append(event.text)
                    elif event.kind == "done":
                        complete = True
                    else:
                        raise ValueError("Invalid memory event")
                    yield None
                if not complete:
                    raise RuntimeError("Incomplete memory generation")
                return "".join(parts).strip()
            finally:
                stream.close()

        result = yield from generate(
            f"Current memory:\n{prompt_memory(current, include_dated=False)}\n\nUser said: {user_text}\nAssistant said: {assistant_text}\n\n"
            + EXTRACT,
            80,
        )
        facts = (
            []
            if not result or result.upper() == "NONE"
            else [
                line.strip()
                for line in result.splitlines()
                if line.strip().startswith("-") and "<!--" not in line
            ]
        )
        if origin in ("voice", "chat"):
            known = {entry.text.casefold() for entry in _fact_lines(current)}
            new = []
            for fact in facts:
                value = fact[1:].strip()
                if value and value.casefold() not in known:
                    new.append(f"- {value} <!-- arietta-memory:{origin} -->")
                    known.add(value.casefold())
            facts = new
        proposed = current + "\n" + "\n".join(facts) + "\n" if facts else current
        consolidated = False
        if turn_count > 0 and turn_count % self.consolidate_every_turns == 0:
            prompt = _consolidation_prompt(proposed)
            if prompt:
                result = yield from generate(prompt, 300)
                cleaned = _consolidated_text(result, proposed, annotate=bool(origin))
                if cleaned is not None:
                    proposed, consolidated = cleaned, True
        if len(proposed) > 128_000:
            raise ValueError("Proposed memory exceeds background generation limit")
        return dict(
            expected=expected,
            text=proposed,
            facts_added=len(facts),
            consolidated=consolidated,
            submitted_ns=submitted_ns,
        )

    def commit_update(self, proposal):
        if not self.enabled or proposal is None:
            return dict(facts_added=0, consolidated=False, conflict=False)
        with self._locked() as lock:
            current = self._background_snapshot()
            edited_ns = int(lock.read() or 0)
            submitted_ns = proposal.get("submitted_ns")
            if current != proposal["expected"] or (
                submitted_ns is not None and submitted_ns <= edited_ns
            ):
                return dict(facts_added=0, consolidated=False, conflict=True)
            if current != proposal["text"]:
                write_private_text(self.path, proposal["text"])
        return dict(
            facts_added=proposal["facts_added"],
            consolidated=proposal["consolidated"],
            conflict=False,
        )

    def _background_snapshot(self):
        try:
            with self.path.open(encoding="utf-8") as handle:
                text = handle.read(128_001)
        except FileNotFoundError:
            return None
        if len(text) > 128_000:
            raise ValueError("Memory file exceeds background generation limit")
        return text
