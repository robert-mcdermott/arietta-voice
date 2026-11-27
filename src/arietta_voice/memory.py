from __future__ import annotations

import fcntl
import hashlib
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from arietta_voice.private_config import write_private_text

# Provenance travels with the fact in the one private memory file. There is no
# second store to reconcile or to purge when the owner forgets something.
ORIGIN = re.compile(r" <!-- arietta-memory:(chat|voice|consolidated|file)(:edited)? -->$")
FACT = re.compile(r"^[-*] +(.+)$")
EXTRACT = (
    "Record only durable facts or preferences the user explicitly stated about themselves. "
    "Do not infer facts from questions, assistant answers, or hypothetical examples. "
    "Skip facts already in memory. Output one plain '- ' bullet per new fact, or only NONE."
)


class MemoryConflict(ValueError):
    pass


def prompt_memory(text: str) -> str:
    return "\n".join(ORIGIN.sub("", line) for line in text.split("\n"))


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
        if (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and not lines[i + 1].startswith(("- ", "* ", "#"))
        ):
            continue
        origin = ORIGIN.search(fact[1])
        value = ORIGIN.sub("", fact[1]).strip()
        if value:
            yield i, value, origin[1] if origin else "file", bool(origin and origin[2])


def _consolidation_prompt(text):
    # The model can consolidate simple facts. Only the owner edits free-form
    # Markdown and facts they have explicitly corrected.
    entries = {i: (value, edited) for i, value, _, edited in _fact_lines(text)}
    for i, line in enumerate(text.splitlines()):
        if i not in entries and line.strip() not in ("", "# Memory"):
            return None
    facts = [f"- {value}" for value, edited in entries.values() if not edited]
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
    origins = {
        value.casefold(): (origin, edited) for _, value, origin, edited in _fact_lines(previous)
    }
    fixed = [line for line in previous.splitlines() if (mark := ORIGIN.search(line)) and mark[2]]
    cleaned = ["# Memory"]
    for line in lines[1:]:
        if not line.strip():
            continue
        match = FACT.fullmatch(line)
        if not match or "<!--" in line:
            return None
        value = match[1].strip()
        source, edited = origins.get(value.casefold(), ("consolidated", False))
        if not edited:
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

    def inspect(self):
        text = self.path.read_text(encoding="utf-8") if self.path.is_file() else ""
        entries = {}
        for i, value, origin, edited in _fact_lines(text):
            entries.setdefault(
                value.casefold(), dict(id=i, text=value, origin=origin, edited=edited)
            )
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
                selected = next((row for row in entries if row[0] == entry_id), None)
                if selected is None:
                    raise MemoryConflict("This fact is no longer available. Reload memory.")
                lines = current.splitlines(keepends=True)
                if content is not None:
                    content = content.strip()
                    if not content or len(content.splitlines()) != 1 or "<!--" in content:
                        raise ValueError("Enter a fact on one line.")
                for i, value, origin, _ in entries:
                    if value.casefold() == selected[1].casefold():
                        end = lines[i][len(lines[i].rstrip("\r\n")) :]
                        lines[i] = (
                            f"- {content} <!-- arietta-memory:{origin}:edited -->{end}"
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
            f"Current memory:\n{prompt_memory(expected or '# Memory')}\n\n"
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

        lines = [line.strip() for line in result.splitlines() if line.strip().startswith("-")]
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
            f"Current memory:\n{prompt_memory(current)}\n\nUser said: {user_text}\nAssistant said: {assistant_text}\n\n"
            + EXTRACT,
            80,
        )
        facts = (
            []
            if not result or result.upper() == "NONE"
            else [line.strip() for line in result.splitlines() if line.strip().startswith("-")]
        )
        if origin in ("voice", "chat"):
            known = {value.casefold() for _, value, _, _ in _fact_lines(current)}
            new = []
            for fact in facts:
                value = fact[1:].strip()
                if value and "<!--" not in value and value.casefold() not in known:
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
