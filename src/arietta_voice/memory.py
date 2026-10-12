from __future__ import annotations

from pathlib import Path
import os
from typing import Protocol


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_atomic("# Memory\n")

    def _write_atomic(self, text: str) -> None:
        # A killed inference worker must leave the previous memory intact.
        working = self.path.with_name(f".{self.path.name}.working")
        try:
            with os.fdopen(os.open(working, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            working.replace(self.path)
        finally:
            working.unlink(missing_ok=True)

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else "# Memory\n"

    def update_from_turn(self, generator: TextGenerator, user_text: str, assistant_text: str) -> int:
        if not self.enabled:
            return 0

        prompt = (
            f"Current memory:\n{self.read_text()}\n\n"
            f"User said: {user_text}\n"
            f"Assistant said: {assistant_text}\n\n"
            "Did the user reveal a durable fact about themselves, their preferences, or an ongoing project? "
            "If yes, output one fact per line starting with '- '. If no, output ONLY: NONE."
        )
        result = generator.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.2,
        ).strip()
        if not result or "NONE" in result.upper():
            return 0

        lines = [line.strip() for line in result.splitlines() if line.strip().startswith("-")]
        if not lines:
            return 0

        self._write_atomic(self.read_text() + "\n" + "\n".join(lines) + "\n")
        return len(lines)

    def maybe_consolidate(self, generator: TextGenerator, turn_count: int) -> bool:
        if not self.enabled or not self.path.exists():
            return False
        if turn_count == 0 or turn_count % self.consolidate_every_turns != 0:
            return False

        prompt = (
            f"Here is a memory file:\n\n{self.read_text()}\n\n"
            "Rewrite it to keep only durable facts. Remove duplicates and transient details. "
            "Return only the cleaned file, starting with '# Memory' and bullet facts."
        )
        result = generator.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        ).strip()
        if not result.startswith("# Memory"):
            return False

        self._write_atomic(result + "\n")
        return True
