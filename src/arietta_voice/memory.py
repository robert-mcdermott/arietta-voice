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
        if not result or result.upper() == "NONE":
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

    def prepare_update(self, generator, user_text, assistant_text, turn_count):
        """Yield between model events; return a proposal without modifying disk."""
        if not self.enabled:
            return None
        expected = self._background_snapshot()
        current = expected if expected is not None else '# Memory\n'
        def generate(prompt, max_tokens):
            stream = generator.stream([{'role':'user', 'content':prompt}], max_tokens=max_tokens, temperature=.2)
            parts, chars, complete = [], 0, False
            try:
                for event in stream:
                    if complete:
                        raise ValueError('Memory event after completion')
                    if event.kind == 'text':
                        chars += len(event.text)
                        if chars > 12_000:
                            raise ValueError('Memory response exceeds limit')
                        if event.text:
                            parts.append(event.text)
                    elif event.kind == 'done':
                        complete = True
                    else:
                        raise ValueError('Invalid memory event')
                    yield None
                if not complete:
                    raise RuntimeError('Incomplete memory generation')
                return ''.join(parts).strip()
            finally:
                stream.close()

        result = yield from generate(
            f'Current memory:\n{current}\n\nUser said: {user_text}\nAssistant said: {assistant_text}\n\n'
            'Did the user reveal a durable fact about themselves, their preferences, or an ongoing project? '
            "If yes, output one fact per line starting with '- '. If no, output ONLY: NONE.", 80)
        facts = [] if not result or result.upper() == 'NONE' else [
            line.strip() for line in result.splitlines() if line.strip().startswith('-')]
        proposed = current + '\n' + '\n'.join(facts) + '\n' if facts else current
        consolidated = False
        if turn_count > 0 and turn_count % self.consolidate_every_turns == 0:
            result = yield from generate(
                f'Here is a memory file:\n\n{proposed}\n\n'
                "Rewrite it to keep only durable facts. Remove duplicates and transient details. "
                "Return only the cleaned file, starting with '# Memory' and bullet facts.", 300)
            if result.startswith('# Memory'):
                proposed, consolidated = result + '\n', True
        if len(proposed) > 128_000:
            raise ValueError('Proposed memory exceeds background generation limit')
        return dict(expected=expected, text=proposed, facts_added=len(facts), consolidated=consolidated)

    def commit_update(self, proposal):
        if not self.enabled or proposal is None:
            return dict(facts_added=0, consolidated=False, conflict=False)
        current = self._background_snapshot()
        if current != proposal['expected']:
            return dict(facts_added=0, consolidated=False, conflict=True)
        if current != proposal['text']:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_atomic(proposal['text'])
        return dict(facts_added=proposal['facts_added'], consolidated=proposal['consolidated'], conflict=False)

    def _background_snapshot(self):
        try:
            with self.path.open(encoding='utf-8') as handle:
                text = handle.read(128_001)
        except FileNotFoundError:
            return None
        if len(text) > 128_000:
            raise ValueError('Memory file exceeds background generation limit')
        return text
