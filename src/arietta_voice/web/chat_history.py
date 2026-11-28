"""Owner-local storage for completed console Chat turns."""

import json
import os
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

PAGE_SIZE = 25


class ChatHistory:
    def __init__(self, directory: Path) -> None:
        self.path = directory / "chat-history.sqlite3"

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            db.create_function("casefold", 1, str.casefold, deterministic=True)
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA secure_delete = ON")
            db.execute(
                "CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, title TEXT NOT NULL)"
            )
            db.execute("""CREATE TABLE IF NOT EXISTS turns (
                turn_id INTEGER PRIMARY KEY, chat TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                question TEXT NOT NULL, answer TEXT NOT NULL, meta TEXT NOT NULL,
                saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS turns_by_chat ON turns(chat, turn_id)")
            yield db

    def append(self, conversation_id: str | None, question: str, result: dict) -> str:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if conversation_id is None:
                conversation_id = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO chats VALUES (?, ?)",
                    (conversation_id, " ".join(question.split())[:100]),
                )
            elif not db.execute("SELECT 1 FROM chats WHERE id = ?", (conversation_id,)).fetchone():
                # A concurrent deletion wins. A stale tab cannot recreate a conversation.
                raise KeyError(conversation_id)
            meta = {
                key: result.get(key)
                for key in ("source", "sources", "used_tool", "model_tool", "smarter")
            }
            db.execute(
                "INSERT INTO turns (chat, question, answer, meta) VALUES (?, ?, ?, ?)",
                (conversation_id, question, result["response"], json.dumps(meta)),
            )
        return conversation_id

    def search(self, query: str, offset: int) -> list[dict]:
        if not self.path.exists():
            return []
        with self._connection() as db:
            rows = db.execute(
                """SELECT chats.id, chats.title, MAX(saved_at) AS saved_at
                FROM chats JOIN turns ON chats.id = turns.chat
                WHERE ? = '' OR chats.id IN (
                    SELECT chat FROM turns WHERE instr(casefold(question), ?) OR instr(casefold(answer), ?)
                )
                GROUP BY chats.id ORDER BY MAX(turn_id) DESC LIMIT ? OFFSET ?""",
                (query, query.casefold(), query.casefold(), PAGE_SIZE + 1, offset),
            )
            return [dict(row) for row in rows]

    def read(self, conversation_id: str) -> list[dict]:
        if self.path.exists():
            with self._connection() as db:
                messages = []
                for row in db.execute(
                    "SELECT question, answer, meta FROM turns WHERE chat = ? ORDER BY turn_id",
                    (conversation_id,),
                ):
                    messages.extend(
                        (
                            {"role": "user", "content": row["question"]},
                            {
                                "role": "assistant",
                                "content": row["answer"],
                                "meta": json.loads(row["meta"]),
                            },
                        )
                    )
                if messages:
                    return messages
        raise KeyError(conversation_id)

    def delete(self, conversation_id: str) -> None:
        if self.path.exists():
            with self._connection() as db:
                if db.execute("DELETE FROM chats WHERE id = ?", (conversation_id,)).rowcount:
                    return
        raise KeyError(conversation_id)
