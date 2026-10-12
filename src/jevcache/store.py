"""Local durable artifacts, expiry, empirical reuse observations, and traces."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .models import snapshot


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class Entry:
    key: str
    kind: str
    scope: str
    task: str
    value: Any
    request: Mapping[str, Any]
    dependencies: Mapping[str, str]
    policy_version: str
    output_contract: str
    created_at: float
    expires_at: float | None
    avoided_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _EmbeddingSearch:
    """SQLite snapshot prepared before potentially slow embedding inference."""

    entries: list[Entry]
    vectors: dict[str, list[float]]
    missing: list[Entry]
    encoder: str
    texts: list[str]
    limit: int


class SQLiteStore:
    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_ms: int = 5,
    ):
        if (
            isinstance(busy_timeout_ms, bool)
            or not isinstance(busy_timeout_ms, int)
            or busy_timeout_ms < 0
        ):
            raise ValueError("Cache busy timeout must be a nonnegative integer")
        self.clock = clock
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                key TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL,
                task TEXT NOT NULL, data TEXT NOT NULL, expires REAL
            );
            CREATE INDEX IF NOT EXISTS entries_partition ON entries(kind, scope, task);
            CREATE TABLE IF NOT EXISTS outcomes (
                cohort TEXT PRIMARY KEY, attempts INTEGER NOT NULL,
                accepted INTEGER NOT NULL, decision_ms REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY, data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT NOT NULL, encoder TEXT NOT NULL, text_hash TEXT NOT NULL,
                vector TEXT NOT NULL, PRIMARY KEY(key, encoder)
            );
        """)
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS entry_search USING fts5(key UNINDEXED, text)"
            )
            self.has_fts = True
        except sqlite3.OperationalError:
            self.has_fts = False
        self.connection.commit()

    def put(self, entry: Entry) -> None:
        if not _finite_number(entry.avoided_ms) or entry.avoided_ms < 0:
            raise ValueError("Avoided time must be finite and nonnegative")
        if entry.expires_at is not None and not _finite_number(entry.expires_at):
            raise ValueError("Expiry must be finite")
        data = snapshot({**entry.__dict__})
        search = str(entry.request.get("text", ""))
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.key,
                    entry.kind,
                    entry.scope,
                    entry.task,
                    json.dumps(data, ensure_ascii=False),
                    entry.expires_at,
                ),
            )
            if self.has_fts:
                self.connection.execute("DELETE FROM entry_search WHERE key = ?", (entry.key,))
                self.connection.execute(
                    "INSERT INTO entry_search VALUES (?, ?)", (entry.key, search)
                )

    @staticmethod
    def _entry(row: sqlite3.Row | None) -> Entry | None:
        if row is None:
            return None
        try:
            data = snapshot(json.loads(row["data"]))
            entry = Entry(**data)
            if (
                not isinstance(entry.request, dict)
                or not isinstance(entry.metadata, dict)
                or not isinstance(entry.dependencies, dict)
                or any(
                    not isinstance(k, str) or not isinstance(v, str)
                    for k, v in entry.dependencies.items()
                )
                or not _finite_number(entry.created_at)
                or not _finite_number(entry.avoided_ms)
                or entry.avoided_ms < 0
                or (entry.expires_at is not None and not _finite_number(entry.expires_at))
                or any(
                    not isinstance(getattr(entry, name), str) or not getattr(entry, name)
                    for name in (
                        "key",
                        "kind",
                        "scope",
                        "task",
                        "policy_version",
                        "output_contract",
                    )
                )
            ):
                return None
            return entry
        except (TypeError, ValueError, RecursionError):
            return None

    def get(self, key: str) -> Entry | None:
        return self._entry(
            self.connection.execute(
                "SELECT data FROM entries WHERE key = ? AND (expires IS NULL OR expires > ?)",
                (key, self.clock()),
            ).fetchone()
        )

    def candidates(
        self, *, kind: str, scope: str, task: str, text: str, limit: int = 5,
        eligible: Callable[[Entry], bool] | None = None
    ) -> list[Entry]:
        """Retrieve locally inside the eligible namespace; never scan via a model."""
        import re

        if limit <= 0:
            return []
        terms = re.findall(r"\w+", text, flags=re.UNICODE)[:32]
        if self.has_fts and terms:
            query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            rows = self.connection.execute(
                """
                SELECT e.data FROM entry_search s JOIN entries e ON e.key = s.key
                WHERE entry_search MATCH ? AND e.kind = ? AND e.scope = ? AND e.task = ?
                  AND (e.expires IS NULL OR e.expires > ?)
                ORDER BY bm25(entry_search)
            """,
                (query, kind, scope, task, self.clock()),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT data FROM entries WHERE kind = ? AND scope = ? AND task = ?
                  AND (expires IS NULL OR expires > ?) ORDER BY rowid DESC
            """,
                (kind, scope, task, self.clock()),
            ).fetchall()
        return [entry for row in rows if (entry := self._entry(row)) is not None
                and (eligible is None or eligible(entry))][:limit]

    def embedding_candidates(self, *, kind: str, scope: str, task: str, text: str,
                             embedder: Any, limit: int = 5,
                             eligible: Callable[[Entry], bool] | None = None) -> list[Entry]:
        """Cosine search over eligible live entries; vectors persist by encoder identity.

        Missing document vectors are batched on first semantic lookup. Exact hits
        never invoke the embedder. This local linear index targets small caches.
        """
        search = self.prepare_embedding_candidates(
            kind=kind, scope=scope, task=task, text=text,
            encoder_identity=embedder.cache_identity, limit=limit, eligible=eligible,
        )
        if search is None:
            return []
        return self.finish_embedding_candidates(search, embedder.embed(search.texts))

    def prepare_embedding_candidates(
        self, *, kind: str, scope: str, task: str, text: str,
        encoder_identity: Mapping[str, Any], limit: int = 5,
        eligible: Callable[[Entry], bool] | None = None,
    ) -> _EmbeddingSearch | None:
        """Read index state; callers may release their lock before inference."""
        from .models import fingerprint
        from .providers import normalized

        if limit <= 0:
            return None
        rows = self.connection.execute(
            "SELECT data FROM entries WHERE kind=? AND scope=? AND task=? "
            "AND (expires IS NULL OR expires > ?)", (kind, scope, task, self.clock())
        ).fetchall()
        entries = [entry for row in rows if (entry := self._entry(row)) is not None
                   and (eligible is None or eligible(entry))]
        if not entries:
            return None
        encoder = fingerprint(encoder_identity)
        vectors, missing = {}, []
        for entry in entries:
            text_hash = fingerprint(entry.request["text"])
            row = self.connection.execute(
                "SELECT vector FROM embeddings WHERE key=? AND encoder=? AND text_hash=?",
                (entry.key, encoder, text_hash)
            ).fetchone()
            if row:
                try:
                    vectors[entry.key] = normalized(json.loads(row["vector"]))
                    continue
                except (ValueError, TypeError):
                    pass
            missing.append(entry)
        texts = [text] + [entry.request["text"] for entry in missing]
        return _EmbeddingSearch(entries, vectors, missing, encoder, texts, limit)

    def finish_embedding_candidates(
        self, search: _EmbeddingSearch, encoded: Any,
    ) -> list[Entry]:
        """Validate and persist vectors after inference; discard changed entries."""
        from .models import fingerprint
        from .providers import normalized

        if not isinstance(encoded, list) or len(encoded) != len(search.texts):
            raise ValueError("Wrong embedding count")
        encoded = [normalized(vector) for vector in encoded]
        query = encoded[0]
        if any(len(vector) != len(query) for vector in encoded):
            raise ValueError("Embedding dimensions differ")
        entries = [entry for entry in search.entries if self.get(entry.key) == entry]
        live_keys = {entry.key for entry in entries}
        vectors = {key: vector for key, vector in search.vectors.items() if key in live_keys}
        with self.connection:
            for entry, vector in zip(search.missing, encoded[1:], strict=True):
                if entry.key not in live_keys:
                    continue
                vectors[entry.key] = vector
                self.connection.execute("INSERT OR REPLACE INTO embeddings VALUES(?,?,?,?)",
                    (entry.key, search.encoder, fingerprint(entry.request["text"]),
                     json.dumps(vector)))
        if any(len(vector) != len(query) for vector in vectors.values()):
            raise ValueError("Stored embedding dimensions differ; revise encoder identity")
        ranked = sorted(entries, key=lambda entry: (
            -sum(a * b for a, b in zip(query, vectors[entry.key], strict=True)), entry.key))
        return ranked[:search.limit]

    def observe_reuse(self, cohort: str, *, valid_hit: bool, decision_ms: float) -> None:
        """Record a verified or labelled outcome, never a model's claimed confidence."""
        if not isinstance(valid_hit, bool):
            raise TypeError("valid_hit must be an externally verified boolean")
        if not _finite_number(decision_ms) or decision_ms < 0:
            raise ValueError("Decision time must be finite and nonnegative")
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO outcomes VALUES (?, 1, ?, ?)
                ON CONFLICT(cohort) DO UPDATE SET
                    attempts = attempts + 1, accepted = accepted + excluded.accepted,
                    decision_ms = decision_ms + excluded.decision_ms
            """,
                (cohort, int(valid_hit), decision_ms),
            )

    def outcome(self, cohort: str) -> tuple[int, int, float]:
        row = self.connection.execute(
            "SELECT * FROM outcomes WHERE cohort = ?", (cohort,)
        ).fetchone()
        if row is None:
            return 0, 0, 0.0
        return row["attempts"], row["accepted"], row["decision_ms"] / row["attempts"]

    def record_trace(self, trace: Mapping[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO traces(data) VALUES (?)", (json.dumps(snapshot(dict(trace))),)
            )

    def traces(self) -> list[dict[str, Any]]:
        return [
            json.loads(row["data"])
            for row in self.connection.execute("SELECT data FROM traces ORDER BY id")
        ]

    def invalidate(self, *, scope: str, dependency: str | None = None) -> int:
        rows = self.connection.execute(
            "SELECT key, data FROM entries WHERE scope = ?", (scope,)
        ).fetchall()
        keys = [
            row["key"]
            for row in rows
            if dependency is None or dependency in json.loads(row["data"])["dependencies"]
        ]
        with self.connection:
            for key in keys:
                self.connection.execute("DELETE FROM entries WHERE key = ?", (key,))
                self.connection.execute("DELETE FROM embeddings WHERE key = ?", (key,))
                if self.has_fts:
                    self.connection.execute("DELETE FROM entry_search WHERE key = ?", (key,))
        return len(keys)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
