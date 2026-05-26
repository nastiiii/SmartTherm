"""SQLite persistence for FAQ, feedback, escalations, and conversation context."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import ROOT

DB_PATH = Path(ROOT / "data" / "assistant.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS faq_cards (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                tags TEXT DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                faq_id INTEGER,
                user_id INTEGER,
                chat_id INTEGER,
                query_text TEXT,
                helpful INTEGER NOT NULL,
                similarity REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (faq_id) REFERENCES faq_cards(id)
            );

            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                chat_id INTEGER,
                username TEXT,
                query_text TEXT,
                top_candidates TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_context (
                chat_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'idle',
                pending_query TEXT,
                candidate_faq_ids TEXT,
                last_faq_id INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS message_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                user_id INTEGER,
                username TEXT,
                direction TEXT NOT NULL,
                text TEXT,
                mode TEXT,
                faq_id INTEGER,
                similarity REAL,
                used_rag INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                user_id INTEGER,
                query_text TEXT NOT NULL,
                mode TEXT NOT NULL,
                best_faq_id INTEGER,
                best_similarity REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS faq_drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                tags TEXT DEFAULT '',
                source_escalation_id INTEGER,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dialog_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                user_query TEXT NOT NULL,
                bot_answer TEXT NOT NULL,
                tier TEXT,
                faq_id INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_dialog_turns_chat ON dialog_turns(chat_id, id DESC);
            """
        )


def list_faq(active_only: bool = True) -> list[dict[str, Any]]:
    sql = "SELECT * FROM faq_cards"
    if active_only:
        sql += " WHERE active = 1"
    sql += " ORDER BY id"
    with db() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_faq(faq_id: int) -> dict[str, Any] | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM faq_cards WHERE id = ?", (faq_id,)).fetchone()
    return dict(row) if row else None


def upsert_faq(
    faq_id: int | None,
    question: str,
    answer: str,
    tags: str = "",
    active: bool = True,
) -> int:
    now = _utc_now()
    with db() as conn:
        if faq_id is None:
            cur = conn.execute(
                """
                INSERT INTO faq_cards (question, answer, tags, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (question, answer, tags, int(active), now, now),
            )
            return int(cur.lastrowid)
        exists = conn.execute(
            "SELECT 1 FROM faq_cards WHERE id = ?", (faq_id,)
        ).fetchone()
        if exists:
            conn.execute(
                """
                UPDATE faq_cards
                SET question = ?, answer = ?, tags = ?, active = ?, updated_at = ?
                WHERE id = ?
                """,
                (question, answer, tags, int(active), now, faq_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO faq_cards (id, question, answer, tags, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (faq_id, question, answer, tags, int(active), now, now),
            )
        return faq_id


def delete_faq(faq_id: int, soft: bool = True) -> None:
    with db() as conn:
        if soft:
            conn.execute(
                "UPDATE faq_cards SET active = 0, updated_at = ? WHERE id = ?",
                (_utc_now(), faq_id),
            )
        else:
            conn.execute("DELETE FROM faq_cards WHERE id = ?", (faq_id,))


def import_faq_from_rows(rows: list[dict[str, Any]], replace: bool = False) -> int:
    if replace:
        with db() as conn:
            conn.execute("DELETE FROM feedback_events")
            conn.execute("DELETE FROM query_log")
            conn.execute("DELETE FROM faq_cards")
    count = 0
    for row in rows:
        upsert_faq(
            int(row["id"]) if row.get("id") else None,
            str(row["question"]),
            str(row["answer"]),
            str(row.get("tags") or ""),
            active=True,
        )
        count += 1
    return count


def export_faq_to_csv(path: Path | None = None) -> int:
    """Write active FAQ cards to CSV (wiki + bot single source of truth)."""
    import pandas as pd

    from app.knowledge_base import KB_CSV

    out = path or KB_CSV
    rows = list_faq(active_only=True)
    df = pd.DataFrame(
        [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "tags": r.get("tags") or "",
            }
            for r in rows
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    return len(df)


def save_feedback(
    faq_id: int | None,
    user_id: int,
    chat_id: int,
    query_text: str,
    helpful: bool,
    similarity: float | None = None,
) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO feedback_events
            (faq_id, user_id, chat_id, query_text, helpful, similarity, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (faq_id, user_id, chat_id, query_text, int(helpful), similarity, _utc_now()),
        )


def feedback_stats() -> dict[str, Any]:
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
        helpful = conn.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE helpful = 1"
        ).fetchone()[0]
        by_faq = conn.execute(
            """
            SELECT f.id, f.question,
                   SUM(CASE WHEN e.helpful = 1 THEN 1 ELSE 0 END) AS helped,
                   SUM(CASE WHEN e.helpful = 0 THEN 1 ELSE 0 END) AS not_helped
            FROM feedback_events e
            LEFT JOIN faq_cards f ON f.id = e.faq_id
            GROUP BY e.faq_id
            ORDER BY not_helped DESC
            LIMIT 20
            """
        ).fetchall()
    return {
        "total": total,
        "helpful": helpful,
        "not_helpful": total - helpful,
        "by_faq": [dict(r) for r in by_faq],
    }


def create_escalation(
    user_id: int,
    chat_id: int,
    username: str | None,
    query_text: str,
    top_candidates: list[dict[str, Any]],
) -> int:
    with db() as conn:
        cur = conn.execute(
            """
            INSERT INTO escalations
            (user_id, chat_id, username, query_text, top_candidates, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                chat_id,
                username or "",
                query_text,
                json.dumps(top_candidates, ensure_ascii=False),
                _utc_now(),
            ),
        )
        return int(cur.lastrowid)


def get_context(chat_id: int) -> dict[str, Any] | None:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM conversation_context WHERE chat_id = ?", (chat_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["candidate_faq_ids"] = json.loads(d["candidate_faq_ids"] or "[]")
    return d


def set_context(
    chat_id: int,
    user_id: int,
    state: str,
    pending_query: str | None = None,
    candidate_faq_ids: list[int] | None = None,
    last_faq_id: int | None = None,
) -> None:
    now = _utc_now()
    ids_json = json.dumps(candidate_faq_ids or [])
    with db() as conn:
        conn.execute(
            """
            INSERT INTO conversation_context
            (chat_id, user_id, state, pending_query, candidate_faq_ids, last_faq_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                user_id = excluded.user_id,
                state = excluded.state,
                pending_query = excluded.pending_query,
                candidate_faq_ids = excluded.candidate_faq_ids,
                last_faq_id = excluded.last_faq_id,
                updated_at = excluded.updated_at
            """,
            (chat_id, user_id, state, pending_query, ids_json, last_faq_id, now),
        )


def clear_context(chat_id: int) -> None:
    with db() as conn:
        conn.execute("DELETE FROM conversation_context WHERE chat_id = ?", (chat_id,))


def log_message(
    chat_id: int,
    user_id: int,
    username: str | None,
    direction: str,
    text: str,
    *,
    mode: str | None = None,
    faq_id: int | None = None,
    similarity: float | None = None,
    used_rag: bool = False,
) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO message_log
            (chat_id, user_id, username, direction, text, mode, faq_id, similarity, used_rag, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                user_id,
                username or "",
                direction,
                text[:4000],
                mode,
                faq_id,
                similarity,
                int(used_rag),
                _utc_now(),
            ),
        )


def log_query(
    chat_id: int,
    user_id: int,
    query_text: str,
    mode: str,
    best_faq_id: int | None,
    best_similarity: float | None,
) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO query_log
            (chat_id, user_id, query_text, mode, best_faq_id, best_similarity, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                user_id,
                query_text[:2000],
                mode,
                best_faq_id,
                best_similarity,
                _utc_now(),
            ),
        )


def dashboard_stats() -> dict[str, Any]:
    with db() as conn:
        faq_count = conn.execute(
            "SELECT COUNT(*) FROM faq_cards WHERE active = 1"
        ).fetchone()[0]
        fb = feedback_stats()
        queries = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        escalations_open = conn.execute(
            "SELECT COUNT(*) FROM escalations WHERE status = 'open'"
        ).fetchone()[0]
        mode_rows = conn.execute(
            """
            SELECT mode, COUNT(*) AS cnt FROM query_log
            GROUP BY mode ORDER BY cnt DESC
            """
        ).fetchall()
    return {
        "faq_active": faq_count,
        "feedback": fb,
        "queries_total": queries,
        "escalations_open": escalations_open,
        "queries_by_mode": [dict(r) for r in mode_rows],
    }


def list_escalations(status: str | None = "open", limit: int = 50) -> list[dict[str, Any]]:
    sql = "SELECT * FROM escalations"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with db() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["top_candidates"] = json.loads(d.get("top_candidates") or "[]")
        out.append(d)
    return out


def get_escalation(esc_id: int) -> dict[str, Any] | None:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM escalations WHERE id = ?", (esc_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["top_candidates"] = json.loads(d.get("top_candidates") or "[]")
    return d


def append_escalation_details(esc_id: int, extra_text: str) -> None:
    """Add user follow-up to an open escalation (same ticket)."""
    esc = get_escalation(esc_id)
    if not esc:
        return
    merged = (esc["query_text"] or "").strip()
    extra = (extra_text or "").strip()
    if extra and extra not in merged:
        merged = f"{merged}\n\n[уточнение] {extra}" if merged else extra
    with db() as conn:
        conn.execute(
            "UPDATE escalations SET query_text = ? WHERE id = ?",
            (merged[:4000], esc_id),
        )


def resolve_escalation(esc_id: int, status: str = "resolved") -> None:
    with db() as conn:
        conn.execute(
            "UPDATE escalations SET status = ? WHERE id = ?",
            (status, esc_id),
        )


def link_escalation_reply(esc_id: int, operator_user_id: int, reply_text: str) -> None:
    with db() as conn:
        conn.execute(
            """
            UPDATE escalations
            SET status = 'answered', top_candidates = ?
            WHERE id = ?
            """,
            (
                json.dumps(
                    {"operator_id": operator_user_id, "reply": reply_text[:2000]},
                    ensure_ascii=False,
                ),
                esc_id,
            ),
        )


def create_faq_draft(
    question: str,
    answer: str,
    tags: str = "",
    source_escalation_id: int | None = None,
) -> int:
    with db() as conn:
        cur = conn.execute(
            """
            INSERT INTO faq_drafts (question, answer, tags, source_escalation_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (question, answer, tags, source_escalation_id, _utc_now()),
        )
        return int(cur.lastrowid)


def list_faq_drafts(status: str = "draft", limit: int = 100) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM faq_drafts WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def push_dialog_turn(
    chat_id: int,
    user_id: int,
    user_query: str,
    bot_answer: str,
    *,
    tier: str | None = None,
    faq_id: int | None = None,
    keep_last: int = 6,
) -> None:
    """Append a turn and trim history per chat (keep only `keep_last` newest)."""
    with db() as conn:
        conn.execute(
            """
            INSERT INTO dialog_turns
            (chat_id, user_id, user_query, bot_answer, tier, faq_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                user_id,
                (user_query or "")[:2000],
                (bot_answer or "")[:2000],
                tier,
                faq_id,
                _utc_now(),
            ),
        )
        conn.execute(
            """
            DELETE FROM dialog_turns
            WHERE chat_id = ? AND id NOT IN (
                SELECT id FROM dialog_turns WHERE chat_id = ?
                ORDER BY id DESC LIMIT ?
            )
            """,
            (chat_id, chat_id, keep_last),
        )


def get_dialog_history(chat_id: int, limit: int = 3) -> list[dict[str, Any]]:
    """Return last `limit` turns, oldest → newest."""
    with db() as conn:
        rows = conn.execute(
            """
            SELECT user_query, bot_answer, tier, faq_id, created_at
            FROM dialog_turns
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        ).fetchall()
    items = [dict(r) for r in rows]
    items.reverse()
    return items


def clear_dialog_history(chat_id: int) -> None:
    with db() as conn:
        conn.execute("DELETE FROM dialog_turns WHERE chat_id = ?", (chat_id,))


def publish_faq_draft(draft_id: int) -> int:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM faq_drafts WHERE id = ?", (draft_id,)
        ).fetchone()
    if not row:
        raise ValueError("Draft not found")
    d = dict(row)
    new_id = upsert_faq(None, d["question"], d["answer"], d.get("tags") or "")
    with db() as conn:
        conn.execute(
            "UPDATE faq_drafts SET status = 'published' WHERE id = ?", (draft_id,)
        )
    return new_id
