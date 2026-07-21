import sqlite3
from langchain_core.messages import HumanMessage, AIMessage

DB_PATH = "chat_history.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_titles (
                session_id TEXT PRIMARY KEY,
                title      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                question   TEXT    NOT NULL,
                answer     TEXT    NOT NULL,
                rating     INTEGER NOT NULL CHECK(rating IN (1, -1)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_pins (
                session_id TEXT PRIMARY KEY
            )
        """)


def get_history(session_id: str) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    history = []
    for role, content in rows:
        if role == "human":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content))
    return history


def save_messages(session_id: str, question: str, answer: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, "human", question), (session_id, "ai", answer)],
        )


def delete_last_pair(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT 2",
            (session_id,),
        ).fetchall()
        if len(rows) == 2:
            ids = [r[0] for r in rows]
            conn.execute(
                f"DELETE FROM chat_history WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )


def clear_history(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))


def delete_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_titles WHERE session_id = ?", (session_id,))


def save_session_title(session_id: str, title: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO session_titles (session_id, title) VALUES (?, ?)",
            (session_id, title),
        )


def pin_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR IGNORE INTO session_pins (session_id) VALUES (?)", (session_id,))


def unpin_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM session_pins WHERE session_id = ?", (session_id,))


def get_sessions() -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT
                ch.session_id,
                COALESCE(st.title,
                    (SELECT content FROM chat_history
                     WHERE session_id = ch.session_id AND role = 'human'
                     ORDER BY id ASC LIMIT 1)) as title,
                MAX(ch.created_at) as last_active,
                CASE WHEN sp.session_id IS NOT NULL THEN 1 ELSE 0 END as pinned
            FROM chat_history ch
            LEFT JOIN session_titles st ON st.session_id = ch.session_id
            LEFT JOIN session_pins sp ON sp.session_id = ch.session_id
            GROUP BY ch.session_id
            ORDER BY pinned DESC, last_active DESC
        """).fetchall()
    return [{"session_id": r[0], "title": r[1] or "새 대화", "last_active": r[2], "pinned": bool(r[3])} for r in rows]


def get_full_history(session_id: str) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def save_feedback(session_id: str, question: str, answer: str, rating: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO feedback (session_id, question, answer, rating) VALUES (?, ?, ?, ?)",
            (session_id, question, answer, rating),
        )


def get_feedback_stats() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative
            FROM feedback
        """).fetchone()
        recent = conn.execute("""
            SELECT question, answer, rating, created_at
            FROM feedback ORDER BY id DESC LIMIT 20
        """).fetchall()
    total, positive, negative = row
    return {
        "total": total or 0,
        "positive": positive or 0,
        "negative": negative or 0,
        "recent": [
            {"question": r[0], "answer": r[1], "rating": r[2], "created_at": r[3]}
            for r in recent
        ],
    }


def get_question_stats(limit: int = 20) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT h1.content, COUNT(*) as count
            FROM chat_history h1
            JOIN chat_history h2 ON h2.id = h1.id + 1 AND h2.role = 'ai'
            WHERE h1.role = 'human'
              AND h2.content NOT LIKE '%MSP 운영과 관련 없는 질문%'
              AND h2.content NOT LIKE '%매뉴얼에서 확인이 어렵습니다%'
            GROUP BY h1.content
            ORDER BY count DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [{"question": row[0], "count": row[1]} for row in rows]
