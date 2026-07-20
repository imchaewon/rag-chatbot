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


def get_sessions() -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT
                ch.session_id,
                COALESCE(st.title,
                    (SELECT content FROM chat_history
                     WHERE session_id = ch.session_id AND role = 'human'
                     ORDER BY id ASC LIMIT 1)) as title,
                MAX(ch.created_at) as last_active
            FROM chat_history ch
            LEFT JOIN session_titles st ON st.session_id = ch.session_id
            GROUP BY ch.session_id
            ORDER BY last_active DESC
        """).fetchall()
    return [{"session_id": r[0], "title": r[1] or "새 대화", "last_active": r[2]} for r in rows]


def get_full_history(session_id: str) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


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
