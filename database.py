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
