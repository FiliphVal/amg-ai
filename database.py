import sqlite3
import uuid
from datetime import datetime

DB_FILE = "amg_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    ''')
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT session_id, title FROM chat_sessions ORDER BY created_at DESC")
    sessions = c.fetchall()
    conn.close()
    return sessions

def save_message_to_db(session_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, str(content), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def create_new_session():
    new_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chat_sessions (session_id, title, created_at) VALUES (?, ?, ?)",
              (new_id, "Ny chatt", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return new_id

def get_messages_for_session(session_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    messages = c.fetchall()
    conn.close()
    return messages

def delete_session(session_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def update_session_title(session_id, new_title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ?", (new_title, session_id))
    conn.commit()
    conn.close()