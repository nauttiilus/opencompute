import sqlite3
import json
from datetime import datetime
from typing import Optional

DB_PATH = "rental_database.db"


def init_rental_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS rentals (
            uid INTEGER PRIMARY KEY,
            hotkey TEXT NOT NULL,
            rented_by TEXT,
            public_key TEXT NOT NULL,
            ssh_key TEXT,
            details TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def add_rental(
    uid: int,
    hotkey: str,
    public_key: str,
    ssh_key: Optional[str],
    details: dict,
    rented_by: Optional[str] = None,
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO rentals (uid, hotkey, rented_by, public_key, ssh_key, details, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        uid,
        hotkey,
        rented_by,
        public_key,
        ssh_key,
        json.dumps(details),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


def get_rental(uid: int) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM rentals WHERE uid = ?', (uid,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "uid": row[0],
            "hotkey": row[1],
            "rented_by": row[2],
            "public_key": row[3],
            "ssh_key": row[4],
            "details": json.loads(row[5]) if row[5] else {},
            "created_at": row[6]
        }
    return None


def remove_rental(uid: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM rentals WHERE uid = ?', (uid,))
    conn.commit()
    conn.close()


def list_all_rentals() -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM rentals')
    rows = c.fetchall()
    conn.close()
    return [
        {
            "uid": row[0],
            "hotkey": row[1],
            "rented_by": row[2],
            "public_key": row[3],
            "ssh_key": row[4],
            "details": json.loads(row[5]) if row[5] else {},
            "created_at": row[6]
        }
        for row in rows
    ]
