import os
import json
import sqlite3


def update_allocation_db(hotkey: str, info: dict, flag: bool):
    """
    Update the SN27 validator allocation database.

    Args:
        hotkey: The miner's hotkey
        info: Allocation details (dict)
        flag: True to add/update allocation, False to remove
    """
    db_path = os.getenv("SN27_DB_PATH", "/root/SN27/database.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    try:
        if flag:
            cursor.execute("""
                INSERT INTO allocation (hotkey, details)
                VALUES (?, ?) ON CONFLICT(hotkey) DO UPDATE SET
                details=excluded.details
            """, (hotkey, json.dumps(info)))
        else:
            cursor.execute("DELETE FROM allocation WHERE hotkey = ?", (hotkey,))
        conn.commit()
        print(f"[DEBUG] Updated SN27 allocation DB for hotkey {hotkey}, flag={flag}")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to update SN27 allocation DB: {e}")
    finally:
        cursor.close()
        conn.close()
