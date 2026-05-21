"""
database.py — SQLite cache, API usage counters, DB initialization.
"""

import sqlite3
import os
import json
from datetime import datetime, date

DB_PATH = "data/trades.db"


def _conn():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    os.makedirs("data", exist_ok=True)
    con = _conn()
    cur = con.cursor()

    # API usage counters
    cur.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            service     TEXT PRIMARY KEY,
            count       INTEGER DEFAULT 0,
            reset_date  TEXT
        )
    """)

    # Generic key/value cache
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key   TEXT PRIMARY KEY,
            payload     TEXT,
            cached_at   TEXT
        )
    """)

    # Seed service rows if missing
    for svc in ("alpha_vantage", "finnhub", "openai"):
        cur.execute("""
            INSERT OR IGNORE INTO api_usage (service, count, reset_date)
            VALUES (?, 0, ?)
        """, (svc, str(date.today())))

    con.commit()
    con.close()


# ─── Usage helpers ────────────────────────────────────────────────────────────

def _reset_if_new_day(cur, service: str):
    cur.execute("SELECT reset_date FROM api_usage WHERE service=?", (service,))
    row = cur.fetchone()
    today = str(date.today())
    if row and row[0] != today:
        cur.execute(
            "UPDATE api_usage SET count=0, reset_date=? WHERE service=?",
            (today, service)
        )


def get_usage(service: str) -> int:
    con = _conn()
    cur = con.cursor()
    _reset_if_new_day(cur, service)
    cur.execute("SELECT count FROM api_usage WHERE service=?", (service,))
    row = cur.fetchone()
    con.commit()
    con.close()
    return row[0] if row else 0


def increment_usage(service: str, amount: int = 1) -> int:
    """Increment counter and return new total. Call BEFORE making the request."""
    con = _conn()
    cur = con.cursor()
    _reset_if_new_day(cur, service)
    cur.execute(
        "UPDATE api_usage SET count = count + ? WHERE service=?",
        (amount, service)
    )
    cur.execute("SELECT count FROM api_usage WHERE service=?", (service,))
    row = cur.fetchone()
    con.commit()
    con.close()
    return row[0] if row else amount


def get_remaining(service: str, daily_limit: int) -> int:
    return max(0, daily_limit - get_usage(service))


# ─── Generic cache helpers ─────────────────────────────────────────────────────

def cache_set(key: str, data) -> None:
    payload = json.dumps(data)
    con = _conn()
    con.execute("""
        INSERT OR REPLACE INTO cache (cache_key, payload, cached_at)
        VALUES (?, ?, ?)
    """, (key, payload, datetime.utcnow().isoformat()))
    con.commit()
    con.close()


def cache_get(key: str, max_age_seconds: int = 1800):
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT payload, cached_at FROM cache WHERE cache_key=?", (key,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    payload, cached_at = row
    age = (datetime.utcnow() - datetime.fromisoformat(cached_at)).total_seconds()
    if age > max_age_seconds:
        return None
    return json.loads(payload)
