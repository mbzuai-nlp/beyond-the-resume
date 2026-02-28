# llm_response_cache.py
import json
import hashlib
import sqlite3
import threading
import pickle
from typing import Optional, Any
from openai import OpenAI

_db_path = ".llm_response_cache.sqlite"
_tls = threading.local()
_db_lock = threading.Lock()

def _get_conn() -> sqlite3.Connection:
    conn = getattr(_tls, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        # v is now BLOB
        conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        conn.commit()
        _tls.conn = conn
    return conn

def key_text(**kwargs) -> str:
    s = json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cache_get_obj(k: str) -> Optional[Any]:
    row = _get_conn().execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
    if not row:
        return None
    blob = row[0]
    # IMPORTANT: only safe if cache is trusted (not writable by attackers)
    return pickle.loads(blob)

def cache_set_obj(k: str, obj: Any) -> None:
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    with _db_lock:
        conn = _get_conn()
        conn.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (k, sqlite3.Binary(blob)))
        conn.commit()

def with_responses_cache(client: OpenAI) -> OpenAI:
    orig_create = client.responses.create

    def cached_create(*args, **kwargs):
        k = key_text(args=args, **kwargs)

        cached = cache_get_obj(k)
        if cached is not None:
            return cached

        resp = orig_create(*args, **kwargs)
        cache_set_obj(k, resp)
        return resp

    client.responses.create = cached_create
    return client
