"""
SQLite-based LLM API cache for Phase 13 v2+.

Caches LLM responses to avoid repeated API calls and reduce cost.
Thread-safe for concurrent access.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class LLMCache:
    """Thread-safe SQLite cache for LLM API responses."""

    def __init__(self, cache_path: str):
        """
        Initialize cache.

        Args:
            cache_path: Path to SQLite database file
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.cache_path), check_same_thread=False)
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_responses (
                    request_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_responses(model)")
            conn.commit()

    def _compute_hash(self, model: str, prompt: str) -> str:
        """Compute cache key hash."""
        key = f"{model}|||{prompt}"
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, model: str, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response.

        Args:
            model: Model identifier
            prompt: Full prompt text

        Returns:
            Cached response dict or None if not found
        """
        request_hash = self._compute_hash(model, prompt)
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT response_json FROM llm_responses WHERE request_hash = ?",
            (request_hash,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, model: str, prompt: str, response: Dict[str, Any]):
        """
        Store response in cache.

        Args:
            model: Model identifier
            prompt: Full prompt text
            response: Response dict to cache
        """
        request_hash = self._compute_hash(model, prompt)
        response_json = json.dumps(response, ensure_ascii=False)

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO llm_responses (request_hash, model, prompt, response_json) VALUES (?, ?, ?, ?)",
                (request_hash, model, prompt, response_json)
            )
            conn.commit()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*), COUNT(DISTINCT model) FROM llm_responses")
        total, num_models = cursor.fetchone()
        return {
            "total_entries": total,
            "num_models": num_models,
            "cache_path": str(self.cache_path)
        }

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
