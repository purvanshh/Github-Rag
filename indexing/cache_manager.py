"""
cache_manager.py — Local SQLite cache manager for query, embedding, and graph caching.
"""

from __future__ import annotations

import os
import sqlite3
import json
import logging
from typing import Any, Dict, Optional

from config import config

logger = logging.getLogger(__name__)


class LocalCacheManager:
    """Zero-dependency local caching manager using SQLite."""

    def __init__(self, repo_name: str) -> None:
        self.db_path = os.path.join(config.repos_dir, repo_name, "cache.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        category TEXT,
                        key TEXT,
                        value TEXT,
                        PRIMARY KEY (category, key)
                    )
                """)
        except sqlite3.Error as exc:
            logger.warning("Failed to initialize SQLite cache database: %s", exc)

    def get(self, category: str, key: str) -> Optional[Any]:
        """Fetch a value from the cache by category and key."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT value FROM cache WHERE category=? AND key=?", (category, key))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as exc:
            logger.warning("Cache get error: %s", exc)
        return None

    def set(self, category: str, key: str, value: Any) -> None:
        """Insert or replace a value in the cache."""
        try:
            val_str = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO cache (category, key, value) VALUES (?, ?, ?)", (category, key, val_str))
        except Exception as exc:
            logger.warning("Cache set error: %s", exc)
