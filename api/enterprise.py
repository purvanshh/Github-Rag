"""
enterprise.py — Enterprise capabilities including RBAC, audit logging, and collections.
"""

import os
import sqlite3
import datetime
import logging
from typing import List, Dict, Any
from fastapi import HTTPException

from config import config

logger = logging.getLogger(__name__)

# RBAC mappings
USER_ROLES = {
    "admin-key": "admin",
    "dev-key": "developer",
    "viewer-key": "viewer"
}

# Repository collections
COLLECTIONS: Dict[str, List[str]] = {
    "default-collection": ["Github-Rag"],
}

DB_PATH = os.path.join(config.repos_dir, "enterprise.db")


def init_enterprise_db() -> None:
    """Initialize audit logging SQLite database."""
    os.makedirs(config.repos_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_audit_action(user: str, action: str, details: str) -> None:
    """Log an event to the enterprise audit log table."""
    init_enterprise_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        timestamp = datetime.datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO audit_logs (timestamp, user, action, details) VALUES (?, ?, ?, ?)",
            (timestamp, user, action, details)
        )
        conn.commit()
    except Exception as exc:
        logger.error("Failed to write to audit log: %s", exc)
    finally:
        conn.close()


def get_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent enterprise audit logs."""
    init_enterprise_db()
    conn = sqlite3.connect(DB_PATH)
    logs = []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, user, action, details FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
        for row in cursor.fetchall():
            logs.append({
                "timestamp": row[0],
                "user": row[1],
                "action": row[2],
                "details": row[3]
            })
    finally:
        conn.close()
    return logs


def verify_role_access(authorization: str | None, allowed_roles: List[str]) -> str:
    """Verify if the API key matches one of the allowed RBAC roles."""
    if not config.security_enabled:
        return "admin"
        
    if not authorization:
        raise HTTPException(status_code=401, detail="API Key missing.")
        
    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]
        
    role = USER_ROLES.get(token)
    if not role:
        raise HTTPException(status_code=403, detail="Invalid API Key or authorization role.")
        
    if role not in allowed_roles:
        raise HTTPException(status_code=403, detail=f"Role '{role}' is not allowed to access this resource.")
        
    return role
