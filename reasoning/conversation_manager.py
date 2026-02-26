"""
conversation_manager.py — Local multi-turn conversation history persistence.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Any

from config import config

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages chat histories for multi-turn conversations."""

    def __init__(self, repo_name: str) -> None:
        self.repo_name = repo_name
        self.conversations_dir = os.path.join(config.repos_dir, repo_name, "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)

    def _get_path(self, conversation_id: str) -> str:
        return os.path.join(self.conversations_dir, f"{conversation_id}.json")

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get the message history list for a conversation ID."""
        path = self._get_path(conversation_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to load conversation history: %s", exc)
        return []

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to the history file."""
        history = self.get_history(conversation_id)
        history.append({"role": role, "content": content})
        path = self._get_path(conversation_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to save conversation message: %s", exc)
