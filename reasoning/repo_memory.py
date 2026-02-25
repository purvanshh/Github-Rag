"""
repo_memory.py — Persistent Repository Memory Engine.
Saves, loads, and manages repository stack profiles, architecture cached layouts,
and common FAQ context to provide historical and domain context to the LLM.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List

from config import config

logger = logging.getLogger(__name__)


class RepositoryMemory:
    """Manages persistent repository memories (profiles, architecture summaries, FAQs)."""

    def __init__(self, repo_name: str) -> None:
        self.repo_name = repo_name
        self.memory_dir = os.path.join(config.repos_dir, repo_name)
        os.makedirs(self.memory_dir, exist_ok=True)
        
        self.profile_path = os.path.join(self.memory_dir, "repo_profile.json")
        self.faq_path = os.path.join(self.memory_dir, "repo_faq.json")
        
        self.profile: Dict[str, Any] = self._load_json(self.profile_path, {
            "tech_stack": [],
            "primary_language": "Unknown",
            "architecture_style": "Monolith",
            "description": "No repository description provided.",
            "design_patterns": []
        })
        
        self.faqs: List[Dict[str, str]] = self._load_json(self.faq_path, [
            {
                "question": "How do I run the test suite?",
                "answer": "Run `pytest` using the command `PYTHONPATH=. ./venv/bin/pytest tests/` in the project root."
            },
            {
                "question": "How do I perform a full repository ingestion?",
                "answer": "Instantiate the `RepoIngestionPipeline` and call `ingest_repository(repo_url)`."
            }
        ])

    def _load_json(self, path: str, default: Any) -> Any:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to load memory file %s: %s", path, exc)
        return default

    def save_memory(self) -> None:
        """Persist profile and FAQs back to disk."""
        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2)
            with open(self.faq_path, "w", encoding="utf-8") as f:
                json.dump(self.faqs, f, indent=2)
            logger.info("Saved repository memory state for %s", self.repo_name)
        except OSError as exc:
            logger.warning("Failed to save repository memory: %s", exc)

    def add_faq(self, question: str, answer: str) -> None:
        """Add a frequently asked question to the repository memory."""
        self.faqs.append({"question": question, "answer": answer})
        self.save_memory()

    def update_profile(self, updates: Dict[str, Any]) -> None:
        """Update fields inside the repository stack profile."""
        self.profile.update(updates)
        self.save_memory()

    def get_memory_context(self) -> str:
        """Generate a formatted markdown string representing the repository memory."""
        lines = []
        lines.append("## Repository Stack Profile")
        lines.append(f"- **Description**: {self.profile.get('description')}")
        lines.append(f"- **Primary Language**: {self.profile.get('primary_language')}")
        lines.append(f"- **Architecture Style**: {self.profile.get('architecture_style')}")
        
        tech_stack = self.profile.get("tech_stack", [])
        if tech_stack:
            lines.append(f"- **Tech Stack**: {', '.join(tech_stack)}")
            
        patterns = self.profile.get("design_patterns", [])
        if patterns:
            lines.append(f"- **Design Patterns**: {', '.join(patterns)}")

        lines.append("\n## Frequently Asked Questions (FAQ)")
        for idx, faq in enumerate(self.faqs, 1):
            lines.append(f"### Q{idx}: {faq['question']}")
            lines.append(f"**A**: {faq['answer']}")
            
        return "\n".join(lines)
