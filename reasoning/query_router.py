"""query_router.py — Intent-based routing for repository queries.

The QueryRouter takes a natural-language query and decides which
RepoAnalyzer capability should handle it (architecture, function usage,
file dependencies, file explanation, repo overview, or generic code question).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from reasoning.repo_analyzer import RepoAnalyzer

QueryCategory = Literal[
    "architecture",
    "function_usage",
    "file_dependencies",
    "file_explanation",
    "repo_overview",
    "code_question",
]


class QueryRouter:
    """Route user queries to the appropriate RepoAnalyzer capability."""

    def __init__(self, analyzer: RepoAnalyzer) -> None:
        """Initialize the router with a RepoAnalyzer instance."""
        self.analyzer = analyzer

    def classify_query(self, query: str) -> QueryCategory:
        """Classify a natural-language query into a high-level category."""
        q = query.strip().lower()

        if any(
            p in q
            for p in (
                "repo overview",
                "repository overview",
                "give me an overview",
                "summarize this repo",
            )
        ):
            return "repo_overview"

        if any(
            p in q
            for p in (
                "architecture summary",
                "system overview",
                "what does this repository do",
                "what does this repo do",
                "overall architecture",
            )
        ):
            return "architecture"

        if ("where is" in q or "where's" in q or "who calls" in q) and "used" in q:
            return "function_usage"

        if any(
            p in q
            for p in (
                "what depends on",
                "which files import",
                "what files import",
                "who imports",
                "file dependencies",
                "dependencies of",
            )
        ):
            return "file_dependencies"

        if any(
            p in q
            for p in (
                "explain file",
                "describe file",
                "summarize file",
                "what does file",
            )
        ):
            return "file_explanation"

        return "code_question"

    def _extract_function_name(self, query: str) -> str | None:
        """Extract a function name from a function_usage-style query."""
        q = query.strip().rstrip("?")
        m = re.search(r"where\s+is\s+(.+?)\s+used", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("`'\"")
        m = re.search(r"who\s+calls\s+(.+)", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("`'\"")
        m = re.search(r"([\w\.]+)\s+used", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("`'\"")
        return None

    def _extract_file_path(self, query: str) -> str | None:
        """Extract a file path from queries referencing specific files."""
        q = query.strip().rstrip("?")
        patterns = [
            r"explain\s+file\s+(.+)",
            r"describe\s+file\s+(.+)",
            r"summarize\s+file\s+(.+)",
            r"what\s+does\s+file\s+(.+)",
            r"what\s+files\s+import\s+(.+)",
            r"which\s+files\s+import\s+(.+)",
            r"who\s+imports\s+(.+)",
            r"what\s+depends\s+on\s+(.+)",
            r"dependencies\s+of\s+(.+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, q, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip().strip("`'\"")
                candidate = candidate.split()[0].rstrip(",.;")
                return candidate
        m = re.search(r"([\w\-/\.]+\.py)", q)
        if m:
            return m.group(1).strip().strip("`'\"")
        return None

    def route_query(self, query: str) -> dict[str, Any]:
        """Route a query to the appropriate RepoAnalyzer method."""
        category = self.classify_query(query)

        if category == "function_usage":
            function_name = self._extract_function_name(query)
            if function_name:
                return self.analyzer.find_function_usage(function_name)
            return self.analyzer.ask_question(query)

        if category == "file_dependencies":
            file_path = self._extract_file_path(query)
            if file_path:
                deps = self.analyzer.get_file_dependencies(file_path)
                return {"file": file_path, "dependencies": deps}
            return self.analyzer.ask_question(query)

        if category == "file_explanation":
            file_path = self._extract_file_path(query)
            if file_path:
                return self.analyzer.explain_file(file_path)
            return self.analyzer.ask_question(query)

        if category == "architecture":
            return self.analyzer.get_architecture_summary()

        if category == "repo_overview":
            return self.analyzer.get_repo_overview()

        return self.analyzer.ask_question(query)
