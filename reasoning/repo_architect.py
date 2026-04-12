"""
repo_architect.py — Generates comprehensive architecture reports and refactoring plans.
"""

import logging
from typing import Any, Dict
import networkx as nx

from config import config, get_gemini_api_key, get_openai_api_key

logger = logging.getLogger(__name__)

ARCHITECT_PROMPT_TEMPLATE = """
You are a principal software architect.
Analyze the following structural overview of a software repository and write a comprehensive Architecture Report:
1. High-level Architectural Style (MVC, layered, microservices, monolithic).
2. Key modules/files, dependency hotspots, and coupling patterns.
3. Design flaws, circular dependencies, technical debt, or potential bottlenecks.
4. Actionable refactoring and design recommendations to improve scalability, modularity, and testability.

Repository Overview Details:
- Most Connected Modules: {most_connected}
- Circular Dependencies Found: {circular_dependencies}
- Directory Structure Outline:
{directory_tree}

Existing Architecture Summary:
{arch_summary}
"""


class AIRepoArchitect:
    """Performs global design review and technical debt assessments."""

    def __init__(self, analyzer: Any) -> None:
        self.analyzer = analyzer
        self._use_gemini = config.llm_provider == "gemini"
        self.model = config.gemini_llm_model if self._use_gemini else config.llm_model
        if self._use_gemini:
            import google.generativeai as genai
            genai.configure(api_key=get_gemini_api_key())
            self._gemini_model = genai.GenerativeModel(self.model)
        else:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=get_openai_api_key())

    def generate_architecture_report(self) -> str:
        """Run architect analysis on the repository structure."""
        # Find circular dependencies
        cycles = []
        try:
            if hasattr(self.analyzer, "_kg") and hasattr(self.analyzer._kg, "graph"):
                g = self.analyzer._kg.graph
                cycles = list(nx.simple_cycles(g))
        except Exception as exc:
            logger.warning("Could not calculate simple cycles: %s", exc)

        most_connected = self.analyzer._kg.get_most_connected(top_k=5)
        dir_tree = self.analyzer._directory_tree
        arch_summary = self.analyzer.get_architecture_summary().get("summary", "No summary.")

        prompt = ARCHITECT_PROMPT_TEMPLATE.format(
            most_connected=str(most_connected),
            circular_dependencies=str(cycles[:10]),
            directory_tree=dir_tree[:4000],  # truncate if very large
            arch_summary=arch_summary,
        )

        if self._use_gemini:
            try:
                response = self._gemini_model.generate_content(prompt)
                return response.text.strip() if response and response.text else "No architecture report generated."
            except Exception as exc:
                logger.warning("Gemini architect report failed: %s", exc)
                return "Failed to generate architecture report."
        else:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": "You are a principal software architect advisor."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("OpenAI architect report failed: %s", exc)
                return "Failed to generate architecture report."
