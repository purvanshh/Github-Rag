"""
explanation_engine.py — Multi-difficulty code explanation engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from config import config, get_gemini_api_key, get_openai_api_key

logger = logging.getLogger(__name__)

EXPLAIN_PROMPT_TEMPLATE = """
You are an expert codebase analysis assistant.
Explain the following code content at the '{level}' level.

Explanation Level Guidelines:
- beginner: Avoid technical jargon. Use analogies and simple conceptual language. Focus on the 'why' and 'what'.
- medium: Normal technical overview. Detail functions, parameters, classes, and interactions.
- advanced: Code complexity analysis, potential bottlenecks, memory footprints, Big-O scaling, AST structures, and design patterns.

Code to explain:
{code}
"""


class CodeExplanationEngine:
    """Generates beginner, medium, and advanced level explanations for symbols and files."""

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

    def explain(self, code_content: str, level: str = "medium") -> str:
        """Explain raw code content at a specific difficulty level."""
        prompt = EXPLAIN_PROMPT_TEMPLATE.format(
            level=level,
            code=code_content,
        )

        if self._use_gemini:
            try:
                response = self._gemini_model.generate_content(prompt)
                return response.text.strip() if response and response.text else ""
            except Exception as exc:
                logger.warning("Gemini explanation failed: %s", exc)
                return "Failed to generate explanation."
        else:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": "You are a senior software developer explaining codebase details."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("OpenAI explanation failed: %s", exc)
                return "Failed to generate explanation."
