"""
code_reviewer.py — Automatically review code files for bugs, design smells, performance, and security issues.
"""

import logging
from typing import Any

from config import config, get_gemini_api_key, get_openai_api_key

logger = logging.getLogger(__name__)

CODE_REVIEW_PROMPT_TEMPLATE = """
You are an expert code reviewer and security auditor.
Analyze the following code from the file '{file_path}' and provide a structured review focusing on:
1. Bugs or correctness issues.
2. Code smells, readability, or formatting issues.
3. Performance bottlenecks or memory issues.
4. Security vulnerabilities (OWASP, injection, data leakage).
5. Architecture compliance or design improvements.

Structure your response clearly with markdown sections for each category and suggest specific code refactoring tips.

Code content to review:
{code}
"""


class AICodeReviewer:
    """Orchestrates automated AI-driven reviews on code files."""

    def __init__(self, analyzer: Any = None) -> None:
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

    def review_code(self, code: str, file_path: str = "unknown") -> str:
        """Run AI code review on code content."""
        prompt = CODE_REVIEW_PROMPT_TEMPLATE.format(
            file_path=file_path,
            code=code,
        )

        if self._use_gemini:
            try:
                response = self._gemini_model.generate_content(prompt)
                return response.text.strip() if response and response.text else "No suggestions."
            except Exception as exc:
                logger.warning("Gemini code review failed: %s", exc)
                return "Failed to run code review."
        else:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a senior software reviewer."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("OpenAI code review failed: %s", exc)
                return "Failed to run code review."
