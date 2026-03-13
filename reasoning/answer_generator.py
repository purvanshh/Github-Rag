"""answer_generator.py — Generate LLM-powered answers about the codebase.

Combines retrieved code context with prompt templates and sends
to GPT-4 / GPT-4o for reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from reasoning.prompt_templates import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE


def normalize_sources(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize raw retrieval results into UI-friendly source metadata.

    Converts internal metadata keys:
        file_path, symbol_name, symbol_type, start_line, end_line
    into the format expected by the UI:
        file, symbol, type, lines (e.g. "10-42").
    """
    sources: List[Dict[str, str]] = []
    for result in results:
        meta = result.get("metadata") or {}
        file_path = meta.get("file_path")
        symbol_name = meta.get("symbol_name")
        symbol_type = meta.get("symbol_type")
        start_line = meta.get("start_line")
        end_line = meta.get("end_line")

        if (
            not file_path
            or symbol_name is None
            or symbol_type is None
            or start_line is None
            or end_line is None
        ):
            continue

        sources.append(
            {
                "file": str(file_path),
                "symbol": str(symbol_name),
                "type": str(symbol_type),
                "lines": f"{start_line}-{end_line}",
            }
        )
    return sources


class AnswerGenerator:
    """Generates answers about the codebase using an LLM."""

    def __init__(
        self,
        retriever: Any,
        model: str = "gpt-4o",
        temperature: float = 0.1,
    ):
        self.retriever = retriever
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Generate an answer to a codebase question.

        Args:
            question: Natural language question about the codebase.

        Returns:
            Dict with 'answer', 'sources', and 'model' used.
        """
        # Retrieve relevant context
        context = self.retriever.retrieve_with_context(question)
        raw_results = self.retriever.retrieve(question)

        # Build prompt
        user_prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content if response.choices else ""

        # Extract source citations in a normalized format
        sources = normalize_sources(raw_results)

        return {
            "answer": answer,
            "sources": sources,
            "model": self.model,
        }

