"""
query_planner.py — Agentic query planner breaking queries into structured subtasks.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from config import config, get_gemini_api_key, get_openai_api_key

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """
You are an Agentic Query Planner for a Codebase RAG system.
Given a user query, your task is to break it down into one or more execution steps (tools to run).
Available tools:
1. "find_references" (argument: "symbol_name") — Find usages/references of a function or class.
2. "find_implementations" (argument: "class_name") — Find subclasses implementing an interface.
3. "find_inheritance" (argument: "class_name") — Find parent classes of a class.
4. "find_dependency_chains" (argument: "file_path") — Find module dependency import chains.
5. "ask_question" (argument: "query") — Perform a vector/semantic search over the codebase.

Respond ONLY with a JSON list of steps. Do not include any explanation or markdown code fences other than JSON.
Example query: "What classes inherit from BaseService and where are they used?"
Steps:
[
  {"tool": "find_implementations", "class_name": "BaseService"},
  {"tool": "ask_question", "query": "How is BaseService implemented and used?"}
]
"""


class AgenticQueryPlanner:
    """Decomposes a repository query into steps, runs tools, and merges results."""

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

    def create_plan(self, query: str) -> List[Dict[str, Any]]:
        """Ask LLM to decompose the query into structured steps."""
        if self._use_gemini:
            prompt = f"{PLANNER_SYSTEM_PROMPT}\n\nQuery: {query}"
            try:
                response = self._gemini_model.generate_content(prompt)
                text = response.text.strip() if response and response.text else "[]"
                # Strip markdown code blocks if any
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
                return json.loads(text)
            except Exception as exc:
                logger.warning("Gemini plan generation failed, falling back to ask_question: %s", exc)
        else:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Query: {query}"},
                    ],
                )
                text = response.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
                return json.loads(text)
            except Exception as exc:
                logger.warning("OpenAI plan generation failed, falling back to ask_question: %s", exc)
        
        # Fallback plan
        return [{"tool": "ask_question", "query": query}]

    def execute_plan(self, plan: List[Dict[str, Any]], conversation_id: str | None = None) -> Dict[str, Any]:
        """Execute plan steps, gather details, and return synthesized answer."""
        context_parts = []
        for i, step in enumerate(plan, 1):
            tool = step.get("tool")
            try:
                if tool == "find_references":
                    symbol = step.get("symbol_name")
                    res = self.analyzer.find_references(symbol)
                    context_parts.append(f"Step {i} (find_references for '{symbol}'): {res}")
                elif tool == "find_implementations":
                    cls = step.get("class_name")
                    res = self.analyzer.find_implementations(cls)
                    context_parts.append(f"Step {i} (find_implementations for '{cls}'): {res}")
                elif tool == "find_inheritance":
                    cls = step.get("class_name")
                    res = self.analyzer.find_inheritance(cls)
                    context_parts.append(f"Step {i} (find_inheritance for '{cls}'): {res}")
                elif tool == "find_dependency_chains":
                    path = step.get("file_path")
                    res = self.analyzer.find_dependency_chains(path)
                    context_parts.append(f"Step {i} (find_dependency_chains for '{path}'): {res}")
                else:
                    q = step.get("query")
                    res = self.analyzer.ask_question(q, conversation_id)
                    context_parts.append(f"Step {i} (ask_question for '{q}'): {res.get('answer')}")
            except Exception as exc:
                logger.warning("Error executing planning step %s: %s", step, exc)

        merged_context = "\n".join(context_parts)
        
        # Synthesize final answer using the answer generator
        final_prompt = (
            f"An agentic planner executed the following steps to resolve the query:\n"
            f"{merged_context}\n\n"
            f"Please synthesize a clear, coherent, and detailed final answer to: "
        )
        return self.analyzer.ask_question(final_prompt, conversation_id)
