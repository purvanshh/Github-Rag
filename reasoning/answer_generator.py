"""
answer_generator.py — Generate LLM-powered answers about the codebase.

Combines retrieved code context with prompt templates and sends
to GPT-4 / GPT-4o for reasoning.
"""

from openai import OpenAI

from reasoning.prompt_templates import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE
from retrieval.retriever import CodeRetriever


class AnswerGenerator:
    """Generates answers about the codebase using an LLM."""

    def __init__(
        self,
        retriever: CodeRetriever,
        model: str = "gpt-4o",
        temperature: float = 0.1,
    ):
        self.retriever = retriever
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def generate_answer(self, question: str) -> dict:
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
        
        answer = response.choices[0].message.content
        
        # Extract source citations
        sources = []
        for result in raw_results:
            meta = result["metadata"]
            sources.append({
                "file": meta["file_path"],
                "symbol": meta["symbol_name"],
                "type": meta["symbol_type"],
                "lines": f"{meta['start_line']}-{meta['end_line']}",
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "model": self.model,
        }
