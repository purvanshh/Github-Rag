"""retriever.py — Retrieve relevant code chunks for a user query.

Handles query embedding, vector search, and result formatting.

This module provides:
    * CodeRetriever       — pure vector similarity search
    * HybridCodeRetriever — vector search + cross-encoder reranking
"""

from __future__ import annotations

from typing import List

from indexing.embedder import BaseEmbedder
from indexing.vector_store import BaseVectorStore
from retrieval.reranker import Reranker, RerankResult


class CodeRetriever:
    """Retrieves relevant code chunks from the vector store using embeddings only."""

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore, top_k: int = 5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve the most relevant code chunks for a natural language query.

        Args:
            query: User's question about the codebase.

        Returns:
            List of result dicts with document, metadata, and distance.
        """
        query_embedding = self.embedder.embed_texts([query])[0]
        results = self.vector_store.query(query_embedding, top_k=self.top_k)
        return results

    def retrieve_with_context(self, query: str) -> str:
        """Retrieve chunks and format them as context for the LLM.

        Args:
            query: User's question about the codebase.

        Returns:
            Formatted string of code context for prompt injection.
        """
        results = self.retrieve(query)

        context_parts: List[str] = []
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            context_parts.append(
                f"--- Context {i} ---\n"
                f"File: {meta['file_path']}\n"
                f"Symbol: {meta['symbol_name']} ({meta['symbol_type']})\n"
                f"Lines: {meta['start_line']}-{meta['end_line']}\n\n"
                f"{result['document']}\n"
            )

        return "\n".join(context_parts)


class HybridCodeRetriever:
    """Hybrid retriever: vector similarity search followed by cross-encoder reranking.

    This implements the pipeline:

        query
        → vector similarity search (top_k_initial results)
        → bge-reranker-large cross-encoder
        → top_k_final results
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        reranker: Reranker | None = None,
        top_k_initial: int = 20,
        top_k_final: int = 5,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        # Default to BAAI/bge-reranker-large as required.
        self.reranker = reranker or Reranker(model_name="BAAI/bge-reranker-large")

    def retrieve(self, query: str) -> list[dict]:
        """Run hybrid retrieval and return the top reranked results.

        Returns:
            List of dicts with ``document``, ``metadata``, and
            ``relevance_score`` keys.
        """
        # 1) Vector similarity search
        query_embedding = self.embedder.embed_texts([query])[0]
        initial_results = self.vector_store.query(
            query_embedding,
            top_k=self.top_k_initial,
        )

        # 2) Cross-encoder reranking
        reranked: list[RerankResult] = self.reranker.rerank(
            query, initial_results, top_k=self.top_k_final
        )

        # Normalize to the same shape used elsewhere in the system.
        output: list[dict] = []
        for item in reranked:
            output.append(
                {
                    "document": item.document,
                    "metadata": item.metadata,
                    "relevance_score": item.relevance_score,
                }
            )
        return output

    def retrieve_with_context(self, query: str) -> str:
        """Retrieve reranked chunks and format them as context for the LLM."""
        results = self.retrieve(query)

        context_parts: List[str] = []
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            score = result.get("relevance_score")
            score_part = f"Score: {score:.4f}\n" if score is not None else ""
            context_parts.append(
                f"--- Context {i} ---\n"
                f"File: {meta['file_path']}\n"
                f"Symbol: {meta['symbol_name']} ({meta['symbol_type']})\n"
                f"Lines: {meta['start_line']}-{meta['end_line']}\n"
                f"{score_part}\n"
                f"{result['document']}\n"
            )

        return "\n".join(context_parts)
