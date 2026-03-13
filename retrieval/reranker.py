"""
reranker.py — Rerank retrieved code chunks for higher relevance.

Uses cross-encoder models (e.g., bge-reranker-large) to improve
retrieval precision beyond vector similarity.
"""

from dataclasses import dataclass


@dataclass
class RerankResult:
    """Result after reranking."""
    document: str
    metadata: dict
    relevance_score: float


class Reranker:
    """Reranks retrieval results using a cross-encoder model."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[RerankResult]:
        """Rerank retrieved results by relevance to the query.
        
        Args:
            query: The user's question.
            results: Raw retrieval results from the vector store.
            top_k: Number of top results to return after reranking.
        
        Returns:
            Reranked list of RerankResult objects.
        """
        if not results:
            return []

        pairs = [(query, r["document"]) for r in results]
        scores = self.model.predict(pairs)

        scored_results = []
        for result, score in zip(results, scores):
            scored_results.append(
                RerankResult(
                    document=result["document"],
                    metadata=result["metadata"],
                    relevance_score=float(score),
                )
            )

        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_results[:top_k]
