"""
retriever.py — Retrieve relevant code chunks for a user query.

Handles query embedding, vector search, and result formatting.
"""

from indexing.embedder import BaseEmbedder
from indexing.vector_store import BaseVectorStore


class CodeRetriever:
    """Retrieves relevant code chunks from the vector store."""

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
        
        context_parts = []
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
