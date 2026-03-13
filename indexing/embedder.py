"""
embedder.py — Generate embeddings for code chunks.

Supports OpenAI text-embedding-3-large and open-source alternatives
(BAAI/bge-large, Instructor-xl).
"""

from abc import ABC, abstractmethod

from ingestion.chunk_code import CodeChunk


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Generate embeddings for code chunks using their embedding text."""
        texts = [chunk.to_embedding_text() for chunk in chunks]
        return self.embed_texts(texts)


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's text-embedding-3-large model."""

    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI API."""
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class LocalEmbedder(BaseEmbedder):
    """Embedder using local HuggingFace models (e.g., BAAI/bge-large-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings locally."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
