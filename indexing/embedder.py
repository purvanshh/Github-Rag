"""
embedder.py — Generate embeddings for code chunks.

Supports OpenAI, Google Gemini, and local HuggingFace embedders.
"""

import time
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

    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None):
        from openai import OpenAI
        from config import get_openai_api_key
        key = api_key or get_openai_api_key()
        self.client = OpenAI(api_key=key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI API."""
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class GeminiEmbedder(BaseEmbedder):
    """Embedder using Google Gemini text-embedding model (gemini-embedding-001)."""

    # Free tier: 100 embed_content requests/min; throttle to stay under limit
    _BATCH_SIZE = 20
    _DELAY_BETWEEN_BATCHES_SEC = 6.5  # ~9 batches/min => well under 100/min

    def __init__(self, model: str = "models/gemini-embedding-001", api_key: str | None = None):
        import google.generativeai as genai
        from config import get_gemini_api_key
        key = api_key or get_gemini_api_key()
        genai.configure(api_key=key)
        self._genai = genai
        self.model = model

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One API call; returns list of embeddings. Retries on 429."""
        result = self._genai.embed_content(
            model=self.model,
            content=texts,
            task_type="retrieval_document",
        )
        if hasattr(result, "embedding"):
            emb = result.embedding
            # API may return single vector (list of floats) or batch (list of list of floats)
            if emb and isinstance(emb[0], (list, tuple)):
                return list(emb)
            return [emb]
        if hasattr(result, "embeddings"):
            out = list(result.embeddings)
            # Flatten if SDK returned list of list of vectors (one batch as single element)
            if len(out) == 1 and isinstance(out[0], (list, tuple)) and out[0] and isinstance(out[0][0], (list, tuple)):
                return list(out[0])
            return out
        if isinstance(result, dict):
            emb = result.get("embedding", result.get("embeddings", []))
            if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
                return list(emb)
            if emb:
                return [emb]
            return result.get("embeddings", [])
        return []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Gemini API with throttling and 429 retry."""
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            while True:
                try:
                    emb = self._embed_batch(batch)
                    all_embeddings.extend(emb)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "quota" in err_str or "retry" in err_str:
                        # Wait and retry (API often suggests ~15s)
                        time.sleep(15)
                        continue
                    raise
            if i + self._BATCH_SIZE < len(texts):
                time.sleep(self._DELAY_BETWEEN_BATCHES_SEC)
        return all_embeddings


class LocalEmbedder(BaseEmbedder):
    """Embedder using local HuggingFace models (e.g., BAAI/bge-large-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings locally."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
