"""
config.py — Centralized configuration for the GitHub RAG system.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Vector Store
    vector_store_type: str = os.getenv("VECTOR_STORE", "chroma")  # "chroma" or "pinecone"
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "codebase")

    # Pinecone (optional)
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "github-rag")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() == "true"

    # Repos
    repos_dir: str = os.getenv("REPOS_DIR", "./repos")

    # Server
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))


config = Config()
