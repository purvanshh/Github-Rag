"""
config.py — Centralized configuration for the GitHub RAG system.

Loads settings from environment variables with sensible defaults.
Loads .env from the project root when present.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root and cwd so the key is always available
_root = Path(__file__).resolve().parent
for _p in [_root / ".env", Path.cwd() / ".env"]:
    if _p.exists():
        load_dotenv(_p, override=True)
        break
load_dotenv(override=False)  # cwd .env without overwriting

from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Provider: "openai" or "gemini"
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    # OpenAI (strip whitespace so key is valid)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    gemini_llm_model: str = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash")

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


def get_openai_api_key() -> str:
    """Return the OpenAI API key, or raise a clear error if missing."""
    key = _clean_api_key(
        os.getenv("OPENAI_API_KEY") or config.openai_api_key or ""
    )
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root: "
            'OPENAI_API_KEY=sk-your-key-here'
        )
    return key


def _clean_api_key(raw: str) -> str:
    """Strip whitespace and optional surrounding quotes from an API key."""
    if not raw:
        return ""
    key = raw.strip().strip("'\"")
    return key.strip()


def get_gemini_api_key() -> str:
    """Return the Gemini API key, or raise a clear error if missing."""
    key = _clean_api_key(
        os.getenv("GEMINI_API_KEY") or config.gemini_api_key or ""
    )
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root: "
            "GEMINI_API_KEY=your-gemini-key"
        )
    return key


def get_embedder():
    """Return the configured embedder (Gemini or OpenAI) based on LLM_PROVIDER."""
    from indexing.embedder import GeminiEmbedder, OpenAIEmbedder
    if config.llm_provider == "gemini":
        return GeminiEmbedder(
            model=config.gemini_embedding_model,
            api_key=get_gemini_api_key(),
        )
    return OpenAIEmbedder(model=config.embedding_model, api_key=get_openai_api_key())
