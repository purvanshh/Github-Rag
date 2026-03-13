"""repo_pipeline.py — End-to-end repository ingestion pipeline.

This module provides a single orchestration entry point that:

    1. Clones a GitHub repository.
    2. Parses source files with Tree-sitter and extracts symbols.
    3. Chunks code into semantic units.
    4. Generates embeddings for chunks.
    5. Stores embeddings in the vector database.
    6. Builds dependency and call graphs.
    7. Stores basic repository metadata.

The result is a repository that is fully prepared for querying via
RepoAnalyzer, QueryRouter, and the API/UI layers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Any

from config import config
from graphs.call_graph import build_call_graph
from graphs.dependency_graph import build_dependency_graph
from indexing.embedder import OpenAIEmbedder
from indexing.vector_store import ChromaVectorStore, BaseVectorStore
from ingestion.clone_repo import clone_repository
from ingestion.chunk_code import CodeChunk, create_chunks_from_symbols
from ingestion.parse_code import ParsedSymbol, parse_directory

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Summary of a repository ingestion run."""

    repo_name: str
    num_files: int
    num_symbols: int
    num_chunks: int
    indexing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "num_files": self.num_files,
            "num_symbols": self.num_symbols,
            "num_chunks": self.num_chunks,
            "indexing_time": self.indexing_time,
        }


class RepoIngestionPipeline:
    """End-to-end pipeline for preparing a repository for querying."""

    def __init__(self) -> None:
        self.embedder = OpenAIEmbedder(model=config.embedding_model)
        self.vector_store: BaseVectorStore = ChromaVectorStore(
            collection_name=config.chroma_collection,
            persist_dir=config.chroma_persist_dir,
        )

    def ingest_repository(self, repo_url: str) -> IngestionResult:
        """Ingest a GitHub repository and prepare it for querying.

        Args:
            repo_url: GitHub URL of the repository to ingest.

        Returns:
            IngestionResult with key statistics about the run.
        """
        start_time = time.time()
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

        logger.info("Starting ingestion for repo: %s", repo_url)

        # 1. Clone repository
        logger.info("[1/8] Cloning repository...")
        repo_path = clone_repository(repo_url)

        # 2. Parse directory with Tree-sitter
        logger.info("[2/8] Parsing source code...")
        symbols: list[ParsedSymbol] = parse_directory(repo_path)
        num_symbols = len(symbols)
        logger.info("Parsed %d symbols", num_symbols)

        # 3. Create chunks from symbols
        logger.info("[3/8] Creating semantic chunks...")
        chunks: list[CodeChunk] = create_chunks_from_symbols(symbols, repo_name)
        num_chunks = len(chunks)
        logger.info("Created %d chunks", num_chunks)

        # 4. Generate embeddings
        logger.info("[4/8] Generating embeddings...")
        embeddings = self.embedder.embed_chunks(chunks)
        logger.info("Generated embeddings for %d chunks", len(embeddings))

        # 5. Store in vector database
        logger.info("[5/8] Storing vectors in Chroma...")
        self.vector_store.add_chunks(chunks, embeddings)

        # 6. Build dependency graph
        logger.info("[6/8] Building dependency graph...")
        dep_graph = build_dependency_graph(repo_path)
        dep_hubs = dep_graph.get_most_connected(top_k=10)

        # 7. Build call graph
        logger.info("[7/8] Building call graph...")
        call_graph = build_call_graph(repo_path)
        most_called = call_graph.get_most_called(top_k=10)

        # 8. Store repo metadata on disk
        logger.info("[8/8] Storing repo metadata...")
        repo_metadata = {
            "repo_name": repo_name,
            "repo_url": repo_url,
            "num_symbols": num_symbols,
            "num_chunks": num_chunks,
            "dependency_hubs": dep_hubs,
            "most_called_functions": most_called,
        }

        metadata_dir = os.path.join(config.repos_dir, repo_name)
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "repo_metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(repo_metadata, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to write repo metadata to %s: %s", metadata_path, exc)

        # Compute stats
        indexing_time = time.time() - start_time

        # Count Python/JS/TS files we actually parsed
        file_paths = {sym.file_path for sym in symbols}
        num_files = len(file_paths)

        result = IngestionResult(
            repo_name=repo_name,
            num_files=num_files,
            num_symbols=num_symbols,
            num_chunks=num_chunks,
            indexing_time=indexing_time,
        )

        logger.info(
            "Completed ingestion for %s in %.2fs (%d files, %d symbols, %d chunks)",
            repo_name,
            indexing_time,
            num_files,
            num_symbols,
            num_chunks,
        )

        return result

