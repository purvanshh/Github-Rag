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

from config import config, get_embedder
from indexing.vector_store import ChromaVectorStore, BaseVectorStore
from ingestion.clone_repo import clone_repository
from ingestion.chunk_code import CodeChunk, create_chunks_from_symbols
from ingestion.parse_code import ParsedSymbol, parse_directory
from graphs.knowledge_graph import RepositoryKnowledgeGraph

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
        self.embedder = get_embedder()
        self.vector_store: BaseVectorStore = ChromaVectorStore(
            collection_name=config.chroma_collection,
            persist_dir=config.chroma_persist_dir,
        )

    def ingest_repository(self, repo_url: str, incremental: bool = False) -> IngestionResult:
        """Ingest a GitHub repository and prepare it for querying.

        Args:
            repo_url: GitHub URL of the repository to ingest.
            incremental: If True, only re-indexes modified/new files.

        Returns:
            IngestionResult with key statistics about the run.
        """
        import hashlib
        
        def compute_file_hash(file_path: str) -> str:
            hasher = hashlib.sha256()
            try:
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            except OSError:
                return ""
            return hasher.hexdigest()

        start_time = time.time()
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

        logger.info("Starting ingestion for repo: %s (incremental=%s)", repo_url, incremental)

        # 1. Clone repository
        logger.info("[1/8] Cloning repository...")
        repo_path = clone_repository(repo_url)

        # Build paths
        metadata_dir = os.path.join(config.repos_dir, repo_name)
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "repo_metadata.json")
        kg_path = os.path.join(metadata_dir, "knowledge_graph.json")

        # Load existing hashes and graph if doing incremental build
        last_indexed_hashes = {}
        kg = RepositoryKnowledgeGraph()
        old_metadata = {}
        if incremental and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    old_metadata = json.load(f)
                last_indexed_hashes = old_metadata.get("file_hashes", {})
                
                if os.path.exists(kg_path):
                    with open(kg_path, "r", encoding="utf-8") as f:
                        kg_data = json.load(f)
                    kg.load_from_dict(kg_data)
            except Exception as exc:
                logger.warning("Could not load incremental cache: %s", exc)

        # Scan all repository files to compute current hashes
        from ingestion.parse_code import get_language_for_file
        from metadata_utils import normalize_file_path
        
        current_hashes = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in (".git", "venv", ".venv", "__pycache__", "node_modules")]
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if get_language_for_file(full_path):
                    rel_path = normalize_file_path(full_path, repo_path)
                    current_hashes[rel_path] = compute_file_hash(full_path)

        # Determine changes
        added_files = set()
        modified_files = set()
        deleted_files = set()

        for file_path, current_hash in current_hashes.items():
            if file_path not in last_indexed_hashes:
                added_files.add(file_path)
            elif last_indexed_hashes[file_path] != current_hash:
                modified_files.add(file_path)
                
        for file_path in last_indexed_hashes:
            if file_path not in current_hashes:
                deleted_files.add(file_path)

        is_incremental_run = incremental and last_indexed_hashes
        
        if is_incremental_run:
            logger.info("Incremental changes: %d added, %d modified, %d deleted files", len(added_files), len(modified_files), len(deleted_files))
            if not added_files and not modified_files and not deleted_files:
                logger.info("No changes detected. Skipping ingestion.")
                num_symbols = old_metadata.get("num_symbols", 0)
                num_chunks = old_metadata.get("num_chunks", 0)
                return IngestionResult(repo_name, len(current_hashes), num_symbols, num_chunks, time.time() - start_time)

            # Step 1: Remove chunks and graph references for deleted and modified files
            files_to_remove = deleted_files | modified_files
            for fpath in files_to_remove:
                # Delete chunks from Chroma DB
                try:
                    self.vector_store.collection.delete(where={"file_path": fpath})
                except Exception as exc:
                    logger.warning("Failed to delete chunks for %s: %s", fpath, exc)

                # Delete nodes/edges from Knowledge Graph
                nodes_to_remove = [
                    node for node, attr in list(kg.graph.nodes(data=True))
                    if attr.get("file_path") == fpath or node.startswith(f"{fpath}:")
                ]
                for node in nodes_to_remove:
                    if node in kg.graph:
                        kg.graph.remove_node(node)

            # Step 2: Parse and chunk only added and modified files
            logger.info("[2/8] Incrementally parsing changed files...")
            from ingestion.parse_code import parse_file
            symbols = []
            for fpath in (added_files | modified_files):
                abs_fpath = os.path.join(repo_path, fpath)
                if os.path.exists(abs_fpath):
                    symbols.extend(parse_file(abs_fpath, repo_path=repo_path, repo_id=repo_name))

            num_symbols = len(symbols)
            logger.info("Incremental run parsed %d new/updated symbols", num_symbols)

            # Create new chunks
            logger.info("[3/8] Creating incremental chunks...")
            chunks = create_chunks_from_symbols(symbols, repo_name)
            num_chunks = len(chunks)
            logger.info("Created %d incremental chunks", num_chunks)

            # Embed and insert
            if chunks:
                logger.info("[4/8] Generating incremental embeddings...")
                embeddings = self._embed_chunks_with_cache(chunks, repo_name)
                logger.info("[5/8] Storing incremental vectors in Chroma...")
                self.vector_store.add_chunks(chunks, embeddings)

            # Re-build knowledge graph for the new/updated symbols
            logger.info("[6/8] Updating Repository Knowledge Graph...")
            kg.build(repo_path, symbols)
            
            # Recalculate stats based on full current graph
            dep_hubs = kg.get_most_connected(top_k=10)
            most_called = kg.get_most_called(top_k=10)
            
            total_symbols = len([n for n, attr in kg.graph.nodes(data=True) if attr.get("type") in ("function", "method", "class")])
            total_chunks = len(self.vector_store.get_all_chunks())
            
        else:
            # Full run
            # 2. Parse directory with Tree-sitter
            logger.info("[2/8] Parsing source code...")
            symbols = parse_directory(repo_path, repo_id=repo_name)
            num_symbols = len(symbols)
            logger.info("Parsed %d symbols", num_symbols)

            # 3. Create chunks from symbols
            logger.info("[3/8] Creating semantic chunks...")
            chunks = create_chunks_from_symbols(symbols, repo_name)
            num_chunks = len(chunks)
            logger.info("Created %d chunks", num_chunks)

            # 4. Generate embeddings
            logger.info("[4/8] Generating embeddings...")
            embeddings = self._embed_chunks_with_cache(chunks, repo_name)
            logger.info("Generated embeddings for %d chunks", len(embeddings))

            # 5. Store in vector database
            logger.info("[5/8] Storing vectors in Chroma...")
            self.vector_store.add_chunks(chunks, embeddings)

            # 6. Build unified Repository Knowledge Graph
            logger.info("[6/8] Building Repository Knowledge Graph...")
            kg = RepositoryKnowledgeGraph()
            kg.build(repo_path, symbols)
            dep_hubs = kg.get_most_connected(top_k=10)
            most_called = kg.get_most_called(top_k=10)
            total_symbols = num_symbols
            total_chunks = num_chunks

        # 6.5. Ingest GitHub metadata & commits
        logger.info("[6.5/8] Integrating GitHub API metadata & commits...")
        try:
            from ingestion.github_integration import GitHubIntegrationEngine
            gh_engine = GitHubIntegrationEngine()
            gh_chunks = gh_engine.ingest_metadata(repo_path, repo_name, repo_url)
            total_chunks = len(self.vector_store.get_all_chunks())
        except Exception as exc:
            logger.warning("Failed to ingest GitHub metadata: %s", exc)

        # 7. Store repo metadata & knowledge graph on disk
        logger.info("[7/8] Storing repo metadata & knowledge graph...")
        repo_metadata = {
            "repo_name": repo_name,
            "repo_url": repo_url,
            "num_symbols": total_symbols,
            "num_chunks": total_chunks,
            "dependency_hubs": dep_hubs,
            "most_called_functions": most_called,
            "file_hashes": current_hashes,
        }

        # Save knowledge graph
        try:
            with open(kg_path, "w", encoding="utf-8") as f:
                json.dump(kg.to_dict(), f, indent=2)
        except OSError as exc:
            logger.warning("Failed to write knowledge graph to %s: %s", kg_path, exc)

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(repo_metadata, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to write repo metadata to %s: %s", metadata_path, exc)

        indexing_time = time.time() - start_time
        result = IngestionResult(
            repo_name=repo_name,
            num_files=len(current_hashes),
            num_symbols=total_symbols,
            num_chunks=total_chunks,
            indexing_time=indexing_time,
        )

        logger.info(
            "Completed ingestion for %s in %.2fs (%d files, %d symbols, %d chunks)",
            repo_name,
            indexing_time,
            len(current_hashes),
            total_symbols,
            total_chunks,
        )

        return result

    def _embed_chunks_with_cache(self, chunks: list[CodeChunk], repo_name: str) -> list[list[float]]:
        from indexing.cache_manager import LocalCacheManager
        cache = LocalCacheManager(repo_name)
        embeddings = [None] * len(chunks)
        uncached_chunks = []
        uncached_indices = []
        
        for idx, chunk in enumerate(chunks):
            txt = chunk.to_embedding_text()
            cached_emb = cache.get("embedding", txt)
            if cached_emb:
                embeddings[idx] = cached_emb
            else:
                uncached_chunks.append(chunk)
                uncached_indices.append(idx)
                
        if uncached_chunks:
            new_embs = self.embedder.embed_chunks(uncached_chunks)
            for new_idx, orig_idx in enumerate(uncached_indices):
                emb = new_embs[new_idx]
                embeddings[orig_idx] = emb
                cache.set("embedding", uncached_chunks[new_idx].to_embedding_text(), emb)
                
        return embeddings

