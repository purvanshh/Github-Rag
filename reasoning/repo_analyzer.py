"""repo_analyzer.py — High-level orchestration of repo intelligence features.

The RepoAnalyzer class unifies:
    * Graph-aware retrieval (vector + graphs + reranker) + LLM answering
    * Architecture summarization
    * Dependency graph queries
    * Function call graph queries
    * File-level explanations
    * Combined repository overview

This provides a single, production-friendly interface that the API layer
and UI can use without needing to know about low-level modules.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from config import config, get_embedder
from graphs.call_graph import (
    CallGraph,
    build_call_graph,
    where_is_function_used,
    which_functions_does_it_call,
)
from graphs.dependency_graph import DependencyGraph, build_dependency_graph
from indexing.vector_store import ChromaVectorStore, BaseVectorStore
from reasoning.answer_generator import AnswerGenerator
from reasoning.architecture_summarizer import (
    build_directory_tree,
    generate_architecture_summary,
)
from retrieval.graph_aware_retriever import GraphAwareRetriever


class RepoAnalyzer:
    """High-level orchestrator for repository intelligence.

    This class wires together embedding, hybrid retrieval, graphs, and LLMs
    into a cohesive interface that can be used by the API and UI layers.

    Graphs (dependency and call graphs) are built once at initialization
    and cached to avoid recomputation.
    """

    def __init__(
        self,
        repo_name: str,
        *,
        repos_root: str | None = None,
        top_k_initial: int = 20,
        top_k_final: int = 5,
    ) -> None:
        """Create a RepoAnalyzer for a given repository.

        Args:
            repo_name: Name of the repository (directory under ``repos_root``).
            repos_root: Optional path where repositories are stored. Defaults
                to ``config.repos_dir``.
            top_k_initial: Number of vector search hits before reranking.
            top_k_final: Number of reranked results to keep (typically 5).
        """
        self.repo_name = repo_name
        self.repos_root = repos_root or config.repos_dir
        self.repo_path = os.path.abspath(
            os.path.join(self.repos_root, self.repo_name)
        )

        # --- Graphs: build once and cache ---
        self._dependency_graph: DependencyGraph = build_dependency_graph(
            self.repo_path
        )
        self._call_graph: CallGraph = build_call_graph(self.repo_path)

        # --- Retrieval stack (GraphAwareRetriever + AnswerGenerator) ---
        self._embedder = get_embedder()
        self._vector_store: BaseVectorStore = ChromaVectorStore(
            collection_name=config.chroma_collection,
            persist_dir=config.chroma_persist_dir,
        )
        self._retriever = GraphAwareRetriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            dependency_graph=self._dependency_graph,
            call_graph=self._call_graph,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
        )
        self._answer_generator = AnswerGenerator(
            retriever=self._retriever,
            model=config.llm_model,
            temperature=config.llm_temperature,
        )

        # --- Repo metadata helpers ---
        self._directory_tree: str = build_directory_tree(self.repo_path)
        self._architecture_summary_cache: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 1. Question answering over the codebase
    # ------------------------------------------------------------------

    def ask_question(self, query: str) -> Dict[str, Any]:
        """Answer an architectural question about the repository.

        Pipeline:
            query
            → hybrid retrieval (vector + reranker)
            → LLM (AnswerGenerator)

        Returns:
            Dict with ``answer``, ``sources``, and ``model`` keys, as
            produced by :class:`AnswerGenerator`.
        """
        return self._answer_generator.generate_answer(query)

    # ------------------------------------------------------------------
    # 2. Architecture summary
    # ------------------------------------------------------------------

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Return a high-level architecture summary of the repository.

        Uses the existing architecture summarizer, and caches the result
        for subsequent calls.
        """
        if self._architecture_summary_cache is None:
            self._architecture_summary_cache = generate_architecture_summary(
                repo_path=self.repo_path,
                model=config.llm_model,
                temperature=config.llm_temperature,
            )
        return self._architecture_summary_cache

    # ------------------------------------------------------------------
    # 3. Function usage via call graph
    # ------------------------------------------------------------------

    def find_function_usage(self, function_name: str) -> Dict[str, List[str]]:
        """Return where a function is used and what it calls.

        Args:
            function_name: Fully-qualified function name when possible, e.g.
                ``package.module.func`` or ``package.module.Class.method``.
                Short names are accepted but may be ambiguous.

        Returns:
            Dict with:
                - ``callers``: functions that call this function.
                - ``callees``: functions that this function calls.
        """
        callers = where_is_function_used(self._call_graph, function_name)
        callees = which_functions_does_it_call(self._call_graph, function_name)
        return {
            "callers": callers,
            "callees": callees,
        }

    # ------------------------------------------------------------------
    # 4. File dependencies via dependency graph
    # ------------------------------------------------------------------

    def _to_relative_path(self, file_path: str) -> str:
        """Normalize a file path to be relative to the repository root."""
        if not os.path.isabs(file_path):
            abs_path = os.path.join(self.repo_path, file_path)
        else:
            abs_path = file_path
        return os.path.relpath(abs_path, self.repo_path)

    def get_file_dependencies(self, file_path: str) -> List[str]:
        """Return all files/modules that the given file depends on.

        Args:
            file_path: Absolute or repo-relative path to the file.
        """
        rel_path = self._to_relative_path(file_path)
        return self._dependency_graph.get_dependencies(rel_path)

    # ------------------------------------------------------------------
    # 5. File-level explanation
    # ------------------------------------------------------------------

    def explain_file(self, file_path: str) -> Dict[str, Any]:
        """Use the LLM to summarize what a file does based on its chunks.

        This method issues a natural-language question that explicitly asks
        the system to explain the target file. Hybrid retrieval uses the
        indexed code chunks (which include file path metadata), so the LLM
        answer is grounded in the file's semantic chunks.

        Args:
            file_path: Absolute or repo-relative path to the file within
                the repository.

        Returns:
            Dict with the LLM's ``answer`` plus ``sources`` and ``model``,
            in the same shape as :meth:`ask_question`.
        """
        rel_path = self._to_relative_path(file_path)
        question = (
            f"Explain the purpose, responsibilities, and key classes/functions "
            f"of the file `{rel_path}` within this repository."
        )
        return self._answer_generator.generate_answer(question)

    # ------------------------------------------------------------------
    # 6. Combined repository overview
    # ------------------------------------------------------------------

    def get_repo_overview(self) -> Dict[str, Any]:
        """Return a combined overview report for the repository.

        The report includes:
            * Architecture summary (LLM-generated)
            * Most connected modules (from the dependency graph)
            * Most called functions (from the call graph)
            * Directory tree (precomputed)
        """
        arch = self.get_architecture_summary()
        most_connected = self._dependency_graph.get_most_connected(top_k=10)
        most_called = self._call_graph.get_most_called(top_k=10)

        return {
            "architecture_summary": arch.get("summary"),
            "architecture_metadata": {
                "model": arch.get("model"),
                "dependency_hubs": arch.get("dependency_hubs"),
            },
            "most_connected_modules": most_connected,
            "most_called_functions": most_called,
            "directory_tree": self._directory_tree,
        }

