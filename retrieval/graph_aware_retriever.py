"""graph_aware_retriever.py — Retrieval augmented by code graphs.

This retriever improves repository reasoning by expanding initial
vector-search results using:

    * Dependency graph (file-level imports)
    * Function call graph (who calls whom)

Pipeline:
    query
    → vector similarity search
    → expand via graphs (files + functions)
    → cross-encoder reranking (bge-reranker-large)
    → final top-k chunks

The return shape is compatible with AnswerGenerator and other
retrievers in this project (dicts with ``document`` and ``metadata``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

from graphs.call_graph import CallGraph
from graphs.dependency_graph import DependencyGraph
from indexing.embedder import BaseEmbedder
from indexing.vector_store import BaseVectorStore
from retrieval.reranker import Reranker, RerankResult


class GraphAwareRetriever:
    """Hybrid + graph-aware retriever.

    This retriever first performs standard vector similarity search,
    then expands candidate chunks based on dependency and call graphs,
    and finally reranks all candidates with a cross-encoder.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        dependency_graph: DependencyGraph,
        call_graph: CallGraph,
        *,
        top_k_initial: int = 20,
        top_k_expanded: int = 50,
        top_k_final: int = 5,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize the graph-aware retriever.

        Args:
            embedder: Embedding provider used for query embeddings.
            vector_store: Vector store backend (e.g., ChromaVectorStore).
            dependency_graph: Pre-built dependency graph for the repository.
            call_graph: Pre-built call graph for the repository.
            top_k_initial: Number of raw vector results to retrieve.
            top_k_expanded: Number of additional vector results to fetch
                when expanding via graph neighborhoods.
            top_k_final: Final number of chunks returned after reranking.
            reranker: Optional preconfigured Reranker. If omitted, a
                default ``BAAI/bge-reranker-large`` instance is created.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.dependency_graph = dependency_graph
        self.call_graph = call_graph
        self.top_k_initial = top_k_initial
        self.top_k_expanded = top_k_expanded
        self.top_k_final = top_k_final
        self.reranker = reranker or Reranker(model_name="BAAI/bge-reranker-large")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_neighbor_files(self, initial_results: List[Dict[str, Any]]) -> Set[str]:
        """Collect related file paths using the dependency graph.

        For each initially retrieved chunk, we include:
            * Files it imports
            * Files that import it
        """
        neighbor_files: set[str] = set()
        for result in initial_results:
            meta = result.get("metadata", {})
            file_path = meta.get("file_path")
            if not file_path:
                continue

            # Outgoing deps: files this file imports
            for dep in self.dependency_graph.get_dependencies(file_path):
                neighbor_files.add(dep)

            # Incoming deps: files that import this file
            for parent in self.dependency_graph.get_dependents(file_path):
                neighbor_files.add(parent)

        return neighbor_files

    def _collect_neighbor_functions(self, initial_results: List[Dict[str, Any]]) -> Set[str]:
        """Collect related function names using the call graph.

        For each initially retrieved chunk, we include:
            * Functions that call this function
            * Functions that it calls
        """
        neighbor_funcs: set[str] = set()
        for result in initial_results:
            meta = result.get("metadata", {})
            symbol_name = meta.get("symbol_name")
            if not symbol_name:
                continue

            # Use short name; call graph is built with FQNs when available,
            # but also supports generic lookups.
            callers = self.call_graph.get_callers(symbol_name)
            callees = self.call_graph.get_callees(symbol_name)
            neighbor_funcs.update(callers)
            neighbor_funcs.update(callees)

        return neighbor_funcs

    def _filter_results_by_files(
        self,
        results: Iterable[Dict[str, Any]],
        file_paths: Set[str],
    ) -> List[Dict[str, Any]]:
        """Filter vector results to those whose ``file_path`` is in *file_paths*."""
        if not file_paths:
            return []
        filtered: list[Dict[str, Any]] = []
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("file_path") in file_paths:
                filtered.append(r)
        return filtered

    def _filter_results_by_functions(
        self,
        results: Iterable[Dict[str, Any]],
        function_names: Set[str],
    ) -> List[Dict[str, Any]]:
        """Filter vector results to those whose ``symbol_name`` is in *function_names*."""
        if not function_names:
            return []
        filtered: list[Dict[str, Any]] = []
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("symbol_name") in function_names:
                filtered.append(r)
        return filtered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Run graph-aware retrieval and return the top reranked chunks.

        Steps:
            1. Vector similarity search to get initial chunks.
            2. Expand candidate set using dependency and call graphs.
            3. Deduplicate candidates by chunk id.
            4. Rerank all candidates with a cross-encoder (bge-reranker-large).
        """
        # 1) Vector similarity search (initial)
        query_embedding = self.embedder.embed_texts([query])[0]
        initial_results = self.vector_store.query(
            query_embedding,
            top_k=self.top_k_initial,
        )

        # 2) Graph-based expansion
        neighbor_files = self._collect_neighbor_files(initial_results)
        neighbor_funcs = self._collect_neighbor_functions(initial_results)

        # Fetch additional chunks from the vector store and then filter by
        # graph neighborhoods. We keep the candidate set bounded via
        # ``top_k_expanded`` and rely on dedup + reranking.
        expanded_results_for_files: List[Dict[str, Any]] = []
        if neighbor_files:
            expanded_results_for_files = self.vector_store.query(
                query_embedding,
                top_k=self.top_k_expanded,
            )
            expanded_results_for_files = self._filter_results_by_files(
                expanded_results_for_files,
                neighbor_files,
            )

        expanded_results_for_funcs: List[Dict[str, Any]] = []
        if neighbor_funcs:
            expanded_results_for_funcs = self.vector_store.query(
                query_embedding,
                top_k=self.top_k_expanded,
            )
            expanded_results_for_funcs = self._filter_results_by_functions(
                expanded_results_for_funcs,
                neighbor_funcs,
            )

        # 3) Merge + deduplicate (prefer earliest occurrence)
        candidates: Dict[str, Dict[str, Any]] = {}

        def add_results(rs: Iterable[Dict[str, Any]]) -> None:
            for r in rs:
                chunk_id = r.get("id") or r.get("metadata", {}).get("id")
                # Fall back to (file_path, symbol_name, start_line) if no id.
                if not chunk_id:
                    meta = r.get("metadata", {})
                    chunk_id = f"{meta.get('file_path')}:{meta.get('symbol_name')}:{meta.get('start_line')}"
                if chunk_id not in candidates:
                    candidates[chunk_id] = r

        add_results(initial_results)
        add_results(expanded_results_for_files)
        add_results(expanded_results_for_funcs)

        merged_results = list(candidates.values())
        if not merged_results:
            return []

        # 4) Cross-encoder reranking over all candidates.
        reranked: List[RerankResult] = self.reranker.rerank(
            query,
            merged_results,
            top_k=self.top_k_final,
        )

        # Normalize to dicts compatible with AnswerGenerator expectations.
        output: List[Dict[str, Any]] = []
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
        """Retrieve chunks and format them as LLM-ready context string."""
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

