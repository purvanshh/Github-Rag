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
        """Collect related function names/FQNs using the call graph.

        For each initially retrieved chunk, we include:
            * Functions that call this function
            * Functions that it calls
        """
        neighbor_funcs: set[str] = set()
        for result in initial_results:
            meta = result.get("metadata", {})
            fqn = meta.get("fqn")
            symbol_name = meta.get("symbol_name")

            for key in (fqn, symbol_name):
                if not key:
                    continue
                callers = self.call_graph.get_callers(key)
                callees = self.call_graph.get_callees(key)
                neighbor_funcs.update(callers)
                neighbor_funcs.update(callees)

        return neighbor_funcs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Run graph-aware retrieval augmented with BM25 lexical search and return the top reranked chunks.

        Steps:
            1. Vector similarity search (initial semantic match).
            2. Lexical search (BM25).
            3. Graph Expansion: Find neighbor files and neighbor functions based on combined search.
            4. Retrieve Expanded Nodes: Vector search restricted to graph neighbors.
            5. Merge: Deduplicate and combine initial semantic, lexical, and expanded results.
            6. Rerank: Cross-encoder reranking of candidate chunks.
        """
        # 1) Vector similarity search (initial)
        query_embedding = self.embedder.embed_texts([query])[0]
        initial_results = self.vector_store.query(
            query_embedding,
            top_k=self.top_k_initial,
        )

        # 2) Lexical search (BM25)
        from retrieval.bm25 import BM25Retriever
        all_chunks = self.vector_store.get_all_chunks()
        bm25 = BM25Retriever()
        bm25.fit(all_chunks)
        lexical_results = bm25.query(query, top_k=self.top_k_initial)

        # 3) Graph Expansion
        # Combine initial semantic and lexical results for expansion
        combined_results = []
        seen_ids = set()
        for r in initial_results + lexical_results:
            chunk_id = r.get("id") or r.get("metadata", {}).get("id")
            if not chunk_id:
                meta = r.get("metadata", {})
                chunk_id = f"{meta.get('file_path')}:{meta.get('symbol_name')}:{meta.get('start_line')}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                combined_results.append(r)

        neighbor_files = self._collect_neighbor_files(combined_results)
        neighbor_funcs = self._collect_neighbor_functions(combined_results)

        # 4) Retrieve Expanded Nodes
        expanded_results_for_files: List[Dict[str, Any]] = []
        if neighbor_files:
            where_files = {"file_path": {"$in": list(neighbor_files)}}
            expanded_results_for_files = self.vector_store.query(
                query_embedding,
                top_k=self.top_k_expanded,
                where=where_files,
            )

        expanded_results_for_funcs: List[Dict[str, Any]] = []
        if neighbor_funcs:
            # First try matching by normalized FQN
            where_fqn = {"fqn": {"$in": list(neighbor_funcs)}}
            expanded_results_for_funcs = self.vector_store.query(
                query_embedding,
                top_k=self.top_k_expanded,
                where=where_fqn,
            )
            # Fall back to symbol_name if no results found (legacy / short names)
            if not expanded_results_for_funcs:
                where_sym = {"symbol_name": {"$in": list(neighbor_funcs)}}
                expanded_results_for_funcs = self.vector_store.query(
                    query_embedding,
                    top_k=self.top_k_expanded,
                    where=where_sym,
                )

        # 5) Merge + deduplicate (prefer earliest occurrence)
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
        add_results(lexical_results)
        add_results(expanded_results_for_files)
        add_results(expanded_results_for_funcs)

        merged_results = list(candidates.values())
        if not merged_results:
            return []

        # 6) Cross-encoder reranking over all candidates.
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

