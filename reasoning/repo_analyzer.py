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
from graphs.knowledge_graph import RepositoryKnowledgeGraph
from indexing.vector_store import ChromaVectorStore, BaseVectorStore
from reasoning.answer_generator import AnswerGenerator
from reasoning.architecture_summarizer import (
    build_directory_tree,
    generate_architecture_summary,
)
from retrieval.graph_aware_retriever import GraphAwareRetriever
from reasoning.repo_memory import RepositoryMemory


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
        from ingestion.parse_code import parse_directory
        self._kg = RepositoryKnowledgeGraph()
        symbols = parse_directory(self.repo_path, repo_id=self.repo_name)
        self._kg.build(self.repo_path, symbols)
        self._dependency_graph = self._kg
        self._call_graph = self._kg

        self._top_k_initial = top_k_initial
        self._top_k_final = top_k_final
        self._memory = RepositoryMemory(self.repo_name)

        # --- Repo metadata helpers ---
        self._directory_tree: str = build_directory_tree(self.repo_path)
        self._architecture_summary_cache: Dict[str, Any] | None = None

    @property
    def retriever(self) -> GraphAwareRetriever:
        if not hasattr(self, "_retriever"):
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
                top_k_initial=self._top_k_initial,
                top_k_final=self._top_k_final,
            )
        return self._retriever

    @property
    def answer_generator(self) -> AnswerGenerator:
        if not hasattr(self, "_answer_generator"):
            model = config.gemini_llm_model if config.llm_provider == "gemini" else config.llm_model
            self._answer_generator = AnswerGenerator(
                retriever=self.retriever,
                model=model,
                temperature=config.llm_temperature,
                repo_name=self.repo_name,
            )
        return self._answer_generator

    @property
    def query_planner(self) -> Any:
        if not hasattr(self, "_query_planner"):
            from reasoning.query_planner import AgenticQueryPlanner
            self._query_planner = AgenticQueryPlanner(self)
        return self._query_planner

    # ------------------------------------------------------------------
    # 1. Question answering over the codebase
    # ------------------------------------------------------------------

    def ask_question(self, query: str, conversation_id: str | None = None) -> Dict[str, Any]:
        """Answer an architectural question about the repository.

        Pipeline:
            query
            → hybrid retrieval (vector + reranker)
            → LLM (AnswerGenerator)

        Returns:
            Dict with ``answer``, ``sources``, and ``model`` keys, as
            produced by :class:`AnswerGenerator`.
        """
        return self.answer_generator.generate_answer(query, conversation_id)

    def ask_question_stream(self, query: str, conversation_id: str | None = None):
        """Answer queries by streaming the response tokens and citations."""
        return self.answer_generator.generate_answer_stream(query, conversation_id)

    def ask_agentic(self, query: str, conversation_id: str | None = None) -> Dict[str, Any]:
        """Answer queries using step-by-step agentic execution plans."""
        plan = self.query_planner.create_plan(query)
        return self.query_planner.execute_plan(plan, conversation_id)

    @property
    def architecture_analyzer(self) -> Any:
        if not hasattr(self, "_architecture_analyzer"):
            from reasoning.architecture_analyzer import ArchitectureAnalyzer
            self._architecture_analyzer = ArchitectureAnalyzer(self)
        return self._architecture_analyzer

    def get_dependency_chart(self) -> str:
        """Returns module dependency flow in Mermaid format."""
        return self.architecture_analyzer.generate_dependency_chart()

    def get_class_hierarchy(self) -> str:
        """Returns class inheritance layout in Mermaid format."""
        return self.architecture_analyzer.generate_class_hierarchy()

    def get_sequence_chart(self, function_name: str) -> str:
        """Returns function execution calls in Mermaid format."""
        return self.architecture_analyzer.generate_sequence_chart(function_name)

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

    def get_memory_context(self) -> str:
        """Get the repository memory context (profile & FAQ info)."""
        return self._memory.get_memory_context()

    def add_faq(self, question: str, answer: str) -> None:
        """Add a frequently asked question to the repository memory."""
        self._memory.add_faq(question, answer)

    def update_profile(self, updates: Dict[str, Any]) -> None:
        """Update repository stack profile details."""
        self._memory.update_profile(updates)

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
        callers = self._call_graph.get_callers(function_name)
        callees = self._call_graph.get_callees(function_name)
        return {
            "callers": callers,
            "callees": callees,
        }

    def find_references(self, symbol_name: str) -> List[str]:
        """Find all nodes that reference or call a given symbol."""
        return self._kg.get_references(symbol_name)

    def find_implementations(self, class_name: str) -> List[str]:
        """Find all subclasses that implement a given class."""
        return self._kg.get_implementations(class_name)

    def find_inheritance(self, class_name: str) -> List[str]:
        """Find parent classes of a given class."""
        return self._kg.get_inheritance(class_name)

    def find_dependency_chains(self, file_path: str) -> List[List[str]]:
        """Find import chains originating from a source file."""
        return self._kg.get_dependency_chains(self._to_relative_path(file_path))

    # ------------------------------------------------------------------
    # 4. File dependencies via dependency graph
    # ------------------------------------------------------------------

    def _to_relative_path(self, file_path: str) -> str:
        """Normalize a file path to be relative to the repository root."""
        from metadata_utils import normalize_file_path
        return normalize_file_path(file_path, self.repo_path)

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

    @property
    def explanation_engine(self) -> Any:
        if not hasattr(self, "_explanation_engine"):
            from reasoning.explanation_engine import CodeExplanationEngine
            self._explanation_engine = CodeExplanationEngine(self)
        return self._explanation_engine

    def explain_file_difficulty(self, file_path: str, level: str = "medium") -> str:
        """Explain the contents of a file at beginner/medium/advanced levels."""
        rel_path = self._to_relative_path(file_path)
        full_path = os.path.join(self.repo_path, rel_path)
        if not os.path.exists(full_path):
            return f"File {file_path} not found."
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 15000:
                content = content[:15000] + "\n... [truncated]"
            return self.explanation_engine.explain(content, level)
        except Exception as exc:
            return f"Failed to read file: {exc}"

    def explain_symbol_difficulty(self, symbol_name: str, level: str = "medium") -> str:
        """Explain a class or method at beginner/medium/advanced levels."""
        chunks = self.retriever.retrieve(symbol_name)
        if not chunks:
            return f"Symbol {symbol_name} not found in index."
        content = "\n\n".join([chunk.get("content", "") if hasattr(chunk, "get") else getattr(chunk, "content", "") for chunk in chunks[:3]])
        return self.explanation_engine.explain(content, level)

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

    def run_qa_benchmark(self) -> Dict[str, Any]:
        """Runs RAG performance benchmark suite and scores the results."""
        from evaluation.benchmark import RepositoryQABenchmark
        bench = RepositoryQABenchmark(self)
        return bench.run_suite()

