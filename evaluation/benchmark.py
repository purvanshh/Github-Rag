"""
benchmark.py — Repository QA Benchmark for scoring retrieval and generation quality.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Predefined benchmark queries & expected cited source files for Github-Rag repo itself
BENCHMARK_SUITE = [
    {
        "query": "Where is the API server configured?",
        "expected_files": ["api/server.py", "config.py"],
    },
    {
        "query": "How is symbol normalization performed?",
        "expected_files": ["metadata_utils.py"],
    },
    {
        "query": "What is the entrypoint of the repository pipeline ingestion?",
        "expected_files": ["ingestion/repo_pipeline.py"],
    },
]


class RepositoryQABenchmark:
    """Benchmark runner evaluating precision, recall, latency, and citations."""

    def __init__(self, analyzer: Any) -> None:
        self.analyzer = analyzer

    def run_suite(self) -> Dict[str, Any]:
        """Run the QA benchmark suite and calculate metric statistics."""
        results = []
        total_latency = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_citation_accuracy = 0.0

        for item in BENCHMARK_SUITE:
            query = item["query"]
            expected = item["expected_files"]

            start_time = time.time()
            try:
                res = self.analyzer.ask_question(query)
                latency = time.time() - start_time
            except Exception as exc:
                logger.error("Benchmark query failed: %s", exc)
                continue

            sources = res.get("sources", [])
            cited_files = {src.get("file") for src in sources if src.get("file")}

            # Calculate metrics
            tp = len(cited_files.intersection(expected))
            precision = tp / len(cited_files) if cited_files else 0.0
            recall = tp / len(expected) if expected else 0.0
            
            # Simple citation accuracy check (at least one valid expected citation)
            citation_accuracy = 1.0 if tp > 0 else 0.0

            results.append({
                "query": query,
                "latency": latency,
                "precision": precision,
                "recall": recall,
                "citation_accuracy": citation_accuracy,
                "cited": list(cited_files),
                "expected": expected,
            })

            total_latency += latency
            total_precision += precision
            total_recall += recall
            total_citation_accuracy += citation_accuracy

        num_queries = len(BENCHMARK_SUITE)
        summary = {
            "num_queries": num_queries,
            "avg_latency": total_latency / num_queries if num_queries else 0.0,
            "avg_precision": total_precision / num_queries if num_queries else 0.0,
            "avg_recall": total_recall / num_queries if num_queries else 0.0,
            "avg_citation_accuracy": total_citation_accuracy / num_queries if num_queries else 0.0,
            "queries": results,
        }
        return summary
