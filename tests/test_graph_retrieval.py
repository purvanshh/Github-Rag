import unittest
from unittest.mock import MagicMock

from retrieval.graph_aware_retriever import GraphAwareRetriever
from retrieval.reranker import RerankResult


class TestGraphRetrieval(unittest.TestCase):
    def test_retrieve_pipeline(self):
        # Mock embedder
        embedder = MagicMock()
        embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]

        # Mock vector store
        vector_store = MagicMock()
        initial_results = [
            {
                "id": "chunk1",
                "document": "def foo(): pass",
                "metadata": {"file_path": "foo.py", "fqn": "foo.foo", "symbol_name": "foo"},
            }
        ]
        expanded_file_results = [
            {
                "id": "chunk2",
                "document": "def bar(): pass",
                "metadata": {"file_path": "bar.py", "fqn": "bar.bar", "symbol_name": "bar"},
            }
        ]
        expanded_func_results = [
            {
                "id": "chunk3",
                "document": "def baz(): pass",
                "metadata": {"file_path": "baz.py", "fqn": "baz.baz", "symbol_name": "baz"},
            }
        ]

        def mock_query(embedding, top_k, where=None):
            if where is None:
                return initial_results
            elif "file_path" in where:
                return expanded_file_results
            elif "fqn" in where:
                return expanded_func_results
            return []

        vector_store.query.side_effect = mock_query

        # Mock dependency graph
        dependency_graph = MagicMock()
        dependency_graph.get_dependencies.return_value = ["bar.py"]
        dependency_graph.get_dependents.return_value = []

        # Mock call graph
        call_graph = MagicMock()
        call_graph.get_callers.return_value = ["baz.baz"]
        call_graph.get_callees.return_value = []

        # Mock reranker
        reranker = MagicMock()
        rerank_results = [
            RerankResult("def foo(): pass", {"file_path": "foo.py", "fqn": "foo.foo"}, 0.9),
            RerankResult("def bar(): pass", {"file_path": "bar.py", "fqn": "bar.bar"}, 0.8),
            RerankResult("def baz(): pass", {"file_path": "baz.py", "fqn": "baz.baz"}, 0.7),
        ]
        reranker.rerank.return_value = rerank_results

        # Instantiate retriever
        retriever = GraphAwareRetriever(
            embedder=embedder,
            vector_store=vector_store,
            dependency_graph=dependency_graph,
            call_graph=call_graph,
            top_k_initial=5,
            top_k_expanded=10,
            top_k_final=3,
            reranker=reranker,
        )

        results = retriever.retrieve("test query")

        # Assertions
        # 1. Embedder was called
        embedder.embed_texts.assert_called_with(["test query"])

        # 2. Vector store query was called multiple times (for initial search + expanded nodes)
        self.assertGreaterEqual(vector_store.query.call_count, 3)

        # 3. Reranker was called with the merged results
        reranker.rerank.assert_called_once()
        merged_arg = reranker.rerank.call_args[0][1]
        self.assertEqual(len(merged_arg), 3)

        # 4. Final output format is correct
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["document"], "def foo(): pass")
        self.assertEqual(results[0]["relevance_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
