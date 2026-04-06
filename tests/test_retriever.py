import unittest
from unittest.mock import MagicMock
from retrieval.retriever import CodeRetriever, HybridCodeRetriever


class TestRetriever(unittest.TestCase):
    def test_code_retriever(self):
        embedder = MagicMock()
        embedder.embed_texts.return_value = [[0.1, 0.2]]
        
        vector_store = MagicMock()
        vector_store.query.return_value = [
            {"document": "def test(): pass", "metadata": {"file_path": "a.py", "symbol_name": "test", "symbol_type": "function", "start_line": 1, "end_line": 2}}
        ]
        
        retriever = CodeRetriever(embedder, vector_store, top_k=2)
        res = retriever.retrieve("hello")
        self.assertEqual(len(res), 1)
        
        context = retriever.retrieve_with_context("hello")
        self.assertIn("a.py", context)

    def test_hybrid_code_retriever(self):
        embedder = MagicMock()
        embedder.embed_texts.return_value = [[0.1, 0.2]]
        
        vector_store = MagicMock()
        vector_store.query.return_value = [
            {"document": "def test(): pass", "metadata": {"file_path": "a.py", "symbol_name": "test", "symbol_type": "function", "start_line": 1, "end_line": 2}}
        ]
        
        reranker = MagicMock()
        mock_result = MagicMock()
        mock_result.document = "def test(): pass"
        mock_result.metadata = {"file_path": "a.py", "symbol_name": "test", "symbol_type": "function", "start_line": 1, "end_line": 2}
        mock_result.relevance_score = 0.95
        reranker.rerank.return_value = [mock_result]
        
        retriever = HybridCodeRetriever(embedder, vector_store, reranker=reranker)
        res = retriever.retrieve("hello")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["relevance_score"], 0.95)


if __name__ == "__main__":
    unittest.main()
