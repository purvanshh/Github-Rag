import sys
from unittest.mock import MagicMock

# Mock sentence_transformers before any imports to prevent load of corrupt torch binaries
mock_sentence_transformers = MagicMock()
sys.modules["sentence_transformers"] = mock_sentence_transformers

import unittest
from retrieval.reranker import Reranker, RerankResult


class TestReranker(unittest.TestCase):
    def test_reranker_prediction(self):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.8, 0.4]
        mock_sentence_transformers.CrossEncoder.return_value = mock_encoder

        reranker = Reranker()
        results = [
            {"document": "doc1", "metadata": {"id": 1}},
            {"document": "doc2", "metadata": {"id": 2}}
        ]
        
        reranked = reranker.rerank("query", results, top_k=2)
        
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].document, "doc1")
        self.assertEqual(reranked[0].relevance_score, 0.8)

        # Test empty results
        self.assertEqual(reranker.rerank("query", []), [])


if __name__ == "__main__":
    unittest.main()
