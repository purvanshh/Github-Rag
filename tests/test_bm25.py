import unittest
from retrieval.bm25 import BM25Retriever


class TestBM25(unittest.TestCase):
    def test_tokenize(self):
        retriever = BM25Retriever()
        
        # Test camelCase splitting
        tokens = retriever.tokenize("myAwesomeVariable")
        self.assertEqual(tokens, ["my", "awesome", "variable"])
        
        # Test snake_case splitting
        tokens = retriever.tokenize("my_awesome_variable")
        self.assertEqual(tokens, ["my", "awesome", "variable"])
        
        # Test basic punctuation and split
        tokens = retriever.tokenize("def format_log(self, text):")
        self.assertEqual(tokens, ["def", "format", "log", "self", "text"])

    def test_fit_and_query(self):
        docs = [
            {"id": "doc1", "document": "import os\ndef write_file(): pass"},
            {"id": "doc2", "document": "import sys\ndef query_database(): pass"},
            {"id": "doc3", "document": "from indexing.vector_store import ChromaVectorStore"},
        ]
        
        retriever = BM25Retriever()
        retriever.fit(docs)
        
        # Querying for "database" should rank doc2 first
        results = retriever.query("database", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc2")
        self.assertGreater(results[0]["relevance_score"], 0)

        # Querying for "os" should rank doc1 first
        results = retriever.query("os", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc1")


if __name__ == "__main__":
    unittest.main()
