import unittest
from unittest.mock import MagicMock
from evaluation.benchmark import RepositoryQABenchmark


class TestQABenchmark(unittest.TestCase):
    def test_run_benchmark_suite(self):
        analyzer = MagicMock()
        analyzer.ask_question.return_value = {
            "answer": "Test answer",
            "sources": [
                {"file": "api/server.py", "lines": "1-10"},
                {"file": "config.py", "lines": "5-15"},
            ]
        }

        bench = RepositoryQABenchmark(analyzer)
        summary = bench.run_suite()

        self.assertEqual(summary["num_queries"], 3)
        self.assertGreater(summary["avg_precision"], 0.0)
        self.assertGreater(summary["avg_recall"], 0.0)
        self.assertGreater(summary["avg_citation_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
