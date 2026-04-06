import sys
from unittest.mock import MagicMock, patch
sys.modules["sentence_transformers"] = MagicMock()
import unittest
import os
import tempfile
import shutil
from graphs.knowledge_graph import RepositoryKnowledgeGraph
from reasoning.repo_analyzer import RepoAnalyzer


class TestCodeIntelligence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.repo_dir = os.path.join(self.test_dir, "test_repo")
        os.makedirs(self.repo_dir)

        from config import config
        self.orig_repos_dir = config.repos_dir
        config.repos_dir = self.test_dir

        # Write dummy python files
        self.file1 = os.path.join(self.repo_dir, "app.py")
        with open(self.file1, "w", encoding="utf-8") as f:
            f.write("""
import service

class AppService(service.BaseService):
    def run(self):
        service.log_action()
""")

        self.file2 = os.path.join(self.repo_dir, "service.py")
        with open(self.file2, "w", encoding="utf-8") as f:
            f.write("""
class BaseService:
    pass

def log_action():
    pass
""")

    def tearDown(self):
        from config import config
        config.repos_dir = self.orig_repos_dir
        shutil.rmtree(self.test_dir)

    @patch("reasoning.repo_analyzer.generate_architecture_summary")
    def test_code_intelligence_queries(self, mock_gen_arch):
        mock_gen_arch.return_value = {
            "summary": "Mocked architecture summary",
            "model": "gemini",
            "file_tree": "",
            "dependency_hubs": ""
        }
        analyzer = RepoAnalyzer(self.repo_dir)

        # 1. Verification of find_inheritance (AppService inherits from BaseService)
        bases = analyzer.find_inheritance("app.AppService")
        self.assertIn("service.BaseService", bases)

        # 2. Verification of find_implementations (BaseService subclass is AppService)
        impls = analyzer.find_implementations("service.BaseService")
        self.assertIn("app.AppService", impls)

        # 3. Verification of find_references (service.log_action called by AppService.run)
        refs = analyzer.find_references("service.log_action")
        self.assertIn("app.AppService.run", refs)

        # 4. Verification of find_dependency_chains (app.py imports service)
        chains = analyzer.find_dependency_chains(self.file1)
        self.assertGreaterEqual(len(chains), 1)
        self.assertIn("service", chains[0])

        # 5. Verification of get_file_dependencies
        deps = analyzer.get_file_dependencies(self.file1)
        self.assertGreaterEqual(len(deps), 1)

        # 6. Verification of find_function_usage
        usage = analyzer.find_function_usage("log_action")
        self.assertIn("callers", usage)

        # 7. Verification of explain_file (mocked) and QA calls
        mock_gen = MagicMock()
        mock_gen.generate_answer.return_value = {"answer": "mocked explanation"}
        mock_gen.generate_answer_stream.return_value = [{"text": "stream_token"}]
        analyzer._answer_generator = mock_gen
        self.assertEqual(analyzer.explain_file(self.file1)["answer"], "mocked explanation")
        self.assertEqual(analyzer.ask_question("test")["answer"], "mocked explanation")
        self.assertEqual(list(analyzer.ask_question_stream("test")), [{"text": "stream_token"}])

        # 8. Verification of explain_file_difficulty (mocked)
        mock_engine = MagicMock()
        mock_engine.explain.return_value = "beginner explanation"
        analyzer._explanation_engine = mock_engine
        explanation = analyzer.explain_file_difficulty(self.file1, "beginner")
        self.assertEqual(explanation, "beginner explanation")

        # 9. Verification of overview / summaries
        overview = analyzer.get_repo_overview()
        self.assertIn("directory_tree", overview)

        # 10. Nonexistent file explanation difficulty
        self.assertIn("not found", analyzer.explain_file_difficulty("nonexistent.py"))

        # 11. Agentic reasoning and diagrams
        mock_planner = MagicMock()
        mock_planner.create_plan.return_value = []
        mock_planner.execute_plan.return_value = {"answer": "agent_answer"}
        analyzer._query_planner = mock_planner
        self.assertEqual(analyzer.ask_agentic("test")["answer"], "agent_answer")

        mock_arch = MagicMock()
        mock_arch.generate_dependency_chart.return_value = "dep_chart"
        mock_arch.generate_class_hierarchy.return_value = "class_chart"
        mock_arch.generate_sequence_chart.return_value = "seq_chart"
        analyzer._architecture_analyzer = mock_arch
        self.assertEqual(analyzer.get_dependency_chart(), "dep_chart")
        self.assertEqual(analyzer.get_class_hierarchy(), "class_chart")
        self.assertEqual(analyzer.get_sequence_chart("func"), "seq_chart")

        # 12. Path conversions
        rel_path = analyzer._to_relative_path(self.file1)
        self.assertTrue(rel_path.endswith("app.py"))

        # 13. Symbol explanation difficulty
        self.assertIn("not found", analyzer.explain_symbol_difficulty("nonexistent_sym"))
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [{"content": "def log_action(): pass"}]
        analyzer._retriever = mock_retriever
        analyzer._explanation_engine = mock_engine
        mock_engine.explain.return_value = "symbol explanation text"
        self.assertEqual(analyzer.explain_symbol_difficulty("log_action"), "symbol explanation text")

        # 14. Run QA benchmark
        with patch("evaluation.benchmark.RepositoryQABenchmark") as mock_bench_cls:
            mock_bench = MagicMock()
            mock_bench.run_suite.return_value = {"num_queries": 5}
            mock_bench_cls.return_value = mock_bench
            self.assertEqual(analyzer.run_qa_benchmark()["num_queries"], 5)


if __name__ == "__main__":
    unittest.main()
