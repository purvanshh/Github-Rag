import unittest
from unittest.mock import MagicMock
from reasoning.query_router import QueryRouter


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.analyzer = MagicMock()
        self.router = QueryRouter(self.analyzer)

    def test_classify_query(self):
        self.assertEqual(self.router.classify_query("give me an overview"), "repo_overview")
        self.assertEqual(self.router.classify_query("what does this repo do"), "architecture")
        self.assertEqual(self.router.classify_query("where is main used"), "function_usage")
        self.assertEqual(self.router.classify_query("dependencies of app.py"), "file_dependencies")
        self.assertEqual(self.router.classify_query("explain file main.py"), "file_explanation")
        self.assertEqual(self.router.classify_query("generic coding question"), "code_question")

    def test_extract_function_name(self):
        self.assertEqual(self.router._extract_function_name("where is get_user used?"), "get_user")
        self.assertEqual(self.router._extract_function_name("who calls run_task"), "run_task")
        self.assertIsNone(self.router._extract_function_name("just a text"))

    def test_extract_file_path(self):
        self.assertEqual(self.router._extract_file_path("explain file src/app.py"), "src/app.py")
        self.assertEqual(self.router._extract_file_path("dependencies of api/server.py"), "api/server.py")
        self.assertEqual(self.router._extract_file_path("what does file utils.py do?"), "utils.py")

    def test_route_query(self):
        self.analyzer.find_function_usage.return_value = {"callers": [], "callees": []}
        self.analyzer.get_file_dependencies.return_value = []
        self.analyzer.explain_file.return_value = {"explanation": "simple"}
        self.analyzer.get_architecture_summary.return_value = {}
        self.analyzer.get_repo_overview.return_value = {}
        self.analyzer.ask_question.return_value = {}

        self.router.route_query("where is main used")
        self.analyzer.find_function_usage.assert_called_once_with("main")

        self.router.route_query("dependencies of app.py")
        self.analyzer.get_file_dependencies.assert_called_once_with("app.py")

        self.router.route_query("explain file app.py")
        self.analyzer.explain_file.assert_called_once_with("app.py")

        self.router.route_query("what does this repo do")
        self.analyzer.get_architecture_summary.assert_called_once()

        self.router.route_query("summarize this repo")
        self.analyzer.get_repo_overview.assert_called_once()


if __name__ == "__main__":
    unittest.main()
