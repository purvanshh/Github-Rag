import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

from config import config
from reasoning.repo_architect import AIRepoArchitect
from reasoning.repo_analyzer import RepoAnalyzer
from fastapi.testclient import TestClient
from api.server import app


class TestRepoArchitect(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.test_dir, "test_code.py")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("def foo(): pass\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_gemini_architect_success(self):
        analyzer = MagicMock()
        analyzer._kg.get_most_connected.return_value = [("a.py", 3)]
        analyzer._directory_tree = "tree"
        analyzer.get_architecture_summary.return_value = {"summary": "Gemini arch summary."}
        
        old_provider = config.llm_provider
        config.llm_provider = "gemini"
        old_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "sk-mock-gemini-key"
        try:
            architect = AIRepoArchitect(analyzer)
            mock_resp = MagicMock()
            mock_resp.text = "Gemini architect report suggestions."
            architect._gemini_model.generate_content = MagicMock(return_value=mock_resp)

            res = architect.generate_architecture_report()
            self.assertEqual(res, "Gemini architect report suggestions.")

            # Test failure branch
            architect._gemini_model.generate_content.side_effect = Exception("Gemini error")
            res_fail = architect.generate_architecture_report()
            self.assertEqual(res_fail, "Failed to generate architecture report.")
        finally:
            config.llm_provider = old_provider
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
            else:
                os.environ.pop("GEMINI_API_KEY", None)

    def test_openai_architect_success(self):
        analyzer = MagicMock()
        analyzer._kg.get_most_connected.return_value = [("a.py", 3)]
        analyzer._directory_tree = "tree"
        analyzer.get_architecture_summary.return_value = {"summary": "OpenAI arch summary."}
        
        old_provider = config.llm_provider
        config.llm_provider = "openai"
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-mock-openai-key"
        try:
            architect = AIRepoArchitect(analyzer)
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "OpenAI architect report suggestions."
            mock_resp.choices = [mock_choice]
            architect._openai_client.chat.completions.create = MagicMock(return_value=mock_resp)

            res = architect.generate_architecture_report()
            self.assertEqual(res, "OpenAI architect report suggestions.")

            # Test failure branch
            architect._openai_client.chat.completions.create.side_effect = Exception("OpenAI error")
            res_fail = architect.generate_architecture_report()
            self.assertEqual(res_fail, "Failed to generate architecture report.")
        finally:
            config.llm_provider = old_provider
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    @patch("ingestion.parse_code.parse_directory")
    def test_repo_analyzer_architect(self, mock_parse):
        mock_parse.return_value = []
        analyzer = RepoAnalyzer(os.path.basename(self.test_dir), repos_root=os.path.dirname(self.test_dir))
        
        mock_architect = MagicMock()
        mock_architect.generate_architecture_report.return_value = "Mocked architect report."
        analyzer._repo_architect = mock_architect

        res = analyzer.generate_architecture_report()
        self.assertEqual(res, "Mocked architect report.")

    @patch("api.server._get_repo_analyzer")
    def test_architect_endpoint(self, mock_get_analyzer):
        client = TestClient(app, raise_server_exceptions=False)
        config.security_enabled = False

        mock_analyzer = MagicMock()
        mock_analyzer.generate_architecture_report.return_value = "REST endpoint architect report."
        mock_get_analyzer.return_value = mock_analyzer

        resp = client.get("/repo/test-repo/architecture-report")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["report"], "REST endpoint architect report.")


if __name__ == "__main__":
    unittest.main()
