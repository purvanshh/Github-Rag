import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

from config import config
from reasoning.code_reviewer import AICodeReviewer
from reasoning.repo_analyzer import RepoAnalyzer
from fastapi.testclient import TestClient
from api.server import app


class TestCodeReviewer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.test_dir, "test_code.py")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("def bad_func():\n    unused = 1\n    pass\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_gemini_reviewer_success(self):
        retriever = MagicMock()
        old_provider = config.llm_provider
        config.llm_provider = "gemini"
        old_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "sk-mock-gemini-key"
        try:
            reviewer = AICodeReviewer()
            mock_resp = MagicMock()
            mock_resp.text = "Gemini review suggestions."
            reviewer._gemini_model.generate_content = MagicMock(return_value=mock_resp)

            res = reviewer.review_code("def foo(): pass", "foo.py")
            self.assertEqual(res, "Gemini review suggestions.")

            # Test failure branch
            reviewer._gemini_model.generate_content.side_effect = Exception("Gemini error")
            res_fail = reviewer.review_code("def foo(): pass")
            self.assertEqual(res_fail, "Failed to run code review.")
        finally:
            config.llm_provider = old_provider
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
            else:
                os.environ.pop("GEMINI_API_KEY", None)

    def test_openai_reviewer_success(self):
        retriever = MagicMock()
        old_provider = config.llm_provider
        config.llm_provider = "openai"
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-mock-openai-key"
        try:
            reviewer = AICodeReviewer()
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "OpenAI review suggestions."
            mock_resp.choices = [mock_choice]
            reviewer._openai_client.chat.completions.create = MagicMock(return_value=mock_resp)

            res = reviewer.review_code("def foo(): pass", "foo.py")
            self.assertEqual(res, "OpenAI review suggestions.")

            # Test failure branch
            reviewer._openai_client.chat.completions.create.side_effect = Exception("OpenAI error")
            res_fail = reviewer.review_code("def foo(): pass")
            self.assertEqual(res_fail, "Failed to run code review.")
        finally:
            config.llm_provider = old_provider
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    @patch("ingestion.parse_code.parse_directory")
    def test_repo_analyzer_review(self, mock_parse):
        mock_parse.return_value = []
        analyzer = RepoAnalyzer(os.path.basename(self.test_dir), repos_root=os.path.dirname(self.test_dir))
        
        # Mock code reviewer
        mock_reviewer = MagicMock()
        mock_reviewer.review_code.return_value = "Mocked review report."
        analyzer._code_reviewer = mock_reviewer

        res = analyzer.run_code_review("test_code.py")
        self.assertEqual(res, "Mocked review report.")

        # Test nonexistent file
        res_nonexistent = analyzer.run_code_review("nonexistent.py")
        self.assertIn("not found", res_nonexistent)

    @patch("api.server._get_repo_analyzer")
    def test_review_endpoint(self, mock_get_analyzer):
        client = TestClient(app, raise_server_exceptions=False)
        config.security_enabled = False

        mock_analyzer = MagicMock()
        mock_analyzer.run_code_review.return_value = "REST endpoint review response."
        mock_get_analyzer.return_value = mock_analyzer

        resp = client.post("/repo/test-repo/review", json={"file_path": "main.py"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["review"], "REST endpoint review response.")


if __name__ == "__main__":
    unittest.main()
