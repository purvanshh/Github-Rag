import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

from config import config
from reasoning.autonomous_agent import AutonomousRepositoryAgent
from reasoning.repo_analyzer import RepoAnalyzer
from fastapi.testclient import TestClient
from api.server import app


class TestAutonomousAgent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.test_dir, "test_file.py")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("print('hello')\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_autonomous_scan_success(self):
        analyzer = MagicMock()
        analyzer.repo_path = self.test_dir
        analyzer.generate_architecture_report.return_value = "Mock architecture report content."
        analyzer.run_code_review.return_value = "Mock file review content."
        analyzer._kg.get_most_connected.return_value = [("test_file.py", 5)]

        agent = AutonomousRepositoryAgent(analyzer)
        res = agent.run_autonomous_scan()

        self.assertEqual(res["status"], "Success")
        self.assertIn("test_file.py", res["reviewed_files"])
        self.assertTrue(os.path.exists(res["report_path"]))

        # Verify content written to README_ai_scan.md
        with open(res["report_path"], "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("🤖 Autonomous AI Repository Scan Report", content)
        self.assertIn("Mock architecture report content.", content)
        self.assertIn("Mock file review content.", content)

    @patch("ingestion.parse_code.parse_directory")
    def test_repo_analyzer_autonomous(self, mock_parse):
        mock_parse.return_value = []
        analyzer = RepoAnalyzer(os.path.basename(self.test_dir), repos_root=os.path.dirname(self.test_dir))
        
        mock_agent = MagicMock()
        mock_agent.run_autonomous_scan.return_value = {"status": "Success"}
        analyzer._autonomous_agent = mock_agent

        res = analyzer.run_autonomous_agent()
        self.assertEqual(res["status"], "Success")

    @patch("api.server._get_repo_analyzer")
    def test_autonomous_endpoint(self, mock_get_analyzer):
        client = TestClient(app, raise_server_exceptions=False)
        config.security_enabled = False

        mock_analyzer = MagicMock()
        mock_analyzer.run_autonomous_agent.return_value = {"status": "Success", "report": "doc"}
        mock_get_analyzer.return_value = mock_analyzer

        resp = client.post("/repo/test-repo/autonomous-run")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "Success")


if __name__ == "__main__":
    unittest.main()
