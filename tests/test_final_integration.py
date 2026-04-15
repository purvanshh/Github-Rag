import unittest
from unittest.mock import patch, MagicMock
import sys
import os

from main import main


class TestFinalIntegration(unittest.TestCase):
    @patch("main.cmd_ingest")
    def test_cli_ingest(self, mock_cmd):
        with patch.object(sys, "argv", ["main.py", "ingest", "https://github.com/mock/repo"]):
            main()
            mock_cmd.assert_called_once()

    @patch("main.cmd_query")
    def test_cli_query(self, mock_cmd):
        with patch.object(sys, "argv", ["main.py", "query", "where is the loop?"]):
            main()
            mock_cmd.assert_called_once()

    @patch("main.cmd_serve")
    def test_cli_serve(self, mock_cmd):
        with patch.object(sys, "argv", ["main.py", "serve"]):
            main()
            mock_cmd.assert_called_once()

    @patch("main.cmd_review")
    def test_cli_review(self, mock_cmd):
        with patch.object(sys, "argv", ["main.py", "review", "app.py", "--repo", "test-repo"]):
            main()
            mock_cmd.assert_called_once()

    @patch("main.cmd_architect_report")
    def test_cli_architect(self, mock_cmd):
        with patch.object(sys, "argv", ["main.py", "architect-report", "--repo", "test-repo"]):
            main()
            mock_cmd.assert_called_once()

    @patch("main.cmd_autonomous_run")
    def test_cli_autonomous(self, mock_cmd):
        with patch.object(sys, "argv", ["main.py", "autonomous-run", "--repo", "test-repo"]):
            main()
            mock_cmd.assert_called_once()

    @patch("reasoning.repo_analyzer.RepoAnalyzer")
    def test_cmd_review_execution(self, mock_analyzer_cls):
        mock_analyzer = MagicMock()
        mock_analyzer.run_code_review.return_value = "review text"
        mock_analyzer_cls.return_value = mock_analyzer

        args = MagicMock()
        args.repo = "test-repo"
        args.file_path = "app.py"

        from main import cmd_review
        with patch("builtins.print") as mock_print:
            cmd_review(args)
            mock_print.assert_any_call("review text")

    @patch("reasoning.repo_analyzer.RepoAnalyzer")
    def test_cmd_architect_execution(self, mock_analyzer_cls):
        mock_analyzer = MagicMock()
        mock_analyzer.generate_architecture_report.return_value = "arch text"
        mock_analyzer_cls.return_value = mock_analyzer

        args = MagicMock()
        args.repo = "test-repo"

        from main import cmd_architect_report
        with patch("builtins.print") as mock_print:
            cmd_architect_report(args)
            mock_print.assert_any_call("arch text")

    @patch("reasoning.repo_analyzer.RepoAnalyzer")
    def test_cmd_autonomous_execution(self, mock_analyzer_cls):
        mock_analyzer = MagicMock()
        mock_analyzer.run_autonomous_agent.return_value = {
            "status": "Success",
            "report_path": "path/README_ai_scan.md"
        }
        mock_analyzer_cls.return_value = mock_analyzer

        args = MagicMock()
        args.repo = "test-repo"

        from main import cmd_autonomous_run
        with patch("builtins.print") as mock_print:
            cmd_autonomous_run(args)
            mock_print.assert_any_call("Report saved to: path/README_ai_scan.md")


if __name__ == "__main__":
    unittest.main()
