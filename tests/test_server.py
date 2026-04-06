import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.server import app
from config import config
from api.security import sign_jwt


class TestApiServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        self.old_security = config.security_enabled

    def tearDown(self):
        config.security_enabled = self.old_security

    def test_basic_endpoints(self):
        config.security_enabled = False
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("app", resp.json())

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/favicon.ico")
        self.assertEqual(resp.status_code, 204)

        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)

    def test_security_failures(self):
        config.security_enabled = True
        
        # Missing token
        resp = self.client.post("/ingest", json={"repo_url": "https://github.com/user/repo"})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Missing authorization header", resp.json()["detail"])
        
        # Invalid token
        resp = self.client.post("/ingest", json={"repo_url": "https://github.com/user/repo"}, headers={"Authorization": "Bearer invalid_token"})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid token or API key", resp.json()["detail"])

        # Valid Bearer token
        token = sign_jwt({"user": "test-user"})
        with patch("api.server.RepoIngestionPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_result = MagicMock()
            mock_result.repo_name = "test-repo"
            mock_pipeline.ingest_repository.return_value = mock_result
            mock_pipeline_cls.return_value = mock_pipeline
            
            resp = self.client.post("/ingest", json={"repo_url": "https://github.com/user/repo"}, headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(resp.status_code, 200)

        # Rate limited
        with patch("api.server.is_rate_limited", return_value=True):
            resp = self.client.post("/ingest", json={"repo_url": "https://github.com/user/repo"}, headers={"Authorization": "test-api-key-12345"})
            self.assertEqual(resp.status_code, 429)
            self.assertIn("Rate limit exceeded", resp.json()["detail"])

    @patch("api.server.RepoIngestionPipeline")
    def test_ingest_routes(self, mock_pipeline_cls):
        config.security_enabled = False
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.repo_name = "test-repo"
        mock_pipeline.ingest_repository.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        resp = self.client.post("/ingest", json={"repo_url": "https://github.com/user/test-repo"})
        self.assertEqual(resp.status_code, 200)

        # Test ingestion exception
        mock_pipeline.ingest_repository.side_effect = Exception("Ingest error")
        resp = self.client.post("/ingest", json={"repo_url": "https://github.com/user/test-repo"})
        self.assertEqual(resp.status_code, 500)

    @patch("api.server._get_repo_analyzer")
    def test_query_routes(self, mock_get_analyzer):
        config.security_enabled = False
        mock_analyzer = MagicMock()
        mock_analyzer.ask_question.return_value = {"answer": "QA", "sources": []}
        mock_analyzer.ask_agentic.return_value = {"answer": "Agentic", "sources": []}
        mock_analyzer.ask_question_stream.return_value = [{"type": "token", "text": "hello"}]
        mock_get_analyzer.return_value = mock_analyzer

        # Post standard query
        resp = self.client.post("/query", json={"repo": "test-repo", "query": "hello"})
        self.assertEqual(resp.status_code, 200)

        # Post agentic query
        resp = self.client.post("/query", json={"repo": "test-repo", "query": "hello", "agentic": True})
        self.assertEqual(resp.status_code, 200)

        # Post streaming query
        resp = self.client.post("/query/stream", json={"repo": "test-repo", "query": "hello"})
        self.assertEqual(resp.status_code, 200)

    @patch("api.server._get_repo_analyzer")
    def test_repo_endpoints(self, mock_get_analyzer):
        config.security_enabled = False
        mock_analyzer = MagicMock()
        mock_analyzer.get_repo_overview.return_value = {
            "directory_tree": "Tree",
            "architecture_summary": "summary_text"
        }
        mock_analyzer.get_file_dependencies.return_value = ["file2.py"]
        mock_analyzer.find_function_usage.return_value = {"callers": [], "callees": []}
        mock_analyzer.get_dependency_chart.return_value = "flowchart"
        mock_analyzer.get_class_hierarchy.return_value = "classDiagram"
        mock_analyzer.get_sequence_chart.return_value = "sequenceDiagram"
        mock_analyzer.explain_file_difficulty.return_value = "explanation"
        
        mock_explain_engine = MagicMock()
        mock_explain_engine.explain.return_value = "symbol explanation"
        mock_analyzer.explanation_engine = mock_explain_engine
        
        mock_get_analyzer.return_value = mock_analyzer

        resp = self.client.get("/repo/test-repo/overview")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/repo/test-repo/dependencies/file1.py")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/repo/test-repo/function/func")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/repo/test-repo/diagrams/dependency")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/repo/test-repo/diagrams/class")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/repo/test-repo/diagrams/sequence/func")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/repo/test-repo/explain/file/file1.py")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/repo/test-repo/explain/symbol/func")
        self.assertEqual(resp.status_code, 200)

    @patch("api.server._get_repo_analyzer")
    def test_benchmark_endpoint(self, mock_get_analyzer):
        config.security_enabled = False
        mock_analyzer = MagicMock()
        mock_analyzer.run_qa_benchmark.return_value = {"num_queries": 3}
        mock_get_analyzer.return_value = mock_analyzer

        resp = self.client.post("/repo/test-repo/benchmark")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["num_queries"], 3)

    def test_security_helpers(self):
        from api.security import sign_jwt, verify_jwt, is_rate_limited, validate_repo_url
        
        # JWT verify
        self.assertIsNone(verify_jwt("invalid.token.here"))
        self.assertIsNone(verify_jwt("a.b.c"))
        
        # Expired token
        expired_token = sign_jwt({"user": "test", "exp": 0})
        self.assertIsNone(verify_jwt(expired_token))
        
        # Rate limit
        key = "test-rate-limit-key"
        for _ in range(60):
            self.assertFalse(is_rate_limited(key))
        self.assertTrue(is_rate_limited(key))
        
        # URL validations
        self.assertTrue(validate_repo_url("https://github.com/a/b"))
        self.assertTrue(validate_repo_url("git@github.com:a/b.git"))
        self.assertFalse(validate_repo_url("invalid-url!@#"))

    @patch("api.server._get_repo_analyzer")
    def test_repo_endpoints_exceptions(self, mock_get_analyzer):
        config.security_enabled = False
        mock_get_analyzer.side_effect = Exception("Mocked error")
        
        for route in [
            "/repo/test-repo/overview",
            "/repo/test-repo/dependencies/file1.py",
            "/repo/test-repo/function/func",
            "/repo/test-repo/diagrams/dependency",
            "/repo/test-repo/diagrams/class",
            "/repo/test-repo/diagrams/sequence/func",
            "/repo/test-repo/explain/file/file1.py",
            "/repo/test-repo/explain/symbol/func",
            "/repo/test-repo/benchmark"
        ]:
            if route == "/repo/test-repo/benchmark":
                resp = self.client.post(route)
            else:
                resp = self.client.get(route)
            self.assertEqual(resp.status_code, 500)

        # Test query exception paths
        resp = self.client.post("/query", json={"repo": "test-repo", "query": "hello"})
        self.assertEqual(resp.status_code, 500)
        
        resp = self.client.post("/query/stream", json={"repo": "test-repo", "query": "hello"})
        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()
