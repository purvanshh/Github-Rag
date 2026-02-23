import unittest
import os
import tempfile
import shutil
from ingestion.github_integration import GitHubIntegrationEngine
from git import Repo


class TestGitHubIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.repo_dir = os.path.join(self.test_dir, "test_repo")
        os.makedirs(self.repo_dir)

        # Initialize mock git repository for commit parsing
        self.repo = Repo.init(self.repo_dir)
        
        # Write mock README
        self.readme_path = os.path.join(self.repo_dir, "README.md")
        with open(self.readme_path, "w", encoding="utf-8") as f:
            f.write("# Mock Repository\nThis is a mock repository for testing.")

        # Commit README
        self.repo.index.add(["README.md"])
        self.repo.index.commit("Initial commit adding README")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_github_metadata_ingestion(self):
        engine = GitHubIntegrationEngine()
        chunks = engine.ingest_metadata(self.repo_dir, "test_repo", "https://github.com/test/test_repo")

        # We expect: 1 README chunk, at least 1 commit chunk, and mock issues (3 of them)
        self.assertGreaterEqual(len(chunks), 5)
        
        # Check README chunk
        readme = next(c for c in chunks if c.symbol_type == "readme")
        self.assertEqual(readme.file_path, "README.md")
        self.assertIn("Mock Repository", readme.content)

        # Check Commit chunk
        commits = [c for c in chunks if c.symbol_type == "github_commit"]
        self.assertGreaterEqual(len(commits), 1)
        self.assertIn("Initial commit adding README", commits[0].content)

        # Check Issue chunks
        issues = [c for c in chunks if c.symbol_type == "github_issue"]
        self.assertEqual(len(issues), 3)
        self.assertIn("Fix token authentication middleware", issues[0].content)


if __name__ == "__main__":
    unittest.main()
