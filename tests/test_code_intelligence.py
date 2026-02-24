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

    def test_code_intelligence_queries(self):
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


if __name__ == "__main__":
    unittest.main()
