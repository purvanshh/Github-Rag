import unittest
import os
import tempfile
import shutil
import json
from reasoning.repo_memory import RepositoryMemory
from reasoning.repo_analyzer import RepoAnalyzer
from config import config


class TestRepoMemory(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.repo_dir = os.path.join(self.test_dir, "test_repo")
        os.makedirs(self.repo_dir)

        # Mock config's repos directory
        self.orig_repos_dir = config.repos_dir
        config.repos_dir = self.test_dir

        # Write simple python file so that RepoAnalyzer has something to parse
        self.file1 = os.path.join(self.repo_dir, "app.py")
        with open(self.file1, "w", encoding="utf-8") as f:
            f.write("def hello(): pass\n")

    def tearDown(self):
        config.repos_dir = self.orig_repos_dir
        shutil.rmtree(self.test_dir)

    def test_repository_memory_operations(self):
        # 1. Initialize Memory Directly
        memory = RepositoryMemory("test_repo")
        
        # Test Default Values
        self.assertEqual(memory.profile["primary_language"], "Unknown")
        self.assertGreaterEqual(len(memory.faqs), 2)
        
        # Update Profile
        memory.update_profile({
            "primary_language": "Python",
            "tech_stack": ["Flask", "ChromaDB", "Gemini API"],
            "description": "Test Project Memory Profile"
        })
        
        # Add FAQ
        memory.add_faq("How do I ingest code?", "Use RepoIngestionPipeline class.")

        # Re-instantiate to verify persistence on disk
        memory2 = RepositoryMemory("test_repo")
        self.assertEqual(memory2.profile["primary_language"], "Python")
        self.assertEqual(memory2.profile["description"], "Test Project Memory Profile")
        self.assertIn("Flask", memory2.profile["tech_stack"])
        
        # Verify custom FAQ loaded
        custom_faq = next((faq for faq in memory2.faqs if faq["question"] == "How do I ingest code?"), None)
        self.assertIsNotNone(custom_faq)
        self.assertEqual(custom_faq["answer"], "Use RepoIngestionPipeline class.")

        # 2. Verify Markdown Context Generation
        context = memory2.get_memory_context()
        self.assertIn("## Repository Stack Profile", context)
        self.assertIn("**Primary Language**: Python", context)
        self.assertIn("Flask, ChromaDB, Gemini API", context)
        self.assertIn("How do I ingest code?", context)

    def test_repo_analyzer_memory_integration(self):
        # Verify RepoAnalyzer exposes these methods correctly
        analyzer = RepoAnalyzer(self.repo_dir)
        
        analyzer.update_profile({"primary_language": "Go"})
        analyzer.add_faq("How to compile?", "Run go build.")
        
        context = analyzer.get_memory_context()
        self.assertIn("**Primary Language**: Go", context)
        self.assertIn("How to compile?", context)


if __name__ == "__main__":
    unittest.main()
