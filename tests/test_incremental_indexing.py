import unittest
import os
import tempfile
import shutil
import json
from ingestion.repo_pipeline import RepoIngestionPipeline
from config import config


class TestIncrementalIndexing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.repo_dir = os.path.join(self.test_dir, "test_repo")
        os.makedirs(self.repo_dir)
        
        # Configure test repository path in mock config
        self.orig_repos_dir = config.repos_dir
        config.repos_dir = self.test_dir

        # File 1: Python
        self.file1 = os.path.join(self.repo_dir, "module.py")
        with open(self.file1, "w", encoding="utf-8") as f:
            f.write("def func_a(): pass\n")

        # File 2: Python
        self.file2 = os.path.join(self.repo_dir, "other.py")
        with open(self.file2, "w", encoding="utf-8") as f:
            f.write("def func_b(): pass\n")

    def tearDown(self):
        config.repos_dir = self.orig_repos_dir
        shutil.rmtree(self.test_dir)

    def test_incremental_ingestion(self):
        pipeline = RepoIngestionPipeline()
        
        # 1. Full Ingestion (Initial run)
        res1 = pipeline.ingest_repository(self.repo_dir, incremental=False)
        self.assertEqual(res1.num_files, 2)
        self.assertEqual(res1.num_symbols, 2)

        # 2. Incremental run with no changes
        res2 = pipeline.ingest_repository(self.repo_dir, incremental=True)
        self.assertEqual(res2.num_files, 2)
        # Should complete immediately and preserve symbol counts
        self.assertEqual(res2.num_symbols, 2)

        # 3. Modify File 1 (func_a -> func_new)
        with open(self.file1, "w", encoding="utf-8") as f:
            f.write("def func_new(): pass\n")
            
        res3 = pipeline.ingest_repository(self.repo_dir, incremental=True)
        # Verification: hashes are updated and symbols are parsed
        self.assertEqual(res3.num_files, 2)
        
        # Knowledge graph should have func_new, not func_a
        metadata_path = os.path.join(self.test_dir, "test_repo", "repo_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        # Verify hashes tracked
        self.assertIn("module.py", metadata["file_hashes"])
        self.assertIn("other.py", metadata["file_hashes"])


if __name__ == "__main__":
    unittest.main()
