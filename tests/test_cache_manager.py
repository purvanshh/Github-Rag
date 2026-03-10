import unittest
import os
import tempfile
import shutil
from indexing.cache_manager import LocalCacheManager


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.environ["REPOS_DIR"] = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if "REPOS_DIR" in os.environ:
            del os.environ["REPOS_DIR"]

    def test_cache_set_get(self):
        cache = LocalCacheManager("test_repo")
        cache.set("embedding", "test_text", [0.1, 0.2, 0.3])
        
        val = cache.get("embedding", "test_text")
        self.assertEqual(val, [0.1, 0.2, 0.3])

        # Test overwrite
        cache.set("embedding", "test_text", [0.4, 0.5, 0.6])
        val = cache.get("embedding", "test_text")
        self.assertEqual(val, [0.4, 0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
