import os
import unittest
from metadata_utils import (
    normalize_repo_id,
    normalize_file_path,
    normalize_fqn,
    normalize_symbol_id,
    module_name_from_path,
)

class TestMetadataNormalization(unittest.TestCase):
    def test_normalize_repo_id(self):
        self.assertEqual(normalize_repo_id("https://github.com/karpathy/nanoGPT"), "nanogpt")
        self.assertEqual(normalize_repo_id("https://github.com/karpathy/nanoGPT.git"), "nanogpt")
        self.assertEqual(normalize_repo_id("nanoGPT"), "nanogpt")
        self.assertEqual(normalize_repo_id(""), "unknown")

    def test_normalize_file_path(self):
        # absolute path
        repo = "/Users/purvansh/Desktop/Projects/Github-Rag"
        path = "/Users/purvansh/Desktop/Projects/Github-Rag/graphs/call_graph.py"
        self.assertEqual(normalize_file_path(path, repo), "graphs/call_graph.py")

        # Windows paths
        win_path = "graphs\\call_graph.py"
        self.assertEqual(normalize_file_path(win_path), "graphs/call_graph.py")

        # leading dot slashes
        self.assertEqual(normalize_file_path("./graphs/call_graph.py"), "graphs/call_graph.py")

    def test_module_name_from_path(self):
        repo = "/Users/purvansh/Desktop/Projects/Github-Rag"
        self.assertEqual(
            module_name_from_path(repo, "/Users/purvansh/Desktop/Projects/Github-Rag/graphs/call_graph.py"),
            "graphs.call_graph"
        )
        self.assertEqual(
            module_name_from_path(repo, "/Users/purvansh/Desktop/Projects/Github-Rag/graphs/__init__.py"),
            "graphs"
        )

    def test_normalize_fqn(self):
        self.assertEqual(normalize_fqn("graphs.call_graph", "CallGraph", "add_call"), "graphs.call_graph.CallGraph.add_call")
        self.assertEqual(normalize_fqn("graphs.call_graph", None, "build_call_graph"), "graphs.call_graph.build_call_graph")
        self.assertEqual(normalize_fqn("", None, "func"), "func")

    def test_normalize_symbol_id(self):
        self.assertEqual(
            normalize_symbol_id("nanoGPT", "src/train.py", "train.train_loop", 15),
            "nanogpt:src/train.py:train.train_loop:15"
        )

if __name__ == "__main__":
    unittest.main()
