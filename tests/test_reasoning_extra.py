import unittest
import tempfile
import os
import shutil
from unittest.mock import MagicMock, patch

from reasoning.architecture_summarizer import (
    build_directory_tree,
    _summarize_dependency_graph,
    generate_architecture_summary,
)
from graphs.dependency_graph import DependencyGraph


class TestReasoningExtra(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.test_dir, "pkg"))
        with open(os.path.join(self.test_dir, "pkg", "a.py"), "w") as f:
            f.write("import os")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_build_directory_tree(self):
        tree = build_directory_tree(self.test_dir)
        self.assertIn("pkg/", tree)
        self.assertIn("a.py", tree)

    def test_summarize_dependency_graph(self):
        dg = DependencyGraph()
        summary = _summarize_dependency_graph(dg)
        self.assertEqual(summary, "No internal Python dependencies detected.")
        
        dg.add_edge("a.py", "b.py", "import b")
        summary_populated = _summarize_dependency_graph(dg)
        self.assertIn("Top dependency hubs", summary_populated)

    @patch("google.generativeai.GenerativeModel")
    @patch("reasoning.architecture_summarizer.build_dependency_graph")
    def test_generate_architecture_summary(self, mock_build_graph, mock_gemini_model):
        mock_graph = MagicMock()
        mock_graph.get_most_connected.return_value = []
        mock_build_graph.return_value = mock_graph

        mock_instance = MagicMock()
        mock_instance.generate_content.return_value.text = "Visual flow architecture mock summary"
        mock_gemini_model.return_value = mock_instance

        from config import config
        old_provider = config.llm_provider
        config.llm_provider = "gemini"
        try:
            summary_dict = generate_architecture_summary(self.test_dir)
            self.assertIn("summary", summary_dict)
            self.assertEqual(summary_dict["summary"], "Visual flow architecture mock summary")
        finally:
            config.llm_provider = old_provider


if __name__ == "__main__":
    unittest.main()
