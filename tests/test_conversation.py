import unittest
import os
import tempfile
import shutil
from reasoning.conversation_manager import ConversationManager
from reasoning.repo_analyzer import RepoAnalyzer


class TestConversationManager(unittest.TestCase):
    def setUp(self):
        from config import config
        self.test_dir = tempfile.mkdtemp()
        self.old_repos_dir = config.repos_dir
        config.repos_dir = self.test_dir

    def tearDown(self):
        from config import config
        config.repos_dir = self.old_repos_dir
        shutil.rmtree(self.test_dir)

    def test_conversation_persistence(self):
        manager = ConversationManager("test_repo")
        manager.add_message("conv-1", "user", "Hello codebase")
        manager.add_message("conv-1", "assistant", "Hello! How can I help you?")

        history = manager.get_history("conv-1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello codebase")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hello! How can I help you?")


if __name__ == "__main__":
    unittest.main()
