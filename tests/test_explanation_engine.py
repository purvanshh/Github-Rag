import unittest
from unittest.mock import MagicMock
from reasoning.explanation_engine import CodeExplanationEngine


class TestExplanationEngine(unittest.TestCase):
    def test_explain_beginner(self):
        analyzer = MagicMock()
        engine = CodeExplanationEngine(analyzer)
        
        # Mock generate_content response if using gemini, or openai choices
        if engine._use_gemini:
            mock_resp = MagicMock()
            mock_resp.text = "This code acts like a librarian."
            engine._gemini_model.generate_content = MagicMock(return_value=mock_resp)
        else:
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "This code acts like a librarian."
            mock_resp.choices = [mock_choice]
            engine._openai_client.chat.completions.create = MagicMock(return_value=mock_resp)
            
        res = engine.explain("def book(): pass", level="beginner")
        self.assertIn("librarian", res)


if __name__ == "__main__":
    unittest.main()
