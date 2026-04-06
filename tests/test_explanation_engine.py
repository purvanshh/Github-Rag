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

    def test_explain_openai(self):
        from config import config
        import os
        old_provider = config.llm_provider
        config.llm_provider = "openai"
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-mock-key"
        try:
            analyzer = MagicMock()
            engine = CodeExplanationEngine(analyzer)
            
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "OpenAI explain response"
            mock_resp.choices = [mock_choice]
            engine._openai_client.chat.completions.create = MagicMock(return_value=mock_resp)
            
            res = engine.explain("def book(): pass", level="beginner")
            self.assertEqual(res, "OpenAI explain response")

            # Test OpenAI exception catch
            engine._openai_client.chat.completions.create.side_effect = Exception("OpenAI fail")
            res_fail = engine.explain("def book(): pass")
            self.assertEqual(res_fail, "Failed to generate explanation.")
        finally:
            config.llm_provider = old_provider
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
