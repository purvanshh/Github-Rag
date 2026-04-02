import unittest
from unittest.mock import MagicMock
from reasoning.answer_generator import AnswerGenerator


class TestStreaming(unittest.TestCase):
    def test_answer_stream_yield(self):
        retriever = MagicMock()
        retriever.retrieve_with_context.return_value = "Mock codebase context"
        retriever.retrieve.return_value = [
            {"metadata": {"file_path": "api/server.py", "symbol_name": "app", "symbol_type": "variable", "start_line": 1, "end_line": 10}}
        ]

        generator = AnswerGenerator(retriever, repo_name="test_repo")
        
        # Mock LLM API stream yield
        if generator._use_gemini:
            mock_chunk = MagicMock()
            mock_chunk.text = "Hello stream"
            generator._gemini_model.generate_content = MagicMock(return_value=[mock_chunk])
        else:
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "Hello stream"
            generator._openai_client.chat.completions.create = MagicMock(return_value=[mock_chunk])

        chunks = list(generator.generate_answer_stream("what is this?"))
        
        self.assertEqual(chunks[0]["type"], "metadata")
        self.assertEqual(chunks[1]["type"], "token")
        self.assertEqual(chunks[1]["text"], "Hello stream")


if __name__ == "__main__":
    unittest.main()
