import unittest
from unittest.mock import MagicMock
from reasoning.query_planner import AgenticQueryPlanner


class TestQueryPlanner(unittest.TestCase):
    def test_fallback_plan_execution(self):
        # Setup mock analyzer
        analyzer = MagicMock()
        analyzer.find_references.return_value = ["app.py:L10"]
        analyzer.ask_question.return_value = {"answer": "synthesized explanation"}

        planner = AgenticQueryPlanner(analyzer)
        
        # We simulate the execute_plan with a specific plan
        plan = [
            {"tool": "find_references", "symbol_name": "normalize_file_path"},
            {"tool": "ask_question", "query": "what is it doing?"}
        ]
        
        result = planner.execute_plan(plan)
        
        self.assertEqual(result, {"answer": "synthesized explanation"})
        analyzer.find_references.assert_called_once_with("normalize_file_path")
        analyzer.ask_question.assert_called()


if __name__ == "__main__":
    unittest.main()
