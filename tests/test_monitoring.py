import unittest
import logging
from io import StringIO
from observability.monitoring import StructuredJSONFormatter


class TestMonitoring(unittest.TestCase):
    def test_structured_json_formatter(self):
        formatter = StructuredJSONFormatter()
        
        logger_name = "test_logger"
        record = logging.LogRecord(
            name=logger_name,
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg="User %s did action %s",
            args=("Alice", "login"),
            exc_info=None
        )
        record.trace_id = "trace-12345"
        
        formatted_str = formatter.format(record)
        
        # Verify it parses as a valid JSON document
        import json
        log_json = json.loads(formatted_str)
        self.assertEqual(log_json["level"], "INFO")
        self.assertEqual(log_json["message"], "User Alice did action login")
        self.assertEqual(log_json["logger"], logger_name)
        self.assertEqual(log_json["trace_id"], "trace-12345")


if __name__ == "__main__":
    unittest.main()
