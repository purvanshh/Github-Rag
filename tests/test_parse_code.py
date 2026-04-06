import unittest
import tempfile
import os
import shutil
from ingestion.parse_code import parse_directory, parse_file


class TestParseCode(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Python file
        self.py_file = os.path.join(self.test_dir, "app.py")
        with open(self.py_file, "w", encoding="utf-8") as f:
            f.write("def my_func():\n    pass\n")

        # JS file
        self.js_file = os.path.join(self.test_dir, "script.js")
        with open(self.js_file, "w", encoding="utf-8") as f:
            f.write("function jsFunc() {\n    return 42;\n}\n")

        # TS file
        self.ts_file = os.path.join(self.test_dir, "types.ts")
        with open(self.ts_file, "w", encoding="utf-8") as f:
            f.write("class TSClass {\n    method() {}\n}\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_multi_language_parsing(self):
        symbols = parse_directory(self.test_dir, repo_id="multi-repo")
        
        types = [s.type for s in symbols]
        names = [s.name for s in symbols]
        
        self.assertIn("my_func", names)
        self.assertIn("jsFunc", names)
        self.assertIn("TSClass", names)
        self.assertIn("method", names)

    def test_parse_file_unsupported(self):
        txt_file = os.path.join(self.test_dir, "notes.txt")
        with open(txt_file, "w") as f:
            f.write("unsupported text")
        symbols = parse_file(txt_file, self.test_dir)
        self.assertEqual(len(symbols), 0)


if __name__ == "__main__":
    unittest.main()
