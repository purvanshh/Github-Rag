import unittest
from ingestion.parse_code import ParsedSymbol
from ingestion.chunk_code import create_chunks_from_symbols


class TestAdvancedChunking(unittest.TestCase):
    def test_contextual_and_hierarchical_chunking(self):
        symbols = [
            # Import statement
            ParsedSymbol(
                name="import os",
                type="import",
                code="import os",
                start_line=1,
                end_line=1,
                file_path="src/main.py",
                language="python",
                fqn="src.main.imports",
                symbol_id="test:src/main.py:src.main.imports:1",
            ),
            # Class definition
            ParsedSymbol(
                name="Server",
                type="class",
                code="class Server:\n    pass",
                start_line=3,
                end_line=6,
                file_path="src/main.py",
                language="python",
                fqn="src.main.Server",
                symbol_id="test:src/main.py:src.main.Server:3",
            ),
            # Method definition inside Server class
            ParsedSymbol(
                name="start",
                type="method",
                code="def start(self):\n    pass",
                start_line=5,
                end_line=6,
                file_path="src/main.py",
                language="python",
                fqn="src.main.Server.start",
                symbol_id="test:src/main.py:src.main.Server.start:5",
                parent_class="Server",
            ),
        ]

        chunks = create_chunks_from_symbols(symbols, repo_name="my_repo")

        # We expect: 1 imports chunk, 1 class chunk, 1 method chunk
        self.assertEqual(len(chunks), 3)

        # Find chunks by type
        import_chunk = next(c for c in chunks if c.symbol_type == "import")
        class_chunk = next(c for c in chunks if c.symbol_type == "class")
        method_chunk = next(c for c in chunks if c.symbol_type == "method")

        # 1. Verification of hierarchical chunking (method links to class chunk ID)
        self.assertIsNotNone(method_chunk.parent_chunk_id)
        self.assertEqual(method_chunk.parent_chunk_id, class_chunk.id)

        # 2. Verification of dependency imports prepending in code chunk content
        self.assertIn("// Dependency imports:", method_chunk.content)
        self.assertIn("import os", method_chunk.content)

        # 3. Verification of contextual class/function header
        self.assertIn("// Context: Method 'start' of class 'Server'", method_chunk.content)


if __name__ == "__main__":
    unittest.main()
