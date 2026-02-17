import unittest
from ingestion.parse_code import ParsedSymbol
from ingestion.symbol_resolver import SymbolResolver


class TestSymbolResolver(unittest.TestCase):
    def test_symbol_resolution(self):
        symbols = [
            # Defined symbol in main.py
            ParsedSymbol(
                name="run_engine",
                type="function",
                code="def run_engine(): pass",
                start_line=10,
                end_line=12,
                file_path="main.py",
                language="python",
                fqn="main.run_engine",
                symbol_id="test:main.py:main.run_engine:10",
            ),
            # Import in main.py
            ParsedSymbol(
                name="import utils.helper as h",
                type="import",
                code="import utils.helper as h",
                start_line=1,
                end_line=1,
                file_path="main.py",
                language="python",
                fqn="main.imports",
                symbol_id="test:main.py:main.imports:1",
            ),
            # From import in main.py
            ParsedSymbol(
                name="from core.config import settings",
                type="import",
                code="from core.config import settings",
                start_line=2,
                end_line=2,
                file_path="main.py",
                language="python",
                fqn="main.imports",
                symbol_id="test:main.py:main.imports:2",
            ),
            # Defined symbol in utils/helper.py
            ParsedSymbol(
                name="format_log",
                type="function",
                code="def format_log(): pass",
                start_line=5,
                end_line=6,
                file_path="utils/helper.py",
                language="python",
                fqn="utils.helper.format_log",
                symbol_id="test:utils/helper.py:utils.helper.format_log:5",
            ),
        ]

        resolver = SymbolResolver(symbols)

        # 1. Resolve local defined function
        self.assertEqual(resolver.resolve_symbol("run_engine", "main.py"), "main.run_engine")

        # 2. Resolve imported alias
        self.assertEqual(resolver.resolve_symbol("h.format_log", "main.py"), "utils.helper.format_log")

        # 3. Resolve from import name
        self.assertEqual(resolver.resolve_symbol("settings", "main.py"), "core.config.settings")

        # 4. Resolve global fallback
        self.assertEqual(resolver.resolve_symbol("format_log", "main.py"), "utils.helper.format_log")


if __name__ == "__main__":
    unittest.main()
