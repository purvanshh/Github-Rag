import unittest
import os
import shutil
import tempfile
from ingestion.parse_code import ParsedSymbol
from graphs.knowledge_graph import RepositoryKnowledgeGraph


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure to mock python source files for ast parsing
        self.test_dir = tempfile.mkdtemp()
        
        # Write dummy python file containing classes, methods and calls
        self.file1 = os.path.join(self.test_dir, "app.py")
        with open(self.file1, "w", encoding="utf-8") as f:
            f.write("""
import core as helper

class Manager:
    def execute(self):
        helper.run()
""")
        
        self.file2 = os.path.join(self.test_dir, "core.py")
        with open(self.file2, "w", encoding="utf-8") as f:
            f.write("""
def run():
    print("running")
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_knowledge_graph_build(self):
        symbols = [
            # App import
            ParsedSymbol(
                name="import core as helper",
                type="import",
                code="import core as helper",
                start_line=2,
                end_line=2,
                file_path=self.file1,
                language="python",
                fqn="app.imports",
                symbol_id="test:app.py:app.imports:2",
            ),
            # Manager class in app.py
            ParsedSymbol(
                name="Manager",
                type="class",
                code="class Manager:\n    def execute(self):\n        helper.run()",
                start_line=4,
                end_line=6,
                file_path=self.file1,
                language="python",
                fqn="app.Manager",
                symbol_id="test:app.py:app.Manager:4",
            ),
            # execute method inside Manager class
            ParsedSymbol(
                name="execute",
                type="method",
                code="def execute(self):\n        helper.run()",
                start_line=5,
                end_line=6,
                file_path=self.file1,
                language="python",
                fqn="app.Manager.execute",
                symbol_id="test:app.py:app.Manager.execute:5",
                parent_class="Manager",
            ),
            # run function in core.py
            ParsedSymbol(
                name="run",
                type="function",
                code="def run():\n    print('running')",
                start_line=2,
                end_line=3,
                file_path=self.file2,
                language="python",
                fqn="core.run",
                symbol_id="test:core.py:core.run:2",
            ),
        ]

        kg = RepositoryKnowledgeGraph()
        kg.build(self.test_dir, symbols)

        # 1. Verification of class contains method
        self.assertIn("app.Manager.execute", kg.graph.successors("app.Manager"))

        # 2. Verification of module contains class/functions
        self.assertIn("app.Manager", kg.graph.successors("app"))
        self.assertIn("core.run", kg.graph.successors("core"))

        # 3. Verification of calls relationship resolved contextually via SymbolResolver
        # execute() calls helper.run() which resolves to core.run
        callees = kg.get_callees("app.Manager.execute")
        self.assertIn("core.run", callees)

        # Verification of callers of core.run
        callers = kg.get_callers("core.run")
        self.assertIn("app.Manager.execute", callers)


if __name__ == "__main__":
    unittest.main()
