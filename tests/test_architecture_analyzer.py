import unittest
from unittest.mock import MagicMock
import networkx as nx
from reasoning.architecture_analyzer import ArchitectureAnalyzer


class TestArchitectureAnalyzer(unittest.TestCase):
    def test_generate_dependency_chart(self):
        analyzer = MagicMock()
        kg = MagicMock()
        
        g = nx.DiGraph()
        g.add_edge("app.py", "service.py", type="imports")
        kg.graph = g
        analyzer._kg = kg
        
        arch = ArchitectureAnalyzer(analyzer)
        chart = arch.generate_dependency_chart()
        
        self.assertIn("flowchart TD", chart)
        self.assertIn("app.py --> service.py", chart)

    def test_generate_class_hierarchy(self):
        analyzer = MagicMock()
        kg = MagicMock()
        
        g = nx.DiGraph()
        g.add_edge("AppService", "BaseService", type="inherits")
        kg.graph = g
        analyzer._kg = kg
        
        arch = ArchitectureAnalyzer(analyzer)
        chart = arch.generate_class_hierarchy()
        
        self.assertIn("classDiagram", chart)
        self.assertIn("BaseService <|-- AppService", chart)


if __name__ == "__main__":
    unittest.main()
