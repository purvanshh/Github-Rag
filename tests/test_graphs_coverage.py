import unittest
from unittest.mock import patch
import os
import tempfile
import shutil
from graphs.call_graph import CallGraph, build_call_graph, where_is_function_used, which_functions_does_it_call
from graphs.dependency_graph import DependencyGraph, build_dependency_graph, get_dependencies


class TestGraphsCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file1 = os.path.join(self.test_dir, "a.py")
        with open(self.file1, "w", encoding="utf-8") as f:
            f.write("import b\nclass MyClass:\n    def method(self):\n        b.bar()\ndef foo():\n    x = MyClass()\n    x.method()\n")
        self.file2 = os.path.join(self.test_dir, "b.py")
        with open(self.file2, "w", encoding="utf-8") as f:
            f.write("def bar():\n    pass\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_call_graph(self):
        cg = CallGraph()
        cg.add_call("caller_func", "callee_func", "file.py")
        
        self.assertEqual(cg.get_callees("caller_func"), ["callee_func"])
        self.assertEqual(cg.get_callers("callee_func"), ["caller_func"])
        self.assertEqual(cg.get_callees("nonexistent"), [])
        self.assertEqual(cg.get_callers("nonexistent"), [])
        
        self.assertEqual(cg.get_most_called(1), [("callee_func", 1)])
        self.assertIn("caller_func", cg.to_dict())
        
        chain = cg.get_call_chain("caller_func", depth=2)
        self.assertEqual(list(chain.keys()), ["callee_func"])
        
        self.assertEqual(cg.get_call_chain("caller_func", depth=0), {})

    def test_dependency_graph(self):
        dg = DependencyGraph()
        dg.add_edge("a.py", "b.py", "import b")
        
        self.assertEqual(dg.get_dependencies("a.py"), ["b.py"])
        self.assertEqual(dg.get_dependents("b.py"), ["a.py"])
        self.assertEqual(dg.get_dependencies("nonexistent"), [])
        self.assertEqual(dg.get_dependents("nonexistent"), [])
        
        # Test visual metrics
        dg.add_edge("c.py", "b.py", "import b")
        hubs = dg.get_most_connected(1)
        self.assertEqual(hubs[0][0], "b.py")
        self.assertIn("a.py", dg.to_dict())

    def test_build_graphs_on_disk(self):
        dg = build_dependency_graph(self.test_dir)
        self.assertIn("a.py", dg.to_dict())
        self.assertEqual(get_dependencies(dg, "a.py"), ["b.py"])

        cg = build_call_graph(self.test_dir)
        self.assertIn("a.foo", cg.to_dict())
        
        # Test visualize_graph raises RuntimeError due to missing matplotlib
        from graphs.dependency_graph import visualize_graph
        with self.assertRaises(RuntimeError):
            visualize_graph(dg, "dummy.png")

    def test_resolve_import_to_file(self):
        from graphs.dependency_graph import _resolve_import_to_file
        
        module_index = {"pkg.sub": "pkg/sub.py"}
        
        # Test relative import resolution with dot level
        res = _resolve_import_to_file("sub", "pkg.main", level=1, module_index=module_index)
        self.assertEqual(res, "pkg/sub.py")
        
        # Test relative import dot resolution exceeding depth
        res2 = _resolve_import_to_file("sub", "pkg.main", level=10, module_index=module_index)
        self.assertIsNone(res2)


if __name__ == "__main__":
    unittest.main()
