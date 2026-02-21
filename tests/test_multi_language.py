import unittest
import os
import tempfile
import shutil
from ingestion.parse_code import parse_file


class TestMultiLanguage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_java_parsing(self):
        java_content = """
package com.example;
import java.util.List;
import java.io.*;

public class UserService {
    private List<String> users;

    public void registerUser(String username) {
        System.out.println("Registered User: " + username);
    }
}
"""
        file_path = os.path.join(self.test_dir, "UserService.java")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(java_content)

        symbols = parse_file(file_path, repo_path=self.test_dir, repo_id="test_repo")
        
        # We expect imports (List, io), class (UserService), method (registerUser)
        types = [s.type for s in symbols]
        self.assertEqual(types.count("import"), 2)
        self.assertEqual(types.count("class"), 1)
        self.assertEqual(types.count("method"), 1)

        user_service = next(s for s in symbols if s.type == "class")
        self.assertEqual(user_service.name, "UserService")

        register_method = next(s for s in symbols if s.type == "method")
        self.assertEqual(register_method.name, "registerUser")
        self.assertEqual(register_method.parent_class, "UserService")

    def test_rust_parsing(self):
        rust_content = """
use std::collections::HashMap;

pub struct Cache {
    data: HashMap<String, String>,
}

impl Cache {
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}
"""
        file_path = os.path.join(self.test_dir, "lib.rs")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(rust_content)

        symbols = parse_file(file_path, repo_path=self.test_dir, repo_id="test_repo")
        
        # We expect use statement (import), struct (class), fn (method or function)
        types = [s.type for s in symbols]
        self.assertGreaterEqual(types.count("import"), 1)
        self.assertGreaterEqual(types.count("class"), 1)
        self.assertGreaterEqual(types.count("method") + types.count("function"), 1)

        cache_struct = next(s for s in symbols if s.type == "class")
        self.assertEqual(cache_struct.name, "Cache")


if __name__ == "__main__":
    unittest.main()
