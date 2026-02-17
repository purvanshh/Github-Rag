"""
symbol_resolver.py — Symbol Resolution Engine.
Resolves variable, class, function, and method calls to their Fully Qualified Names (FQNs)
by parsing scope, file-level imports, and local module definitions.
"""

from __future__ import annotations

import re
import os
from typing import Dict, List, Set, Tuple
from ingestion.parse_code import ParsedSymbol
from metadata_utils import normalize_file_path, module_name_from_path


class SymbolResolver:
    """Resolves code symbols contextually to their Fully Qualified Name (FQN) in the repository."""

    def __init__(self, symbols: list[ParsedSymbol], repo_path: str | None = None) -> None:
        self.repo_path = repo_path
        self.symbols = symbols
        
        # Map of module_name -> { symbol_name: fqn }
        self.module_definitions: Dict[str, Dict[str, str]] = {}
        # Map of file_path -> { imported_name/alias: origin_fqn }
        self.file_imports: Dict[str, Dict[str, str]] = {}
        # Map of class_fqn -> set of method names
        self.class_methods: Dict[str, Set[str]] = {}
        
        self._build_index()

    def _build_index(self) -> None:
        """Scan definitions and import statements to build reference lookup tables."""
        # 1. Index all defined symbols (functions, classes, methods)
        for sym in self.symbols:
            if sym.type == "import":
                continue

            mod_name = module_name_from_path(self.repo_path or "", sym.file_path)
            if mod_name not in self.module_definitions:
                self.module_definitions[mod_name] = {}

            # Map the local short name to its FQN
            self.module_definitions[mod_name][sym.name] = sym.fqn

            # Register class methods
            if sym.type == "method" and sym.parent_class:
                class_fqn = f"{mod_name}.{sym.parent_class}"
                if class_fqn not in self.class_methods:
                    self.class_methods[class_fqn] = set()
                self.class_methods[class_fqn].add(sym.name)

        # 2. Parse import statements to map imported aliases to source FQNs
        for sym in self.symbols:
            if sym.type != "import":
                continue

            file_path = normalize_file_path(sym.file_path, self.repo_path)
            if file_path not in self.file_imports:
                self.file_imports[file_path] = {}

            self._parse_import_code(sym.code, file_path)

    def _parse_import_code(self, code: str, file_path: str) -> None:
        """Parse python import statement string and register in import lookup table."""
        # Clean backslashes and lines
        clean_code = " ".join(line.strip() for line in code.split("\n") if line.strip())
        
        # Regex for 'from module import x, y as z'
        from_match = re.match(r"^from\s+([\w\.]+)\s+import\s+(.+)$", clean_code)
        if from_match:
            module_src = from_match.group(1)
            imports_str = from_match.group(2).strip().strip("()")
            
            items = [item.strip() for item in imports_str.split(",") if item.strip()]
            for item in items:
                if " as " in item:
                    orig, alias = item.split(" as ")
                    orig, alias = orig.strip(), alias.strip()
                    self.file_imports[file_path][alias] = f"{module_src}.{orig}"
                else:
                    self.file_imports[file_path][item] = f"{module_src}.{item}"
            return

        # Regex for 'import module1, module2 as alias'
        import_match = re.match(r"^import\s+(.+)$", clean_code)
        if import_match:
            imports_str = import_match.group(1).strip()
            items = [item.strip() for item in imports_str.split(",") if item.strip()]
            for item in items:
                if " as " in item:
                    orig, alias = item.split(" as ")
                    orig, alias = orig.strip(), alias.strip()
                    self.file_imports[file_path][alias] = orig
                else:
                    self.file_imports[file_path][item] = item
            return

    def resolve_symbol(self, symbol_name: str, file_path: str) -> str | None:
        """Resolve a short symbol name contextually to its FQN.

        Args:
            symbol_name: Dotted or bare symbol name (e.g. 'helper' or 'helper.run').
            file_path: The file where the reference occurs.
        """
        if not symbol_name:
            return None

        file_path = normalize_file_path(file_path, self.repo_path)
        module_name = module_name_from_path(self.repo_path or "", file_path)

        parts = symbol_name.split(".")
        base_name = parts[0]
        resolved_base = None

        # 1. Resolve from file-level imports
        if file_path in self.file_imports and base_name in self.file_imports[file_path]:
            resolved_base = self.file_imports[file_path][base_name]

        # 2. Resolve from local module definitions
        if not resolved_base and module_name in self.module_definitions:
            if base_name in self.module_definitions[module_name]:
                resolved_base = self.module_definitions[module_name][base_name]

        # Reconstruct FQN if base was resolved
        if resolved_base:
            if len(parts) > 1:
                return f"{resolved_base}.{'.'.join(parts[1:])}"
            return resolved_base

        # 3. Fallback: Global lookup in repo
        for mod, defs in self.module_definitions.items():
            if base_name in defs:
                resolved_base = defs[base_name]
                if len(parts) > 1:
                    return f"{resolved_base}.{'.'.join(parts[1:])}"
                return resolved_base

        # Fallback to local scope dotted path
        if module_name:
            return f"{module_name}.{symbol_name}"
        return symbol_name
