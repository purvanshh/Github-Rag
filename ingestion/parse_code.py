"""
parse_code.py — Parse source code files using Tree-sitter.

Extracts AST-level symbols: functions, classes, imports, methods.
"""

from dataclasses import dataclass, field
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser


@dataclass
class ParsedSymbol:
    """Represents an extracted code symbol."""
    name: str
    type: str  # "function", "class", "method", "import"
    code: str
    start_line: int
    end_line: int
    file_path: str
    language: str
    docstring: str | None = None
    parent_class: str | None = None


# Supported languages and their file extensions
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
}


def get_language_for_file(file_path: str) -> str | None:
    """Determine the Tree-sitter language for a file based on its extension."""
    import os
    ext = os.path.splitext(file_path)[1]
    return LANGUAGE_MAP.get(ext)


# Map language name to their tree-sitter Language object
TREE_SITTER_LANGUAGES = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
}


def _get_parser(language: str) -> Parser | None:
    """Get a Tree-sitter parser for the given language."""
    lang_obj = TREE_SITTER_LANGUAGES.get(language)
    if lang_obj is None:
        return None
    parser = Parser(lang_obj)
    return parser


def parse_file(file_path: str, language: str | None = None) -> list[ParsedSymbol]:
    """Parse a source file and extract symbols using Tree-sitter.
    
    Args:
        file_path: Path to the source file.
        language: Programming language. Auto-detected if not provided.
    
    Returns:
        List of ParsedSymbol objects extracted from the file.
    """
    if language is None:
        language = get_language_for_file(file_path)
        if language is None:
            return []

    parser = _get_parser(language)
    if parser is None:
        return []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()

    tree = parser.parse(source_code.encode("utf-8"))

    symbols = []
    # TODO: Walk the AST tree and extract functions, classes, imports
    # This will be implemented per-language with Tree-sitter queries
    
    return symbols


def parse_directory(repo_path: str) -> list[ParsedSymbol]:
    """Recursively parse all supported source files in a directory.
    
    Args:
        repo_path: Root path of the repository.
    
    Returns:
        List of all ParsedSymbol objects found in the repo.
    """
    import os
    
    all_symbols = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file_name in files:
            file_path = os.path.join(root, file_name)
            language = get_language_for_file(file_path)
            if language:
                symbols = parse_file(file_path, language)
                all_symbols.extend(symbols)
    
    return all_symbols
