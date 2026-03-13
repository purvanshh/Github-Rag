"""
parse_code.py — Parse source code files using Tree-sitter.

Extracts AST-level symbols: functions, classes, methods, and imports.
Supports Python, JavaScript, and TypeScript.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

TREE_SITTER_LANGUAGES: dict[str, Language] = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
    "typescript": Language(tstypescript.language_typescript()),
}


def get_language_for_file(file_path: str) -> str | None:
    """Determine the Tree-sitter language for a file based on its extension."""
    ext = os.path.splitext(file_path)[1]
    return LANGUAGE_MAP.get(ext)


def _get_parser(language: str) -> Parser | None:
    """Return a Tree-sitter parser for *language*, or ``None`` if unsupported."""
    lang_obj = TREE_SITTER_LANGUAGES.get(language)
    if lang_obj is None:
        return None
    parser = Parser(lang_obj)
    return parser


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _node_text(node: Node, source: bytes) -> str:
    """Extract the source text covered by a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_docstring_python(node: Node, source: bytes) -> str | None:
    """Return the docstring of a Python function/class node, if present.

    Python docstrings are the first expression_statement child whose
    sole child is a ``string`` node.
    """
    body = node.child_by_field_name("body")
    if body is None:
        return None
    for child in body.children:
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr and expr.type == "string":
                raw = _node_text(expr, source)
                return raw.strip("\"'").strip()
        elif child.type in ("comment", "pass_statement"):
            continue
        else:
            break
    return None


def _extract_jsdoc(node: Node, source: bytes) -> str | None:
    """Return the JSDoc / leading block-comment for a JS/TS node, if present.

    Looks at the previous sibling; if it is a ``comment`` node whose text
    starts with ``/**``, treat it as the docstring.
    """
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "comment":
        text = _node_text(prev, source)
        if text.startswith("/**"):
            cleaned = text.strip("/").strip("*").strip()
            return cleaned if cleaned else None
    return None


def _function_name(node: Node, source: bytes) -> str:
    """Extract the function/method name from a node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node, source)
    return "<anonymous>"


def _class_name(node: Node, source: bytes) -> str:
    """Extract the class name from a node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node, source)
    return "<anonymous>"


# ---------------------------------------------------------------------------
# Per-language extractors
# ---------------------------------------------------------------------------

def _extract_python(root: Node, source: bytes, file_path: str) -> list[ParsedSymbol]:
    """Walk a Python AST and extract functions, classes, methods, and imports."""
    symbols: list[ParsedSymbol] = []

    def _walk(node: Node, parent_class_name: str | None = None) -> None:
        if node.type == "function_definition":
            name = _function_name(node, source)
            sym_type = "method" if parent_class_name else "function"
            symbols.append(ParsedSymbol(
                name=name,
                type=sym_type,
                code=_node_text(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file_path=file_path,
                language="python",
                docstring=_extract_docstring_python(node, source),
                parent_class=parent_class_name,
            ))
            return  # don't recurse into nested defs — they'll be their own symbols

        if node.type == "class_definition":
            name = _class_name(node, source)
            symbols.append(ParsedSymbol(
                name=name,
                type="class",
                code=_node_text(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file_path=file_path,
                language="python",
                docstring=_extract_docstring_python(node, source),
            ))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk(child, parent_class_name=name)
            return

        if node.type in ("import_statement", "import_from_statement"):
            symbols.append(ParsedSymbol(
                name=_node_text(node, source).strip(),
                type="import",
                code=_node_text(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file_path=file_path,
                language="python",
            ))
            return

        if node.type == "decorated_definition":
            for child in node.children:
                _walk(child, parent_class_name)
            return

        for child in node.children:
            _walk(child, parent_class_name)

    _walk(root)
    return symbols


# JS/TS share the same AST structure for the constructs we care about.
_JS_FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
}
_JS_CLASS_TYPES = {"class_declaration"}
_JS_METHOD_TYPES = {"method_definition"}
_JS_IMPORT_TYPES = {"import_statement"}


def _extract_js_ts(
    root: Node,
    source: bytes,
    file_path: str,
    language: str,
) -> list[ParsedSymbol]:
    """Walk a JavaScript / TypeScript AST and extract symbols."""
    symbols: list[ParsedSymbol] = []

    def _walk(node: Node, parent_class_name: str | None = None) -> None:
        # --- functions ---
        if node.type in _JS_FUNCTION_TYPES:
            name = _function_name(node, source)
            symbols.append(ParsedSymbol(
                name=name,
                type="function",
                code=_node_text(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file_path=file_path,
                language=language,
                docstring=_extract_jsdoc(node, source),
            ))
            return

        # --- arrow / anonymous functions assigned to variables ---
        if node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.children:
                if declarator.type == "variable_declarator":
                    value = declarator.child_by_field_name("value")
                    if value and value.type in ("arrow_function", "function_expression"):
                        name_node = declarator.child_by_field_name("name")
                        name = _node_text(name_node, source) if name_node else "<anonymous>"
                        symbols.append(ParsedSymbol(
                            name=name,
                            type="function",
                            code=_node_text(node, source),
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            file_path=file_path,
                            language=language,
                            docstring=_extract_jsdoc(node, source),
                        ))
                        return
            # Not a function assignment — fall through to recurse
            for child in node.children:
                _walk(child, parent_class_name)
            return

        # --- classes ---
        if node.type in _JS_CLASS_TYPES:
            name = _class_name(node, source)
            symbols.append(ParsedSymbol(
                name=name,
                type="class",
                code=_node_text(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file_path=file_path,
                language=language,
                docstring=_extract_jsdoc(node, source),
            ))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk(child, parent_class_name=name)
            return

        # --- methods (inside class body) ---
        if node.type in _JS_METHOD_TYPES:
            name = _function_name(node, source)
            symbols.append(ParsedSymbol(
                name=name,
                type="method",
                code=_node_text(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file_path=file_path,
                language=language,
                docstring=_extract_jsdoc(node, source),
                parent_class=parent_class_name,
            ))
            return

        # --- imports ---
        if node.type in _JS_IMPORT_TYPES:
            symbols.append(ParsedSymbol(
                name=_node_text(node, source).strip(),
                type="import",
                code=_node_text(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file_path=file_path,
                language=language,
            ))
            return

        # --- export wrappers (export default function …) ---
        if node.type in ("export_statement", "export_default_declaration"):
            for child in node.children:
                _walk(child, parent_class_name)
            return

        for child in node.children:
            _walk(child, parent_class_name)

    _walk(root)
    return symbols


# Dispatcher: language name → extractor function
_EXTRACTORS: dict[str, callable] = {
    "python": lambda root, src, fp: _extract_python(root, src, fp),
    "javascript": lambda root, src, fp: _extract_js_ts(root, src, fp, "javascript"),
    "typescript": lambda root, src, fp: _extract_js_ts(root, src, fp, "typescript"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(file_path: str, language: str | None = None) -> list[ParsedSymbol]:
    """Parse a source file and extract symbols using Tree-sitter.

    Args:
        file_path: Path to the source file.
        language: Programming language. Auto-detected from extension if not provided.

    Returns:
        List of ParsedSymbol objects extracted from the file.
    """
    if language is None:
        language = get_language_for_file(file_path)
        if language is None:
            return []

    parser = _get_parser(language)
    if parser is None:
        logger.debug("No parser available for language '%s'", language)
        return []

    extractor = _EXTRACTORS.get(language)
    if extractor is None:
        logger.debug("No extractor implemented for language '%s'", language)
        return []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)
    return extractor(tree.root_node, source_bytes, file_path)


def parse_directory(repo_path: str) -> list[ParsedSymbol]:
    """Recursively parse all supported source files in a directory.

    Args:
        repo_path: Root path of the repository.

    Returns:
        List of all ParsedSymbol objects found in the repo.
    """
    all_symbols: list[ParsedSymbol] = []
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
        "egg-info",
    }

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file_name in files:
            file_path = os.path.join(root, file_name)
            language = get_language_for_file(file_path)
            if language:
                symbols = parse_file(file_path, language)
                all_symbols.extend(symbols)

    logger.info(
        "Parsed %d symbols across %s",
        len(all_symbols),
        repo_path,
    )
    return all_symbols
