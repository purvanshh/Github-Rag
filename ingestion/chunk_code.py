"""
chunk_code.py — Smart code chunking based on AST symbols.

Chunks code by semantic boundaries (functions, classes, methods)
instead of naive token-count splitting.

Chunking strategy:
  1. Each function / method becomes its own chunk.
  2. Each class gets a *signature* chunk (class header + docstring, without
     method bodies) so the LLM can see the class shape.  Individual methods
     are separate chunks with ``parent_class`` metadata.
  3. Consecutive import statements in a file are grouped into a single
     "imports" chunk.
  4. A ``max_chunk_lines`` guard splits any oversized symbol into
     overlapping sub-chunks so embedding models stay within context limits.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from ingestion.parse_code import ParsedSymbol

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHUNK_LINES = 300
OVERLAP_LINES = 30


@dataclass
class CodeChunk:
    """A semantic code chunk with rich metadata."""
    id: str = ""
    content: str = ""
    file_path: str = ""
    symbol_name: str = ""
    symbol_type: str = ""  # "function", "class", "method", "import"
    language: str = ""
    start_line: int = 0
    end_line: int = 0
    repo_name: str = ""
    docstring: str | None = None
    parent_class: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _make_chunk_id(
                self.repo_name, self.file_path,
                self.symbol_name, self.start_line,
            )

    def to_embedding_text(self) -> str:
        """Create the text representation used for embedding.

        Combines file path, symbol info, docstring, and code for
        better retrieval accuracy.
        """
        parts = [
            f"File: {self.file_path}",
            f"{self.symbol_type}: {self.symbol_name}",
        ]
        if self.parent_class:
            parts.append(f"Class: {self.parent_class}")
        if self.docstring:
            parts.append(f"Docstring: {self.docstring}")
        parts.append(f"Code:\n{self.content}")
        return "\n\n".join(parts)

    def to_metadata(self) -> dict:
        """Return metadata dict for vector store."""
        meta = {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "repo_name": self.repo_name,
        }
        if self.parent_class:
            meta["parent_class"] = self.parent_class
        return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(repo: str, path: str, symbol: str, line: int) -> str:
    """Deterministic short id for a chunk."""
    raw = f"{repo}:{path}:{symbol}:{line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_large_chunk(chunk: CodeChunk, max_lines: int) -> list[CodeChunk]:
    """Split an oversized chunk into overlapping sub-chunks."""
    lines = chunk.content.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return [chunk]

    sub_chunks: list[CodeChunk] = []
    start = 0
    part = 1
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        sub_content = "".join(lines[start:end])
        sub_chunks.append(CodeChunk(
            content=sub_content,
            file_path=chunk.file_path,
            symbol_name=f"{chunk.symbol_name}__part{part}",
            symbol_type=chunk.symbol_type,
            language=chunk.language,
            start_line=chunk.start_line + start,
            end_line=chunk.start_line + end - 1,
            repo_name=chunk.repo_name,
            docstring=chunk.docstring if part == 1 else None,
            parent_class=chunk.parent_class,
        ))
        start += max_lines - OVERLAP_LINES
        part += 1

    return sub_chunks


def _group_imports(symbols: list[ParsedSymbol]) -> dict[str, list[ParsedSymbol]]:
    """Group consecutive import symbols by file path."""
    groups: dict[str, list[ParsedSymbol]] = defaultdict(list)
    for sym in symbols:
        if sym.type == "import":
            groups[sym.file_path].append(sym)
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_chunks_from_symbols(
    symbols: list[ParsedSymbol],
    repo_name: str,
    max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES,
) -> list[CodeChunk]:
    """Convert parsed symbols into semantic code chunks.

    Strategy:
      * Functions and methods → one chunk each.
      * Classes → a *signature-only* chunk (header + docstring, no method
        bodies).  Methods are already separate symbols.
      * Imports per file → merged into a single chunk.
      * Oversized chunks are split with overlap.

    Args:
        symbols: List of parsed symbols from Tree-sitter.
        repo_name: Name of the repository.
        max_chunk_lines: Maximum lines per chunk before splitting.

    Returns:
        List of CodeChunk objects ready for embedding.
    """
    chunks: list[CodeChunk] = []
    seen_import_files: set[str] = set()
    import_groups = _group_imports(symbols)

    for symbol in symbols:
        # --- imports: merge per-file ---
        if symbol.type == "import":
            if symbol.file_path in seen_import_files:
                continue
            seen_import_files.add(symbol.file_path)
            group = import_groups[symbol.file_path]
            merged_code = "\n".join(s.code for s in group)
            merged_name = f"imports ({len(group)} statements)"
            chunks.append(CodeChunk(
                content=merged_code,
                file_path=symbol.file_path,
                symbol_name=merged_name,
                symbol_type="import",
                language=symbol.language,
                start_line=group[0].start_line,
                end_line=group[-1].end_line,
                repo_name=repo_name,
            ))
            continue

        # --- classes: emit a signature-only chunk ---
        if symbol.type == "class":
            chunks.append(CodeChunk(
                content=symbol.code,
                file_path=symbol.file_path,
                symbol_name=symbol.name,
                symbol_type="class",
                language=symbol.language,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                repo_name=repo_name,
                docstring=symbol.docstring,
            ))
            continue

        # --- functions / methods ---
        chunks.append(CodeChunk(
            content=symbol.code,
            file_path=symbol.file_path,
            symbol_name=symbol.name,
            symbol_type=symbol.type,
            language=symbol.language,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            repo_name=repo_name,
            docstring=symbol.docstring,
            parent_class=symbol.parent_class,
        ))

    # Post-process: split oversized chunks
    final_chunks: list[CodeChunk] = []
    for chunk in chunks:
        final_chunks.extend(_split_large_chunk(chunk, max_chunk_lines))

    logger.info(
        "Created %d chunks from %d symbols (repo=%s)",
        len(final_chunks), len(symbols), repo_name,
    )
    return final_chunks
