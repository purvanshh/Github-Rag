"""
chunk_code.py — Smart code chunking based on AST symbols.

Chunks code by semantic boundaries (functions, classes, methods)
instead of naive token-count splitting.
"""

from dataclasses import dataclass
from ingestion.parse_code import ParsedSymbol


@dataclass
class CodeChunk:
    """A semantic code chunk with rich metadata."""
    content: str
    file_path: str
    symbol_name: str
    symbol_type: str  # "function", "class", "method", "module"
    language: str
    start_line: int
    end_line: int
    repo_name: str
    docstring: str | None = None

    def to_embedding_text(self) -> str:
        """Create the text representation used for embedding.
        
        Combines file path, symbol info, docstring, and code for
        better retrieval accuracy.
        """
        parts = [
            f"File: {self.file_path}",
            f"{self.symbol_type}: {self.symbol_name}",
        ]
        if self.docstring:
            parts.append(f"Docstring: {self.docstring}")
        parts.append(f"Code:\n{self.content}")
        return "\n\n".join(parts)

    def to_metadata(self) -> dict:
        """Return metadata dict for vector store."""
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "repo_name": self.repo_name,
        }


def create_chunks_from_symbols(
    symbols: list[ParsedSymbol],
    repo_name: str,
) -> list[CodeChunk]:
    """Convert parsed symbols into semantic code chunks.
    
    Args:
        symbols: List of parsed symbols from Tree-sitter.
        repo_name: Name of the repository.
    
    Returns:
        List of CodeChunk objects ready for embedding.
    """
    chunks = []
    for symbol in symbols:
        chunk = CodeChunk(
            content=symbol.code,
            file_path=symbol.file_path,
            symbol_name=symbol.name,
            symbol_type=symbol.type,
            language=symbol.language,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            repo_name=repo_name,
            docstring=symbol.docstring,
        )
        chunks.append(chunk)
    return chunks
