"""architecture_summarizer.py — Automatic high-level architecture summaries.

Generates a natural language overview of a repository using:
    * Directory structure (file tree)
    * Dependency graph (import relationships)
    * Major modules (most-imported files)

The summary is produced by an LLM using the ARCHITECTURE_PROMPT_TEMPLATE.
"""

from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI

from graphs.dependency_graph import DependencyGraph, build_dependency_graph
from reasoning.prompt_templates import ARCHITECTURE_PROMPT_TEMPLATE


def _iter_repo_entries(repo_path: str) -> Iterable[tuple[str, list[str], list[str]]]:
    """Yield (root, dirs, files) tuples for the repository, skipping noise."""
    skip_dirs = {
        ".git",
        "venv",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        yield root, sorted(dirs), sorted(files)


def build_directory_tree(
    repo_path: str,
    max_depth: int = 5,
    max_entries_per_dir: int = 50,
) -> str:
    """Return a pretty-printed directory tree for use in prompts.

    Args:
        repo_path: Path to the repository root.
        max_depth: Maximum depth of directories to traverse.
        max_entries_per_dir: Maximum number of entries per directory before
            collapsing the remainder with a ``...`` line.
    """
    repo_path = os.path.abspath(repo_path)
    lines: list[str] = []
    root_prefix_len = len(repo_path.rstrip(os.sep)) + 1

    for root, dirs, files in _iter_repo_entries(repo_path):
        rel_root = root[root_prefix_len:] if root.startswith(repo_path) else root
        depth = rel_root.count(os.sep) + (1 if rel_root else 0)
        if depth > max_depth:
            continue

        indent = "  " * depth
        if rel_root:
            lines.append(f"{indent}{rel_root}/")

        entries = dirs + files
        if not entries:
            continue

        for name in entries[:max_entries_per_dir]:
            lines.append(f"{indent}  {name}")
        if len(entries) > max_entries_per_dir:
            lines.append(f"{indent}  ... ({len(entries) - max_entries_per_dir} more)")

    return "\n".join(lines)


def _summarize_dependency_graph(graph: DependencyGraph, top_k: int = 10) -> str:
    """Create a textual summary of major modules in the dependency graph."""
    hubs = graph.get_most_connected(top_k=top_k)
    if not hubs:
        return "No internal Python dependencies detected."

    lines = ["Top dependency hubs (most imported modules/files):"]
    for name, degree in hubs:
        lines.append(f"- {name} (in-degree: {degree})")
    return "\n".join(lines)


def generate_architecture_summary(
    repo_path: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
) -> dict:
    """Generate a high-level architecture summary for a repository.

    Combines directory structure and dependency information into a prompt
    and uses an LLM to synthesize:

        * System Overview
        * Key Modules
        * Data Flow
        * Technologies Used

    Args:
        repo_path: Path to the repository root.
        model: OpenAI chat model name.
        temperature: Sampling temperature for the LLM.

    Returns:
        Dict containing the summary, model used, and auxiliary data used to
        build the prompt (file tree and dependency hubs).
    """
    repo_path = os.path.abspath(repo_path)

    # 1) Directory structure
    file_tree = build_directory_tree(repo_path)

    # 2) Dependency graph + major modules
    dep_graph = build_dependency_graph(repo_path)
    dep_summary = _summarize_dependency_graph(dep_graph)

    context = f"{dep_summary}\n"

    # 3) Build prompt
    user_prompt = ARCHITECTURE_PROMPT_TEMPLATE.format(
        file_tree=file_tree,
        context=context,
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior software architect. "
                    "Given repository metadata, produce a clear, concise "
                    "architecture summary."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    summary = response.choices[0].message.content

    return {
        "summary": summary,
        "model": model,
        "file_tree": file_tree,
        "dependency_hubs": dep_graph.get_most_connected(top_k=10),
    }

