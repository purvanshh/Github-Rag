"""server.py — FastAPI server for the GitHub RAG system.

Exposes REST endpoints for:
    * Repo ingestion
    * Question answering
    * Repository overview
    * File dependencies
    * Function usage
"""

from __future__ import annotations

import logging

# Load .env before any other project imports so OPENAI_API_KEY is set
from pathlib import Path
from dotenv import load_dotenv
_server_root = Path(__file__).resolve().parent.parent
load_dotenv(_server_root / ".env", override=True)
load_dotenv(override=False)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import config
from ingestion.repo_pipeline import RepoIngestionPipeline
from reasoning.repo_analyzer import RepoAnalyzer
from reasoning.query_router import QueryRouter


app = FastAPI(
    title="GitHub Codebase Intelligence",
    description="Repo-level RAG system with AST parsing, dependency graphs, and LLM reasoning.",
    version="0.1.0",
)


# ---------- Request / Response Models ----------


class IngestRequest(BaseModel):
    repo_url: str


class IngestResponse(BaseModel):
    status: str
    repo_name: str
    num_files: int
    num_symbols: int
    num_chunks: int
    indexing_time: float


class QueryRequest(BaseModel):
    repo: str
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str


class RepoOverviewResponse(BaseModel):
    architecture_summary: str | None
    architecture_metadata: dict
    most_connected_modules: list[tuple[str, int]]
    most_called_functions: list[tuple[str, int]]
    directory_tree: str


class DependenciesResponse(BaseModel):
    file: str
    dependencies: list[str]


class FunctionUsageResponse(BaseModel):
    function: str
    callers: list[str]
    callees: list[str]


# ---------- Helpers ----------


ANALYZER_CACHE: dict[str, RepoAnalyzer] = {}


def _get_repo_analyzer(repo_name: str) -> RepoAnalyzer:
    """Return a cached RepoAnalyzer for the given repository.

    This avoids rebuilding dependency and call graphs on every request.
    """
    if repo_name not in ANALYZER_CACHE:
        ANALYZER_CACHE[repo_name] = RepoAnalyzer(
            repo_name=repo_name,
            repos_root=config.repos_dir,
        )
    return ANALYZER_CACHE[repo_name]


# ---------- Endpoints ----------


@app.get("/")
def root() -> dict:
    """Root path: point users to the API docs and health check."""
    return {
        "app": "GitHub Codebase Intelligence",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Avoid 404 when the browser requests a favicon."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_repo(request: IngestRequest) -> IngestResponse:
    """Clone, parse, chunk, embed, and index a GitHub repo."""
    pipeline = RepoIngestionPipeline()
    try:
        result = pipeline.ingest_repository(request.repo_url)
    except Exception as exc:  # pragma: no cover - network / IO errors
        logging.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        status="ok",
        repo_name=result.repo_name,
        num_files=result.num_files,
        num_symbols=result.num_symbols,
        num_chunks=result.num_chunks,
        indexing_time=result.indexing_time,
    )


@app.post("/query", response_model=QueryResponse)
def query_codebase(request: QueryRequest) -> QueryResponse:
    """Ask a question about an indexed codebase via QueryRouter + RepoAnalyzer."""
    if not request.repo:
        raise HTTPException(status_code=400, detail="Missing 'repo' in request body.")

    analyzer = _get_repo_analyzer(request.repo)
    router = QueryRouter(analyzer)

    try:
        result = router.route_query(request.query)
    except Exception as exc:  # pragma: no cover - LLM / IO errors
        logging.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # For non-QA responses (e.g., dependencies, function usage), we wrap
    # them into a simple answer string for now.
    if "answer" in result and "sources" in result:
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            model=result.get("model", config.llm_model),
        )

    # Fallback: treat the result as JSON and return a textual summary.
    return QueryResponse(
        answer=str(result),
        sources=[],
        model=config.llm_model,
    )


@app.get("/repo/{repo}/overview", response_model=RepoOverviewResponse)
def repo_overview(repo: str) -> RepoOverviewResponse:
    """Return a combined overview for a repository."""
    try:
        analyzer = _get_repo_analyzer(repo)
        overview = analyzer.get_repo_overview()
        return RepoOverviewResponse(
            architecture_summary=overview.get("architecture_summary"),
            architecture_metadata=overview.get("architecture_metadata", {}),
            most_connected_modules=overview.get("most_connected_modules", []),
            most_called_functions=overview.get("most_called_functions", []),
            directory_tree=overview.get("directory_tree", ""),
        )
    except Exception as exc:
        logging.exception("Repo overview failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/repo/{repo}/dependencies/{file_path:path}", response_model=DependenciesResponse)
def repo_file_dependencies(repo: str, file_path: str) -> DependenciesResponse:
    """Return dependency graph information for a given file."""
    try:
        analyzer = _get_repo_analyzer(repo)
        deps = analyzer.get_file_dependencies(file_path)
        return DependenciesResponse(file=file_path, dependencies=deps)
    except Exception as exc:
        logging.exception("File dependencies failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/repo/{repo}/function/{name}", response_model=FunctionUsageResponse)
def repo_function_usage(repo: str, name: str) -> FunctionUsageResponse:
    """Return call graph information for a given function."""
    try:
        analyzer = _get_repo_analyzer(repo)
        usage = analyzer.find_function_usage(name)
        return FunctionUsageResponse(
            function=name,
            callers=usage.get("callers", []),
            callees=usage.get("callees", []),
        )
    except Exception as exc:
        logging.exception("Function usage failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)

