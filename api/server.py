"""
server.py — FastAPI server for the GitHub RAG system.

Exposes REST endpoints for repo ingestion and question answering.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    repo_name: str | None = None
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str


# ---------- Endpoints ----------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_repo(request: IngestRequest):
    """Clone, parse, chunk, embed, and index a GitHub repo."""
    # TODO: Wire up the full ingestion pipeline
    raise HTTPException(status_code=501, detail="Ingestion pipeline not yet wired up.")


@app.post("/query", response_model=QueryResponse)
def query_codebase(request: QueryRequest):
    """Ask a question about an indexed codebase."""
    # TODO: Wire up retriever + answer generator
    raise HTTPException(status_code=501, detail="Query pipeline not yet wired up.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
