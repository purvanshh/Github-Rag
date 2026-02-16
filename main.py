"""
main.py — Entry point for the GitHub Codebase Intelligence System.

Provides a CLI for:
  - Ingesting a GitHub repo
  - Querying the indexed codebase
  - Starting the API server
"""

# Load .env first so OPENAI_API_KEY is available in all commands
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
load_dotenv(override=False)

import argparse
import sys


def cmd_ingest(args):
    """Clone, parse, chunk, embed, and index a repository."""
    from ingestion.repo_pipeline import RepoIngestionPipeline

    repo_url = args.repo_url

    print(f"\n{'='*60}")
    print(f"  Ingesting: {repo_url}")
    print(f"{'='*60}\n")

    pipeline = RepoIngestionPipeline()
    result = pipeline.ingest_repository(repo_url)

    print(f"\n✅ Successfully indexed {result.num_chunks} chunks from '{result.repo_name}'")


def cmd_query(args):
    """Ask a question about the indexed codebase."""
    from indexing.vector_store import ChromaVectorStore
    from retrieval.retriever import CodeRetriever
    from reasoning.answer_generator import AnswerGenerator
    from config import config, get_embedder

    embedder = get_embedder()
    vector_store = ChromaVectorStore(
        collection_name=config.chroma_collection,
        persist_dir=config.chroma_persist_dir,
    )
    retriever = CodeRetriever(embedder, vector_store, top_k=config.top_k)
    model = config.gemini_llm_model if config.llm_provider == "gemini" else config.llm_model
    generator = AnswerGenerator(retriever, model=model)

    result = generator.generate_answer(args.question)

    print(f"\n{'='*60}")
    print("  Answer")
    print(f"{'='*60}\n")
    print(result["answer"])

    print(f"\n{'='*60}")
    print("  Sources")
    print(f"{'='*60}\n")
    for src in result["sources"]:
        print(f"  • {src['file']} — {src['symbol']} ({src['type']}, lines {src['lines']})")


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    from config import config, get_embedder

    print(f"\n🚀 Starting API server on {config.api_host}:{config.api_port}\n")
    uvicorn.run("api.server:app", host=config.api_host, port=config.api_port, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="GitHub Codebase Intelligence System — Repo-level RAG",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a GitHub repository")
    ingest_parser.add_argument("repo_url", help="GitHub repo URL to ingest")

    # Query
    query_parser = subparsers.add_parser("query", help="Ask a question about the codebase")
    query_parser.add_argument("question", help="Natural language question")

    # Serve
    subparsers.add_parser("serve", help="Start the API server")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
