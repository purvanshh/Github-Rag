"""
main.py — Entry point for the GitHub Codebase Intelligence System.

Provides a CLI for:
  - Ingesting a GitHub repo
  - Querying the indexed codebase
  - Starting the API server
"""

import argparse
import sys


def cmd_ingest(args):
    """Clone, parse, chunk, embed, and index a repository."""
    from ingestion.clone_repo import clone_repository
    from ingestion.parse_code import parse_directory
    from ingestion.chunk_code import create_chunks_from_symbols
    from indexing.embedder import OpenAIEmbedder
    from indexing.vector_store import ChromaVectorStore
    from config import config

    repo_url = args.repo_url
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    print(f"\n{'='*60}")
    print(f"  Ingesting: {repo_url}")
    print(f"{'='*60}\n")

    # Step 1: Clone
    print("[1/4] Cloning repository...")
    repo_path = clone_repository(repo_url)

    # Step 2: Parse with Tree-sitter
    print("[2/4] Parsing source code with Tree-sitter...")
    symbols = parse_directory(repo_path)
    print(f"       Found {len(symbols)} symbols")

    # Step 3: Create chunks
    print("[3/4] Creating semantic code chunks...")
    chunks = create_chunks_from_symbols(symbols, repo_name)
    print(f"       Created {len(chunks)} chunks")

    # Step 4: Embed and index
    print("[4/4] Generating embeddings & indexing...")
    embedder = OpenAIEmbedder(model=config.embedding_model)
    vector_store = ChromaVectorStore(
        collection_name=config.chroma_collection,
        persist_dir=config.chroma_persist_dir,
    )
    embeddings = embedder.embed_chunks(chunks)
    vector_store.add_chunks(chunks, embeddings)

    print(f"\n✅ Successfully indexed {len(chunks)} chunks from '{repo_name}'")


def cmd_query(args):
    """Ask a question about the indexed codebase."""
    from indexing.embedder import OpenAIEmbedder
    from indexing.vector_store import ChromaVectorStore
    from retrieval.retriever import CodeRetriever
    from reasoning.answer_generator import AnswerGenerator
    from config import config

    embedder = OpenAIEmbedder(model=config.embedding_model)
    vector_store = ChromaVectorStore(
        collection_name=config.chroma_collection,
        persist_dir=config.chroma_persist_dir,
    )
    retriever = CodeRetriever(embedder, vector_store, top_k=config.top_k)
    generator = AnswerGenerator(retriever, model=config.llm_model)

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
    from config import config

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
