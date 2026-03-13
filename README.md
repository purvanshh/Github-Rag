# 🧠 GitHub Codebase Intelligence System

> **Repo‑level RAG for GitHub repositories.**  
> Clone any repo, parse it with Tree‑sitter, build semantic code indexes and graphs, then answer architectural questions via an LLM with file‑level citations.

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/purvanshh/github-rag.git
cd Github-Rag

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

Set your OpenAI key and (optionally) override defaults:

```bash
export OPENAI_API_KEY=sk-...
export EMBEDDING_MODEL=text-embedding-3-large      # optional
export LLM_MODEL=gpt-4o                            # optional
export REPOS_DIR=./repos                           # optional
export CHROMA_PERSIST_DIR=./chroma_db              # optional
```

### 3. Ingest a repository (CLI)

```bash
python main.py ingest https://github.com/karpathy/nanoGPT
```

This will:
- clone the repo into `REPOS_DIR/nanoGPT`
- parse & chunk the code
- embed chunks & store them in Chroma
- build dependency and call graphs

### 4. Ask questions (CLI)

```bash
python main.py query "Where is the training loop implemented?"
```

### 5. Start the API server

```bash
python main.py serve
# or:
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 6. Launch the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

In the UI you can:
- paste a GitHub repo URL and click **Analyze Repository**
- ask questions like:
  - `How does authentication work?`
  - `Where is authenticate_user used?`
  - `Explain file auth/service.py`
- see an architecture dashboard (summary, dependency hubs, call‑graph hotspots, directory tree)

---

## 🏗️ System Architecture

### Ingestion pipeline

```text
GitHub Repo URL
     ↓
RepoIngestionPipeline (ingestion/repo_pipeline.py)
  1. Clone repo (GitPython)
  2. Parse code (Tree-sitter)
  3. Extract symbols (functions / classes / methods / imports)
  4. Smart code chunking (by symbol, not token count)
  5. Embeddings (OpenAI text-embedding-3-large)
  6. Vector DB (ChromaDB)
  7. Dependency graph (NetworkX)
  8. Call graph (NetworkX)
     ↓
Repo ready for analysis
```

### Query pipeline

```text
User Query
     ↓
QueryRouter (intent classification)
     ↓
RepoAnalyzer (per-repo orchestrator)
     ↓
GraphAwareRetriever
  - Vector similarity search
  - Expand via dependency graph (imports & dependents)
  - Expand via call graph (callers & callees)
  - Cross-encoder reranking (bge-reranker-large)
     ↓
AnswerGenerator (LLM)
  - Builds prompt with retrieved context
  - Calls GPT-4o
  - Normalizes sources for UI
     ↓
Answer + file/symbol/line citations
```

---

## 📁 Project Structure

```text
github-rag/
├── ingestion/
│   ├── clone_repo.py         # Clone/pull GitHub repos via GitPython
│   ├── parse_code.py         # Tree-sitter parsing & symbol extraction
│   ├── chunk_code.py         # Semantic code chunking around symbols
│   └── repo_pipeline.py      # End-to-end ingestion orchestration
│
├── indexing/
│   ├── embedder.py           # OpenAI + local embedding backends
│   └── vector_store.py       # ChromaDB vector store abstraction
│
├── retrieval/
│   ├── retriever.py          # Basic hybrid retriever (vector + reranker)
│   ├── graph_aware_retriever.py  # Graph-aware hybrid retriever
│   └── reranker.py           # Cross-encoder reranking (bge-reranker-large)
│
├── graphs/
│   ├── dependency_graph.py   # File-level import/dependency graph (NetworkX)
│   └── call_graph.py         # Function-level call graph (NetworkX + Tree-sitter)
│
├── reasoning/
│   ├── prompt_templates.py   # Structured prompts for QA & architecture
│   ├── answer_generator.py   # GPT-4o answer generation + normalized sources
│   ├── architecture_summarizer.py # LLM-based repo architecture summaries
│   ├── repo_analyzer.py      # High-level orchestration for a single repo
│   └── query_router.py       # Intent classification & routing to RepoAnalyzer
│
├── api/
│   └── server.py             # FastAPI REST API (ingest/query/overview/graphs)
│
├── ui/
│   └── streamlit_app.py      # Streamlit UI: ingestion, QA, dashboard
│
├── graphs/__init__.py
├── ingestion/__init__.py
├── main.py                   # CLI entry point (ingest / query / serve)
├── config.py                 # Centralized configuration via env vars
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🔧 Tech Stack

| Layer           | Technology                           |
| -------------- | ------------------------------------- |
| Language       | Python 3.10+                          |
| Code parsing   | Tree-sitter (Python, JS, TS)          |
| Embeddings     | OpenAI `text-embedding-3-large`       |
| Vector DB      | ChromaDB (local)                      |
| LLM            | GPT-4o                                |
| Reranker       | `BAAI/bge-reranker-large`             |
| Graphs         | NetworkX                              |
| API            | FastAPI + Uvicorn                     |
| UI             | Streamlit                             |
| Git integration| GitPython                             |

---

## 🧠 Core Components

- **RepoIngestionPipeline**
  - Single entrypoint to prepare a repo:
    - clone → parse → chunk → embed → index → build graphs → store metadata.

- **RepoAnalyzer**
  - Per‑repo orchestrator:
    - `ask_question(query)`
    - `get_architecture_summary()`
    - `find_function_usage(function_name)`
    - `get_file_dependencies(file_path)`
    - `explain_file(file_path)`
    - `get_repo_overview()`

- **QueryRouter**
  - Classifies queries into:
    - `architecture`, `function_usage`, `file_dependencies`,
      `file_explanation`, `repo_overview`, `code_question`
  - Routes to the appropriate `RepoAnalyzer` method.

- **GraphAwareRetriever**
  - Vector similarity search in Chroma.
  - Graph expansion via:
    - dependency graph (imported/importing files)
    - call graph (callers/callees)
  - Deduplicates candidates and reranks with `bge-reranker-large`.

- **AnswerGenerator**
  - Builds prompts from retrieved context.
  - Calls GPT‑4o via OpenAI SDK.
  - Returns:
    - `answer` (markdown)
    - `sources` (file/symbol/type/lines)
    - `model`

---

## 🧪 Recommended Test Repos

- [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT)
- [`tiangolo/fastapi`](https://github.com/tiangolo/fastapi)
- [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain)

---

## 🗺️ Roadmap (High Level)

- [x] Project structure & CLI
- [x] Tree-sitter AST symbol extraction (Python, JS, TS)
- [x] Semantic chunking pipeline
- [x] Embedding & indexing pipeline (Chroma)
- [x] Hybrid retriever + cross-encoder reranker
- [x] Graph-aware retrieval (dependency + call graphs)
- [x] LLM answer generation with citations
- [x] Dependency graph builder
- [x] Function call graph builder
- [x] API server wiring (FastAPI)
- [x] Streamlit UI (ingestion, QA, dashboard)
- [x] Architecture summary generation

---

## 📄 License

MIT

