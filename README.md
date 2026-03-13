# 🧠 GitHub Codebase Intelligence System

> **Repo-level RAG** — Clone any GitHub repository, parse it with Tree-sitter AST, build semantic code chunks, embed them, and answer natural language questions about the codebase with LLM-powered reasoning and file-level citations.

---

## ⚡ Quick Start

### 1. Clone & Set Up Environment

```bash
cd Github-Rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest a Repository

```bash
python main.py ingest https://github.com/karpathy/nanoGPT
```

### 4. Ask Questions

```bash
python main.py query "Where is the training loop implemented?"
```

### 5. Start the API Server

```bash
python main.py serve
```

### 6. Launch the UI (optional)

```bash
streamlit run ui/streamlit_app.py
```

---

## 🏗️ Architecture

```
GitHub Repo URL
     ↓
Repo Cloner (GitPython)
     ↓
Language Parser (Tree-sitter)
     ↓
Symbol Extractor (functions / classes / imports)
     ↓
Smart Code Chunking (by symbol, not token count)
     ↓
Embeddings (OpenAI text-embedding-3-large)
     ↓
Vector DB (ChromaDB)
     ↓
Retriever + Reranker
     ↓
LLM Reasoning (GPT-4o)
     ↓
Answer + File Citations
```

### Advanced Layer

```
Dependency Graph (networkx) — file-level import relationships
Call Graph (networkx)       — function-level call tracking
Architecture Summary        — LLM-generated system overview
```

---

## 📁 Project Structure

```
github-rag/
├── ingestion/
│   ├── clone_repo.py       # Clone GitHub repos via GitPython
│   ├── parse_code.py       # Tree-sitter AST parsing & symbol extraction
│   └── chunk_code.py       # Semantic code chunking by symbol
│
├── indexing/
│   ├── embedder.py         # OpenAI & local HuggingFace embedders
│   └── vector_store.py     # ChromaDB vector store
│
├── retrieval/
│   ├── retriever.py        # Query embedding + vector search
│   └── reranker.py         # Cross-encoder reranking (bge-reranker)
│
├── reasoning/
│   ├── prompt_templates.py # Structured prompts for LLM reasoning
│   └── answer_generator.py # GPT-4o answer generation with citations
│
├── graphs/
│   ├── dependency_graph.py # Import/dependency graph (networkx)
│   └── call_graph.py       # Function call graph (networkx)
│
├── api/
│   └── server.py           # FastAPI REST endpoints
│
├── ui/
│   └── streamlit_app.py    # Streamlit web interface
│
├── main.py                 # CLI entry point (ingest / query / serve)
├── config.py               # Centralized config via env vars
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
└── .gitignore
```

---

## 🔧 Tech Stack

| Layer        | Technology                          |
| ------------ | ----------------------------------- |
| Language     | Python 3.11+                        |
| Code Parsing | Tree-sitter                         |
| Embeddings   | OpenAI `text-embedding-3-large`     |
| Vector DB    | ChromaDB (local)                    |
| LLM          | GPT-4o                              |
| Reranker     | `BAAI/bge-reranker-large` (optional)|
| Graphs       | NetworkX                            |
| API          | FastAPI + Uvicorn                   |
| UI           | Streamlit                           |
| Git          | GitPython                           |

---

## 🧪 Recommended Test Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [tiangolo/fastapi](https://github.com/tiangolo/fastapi)
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain)

---

## 🗺️ Roadmap

- [x] Project structure & CLI
- [ ] Tree-sitter AST symbol extraction (per language)
- [ ] Semantic chunking pipeline
- [ ] Embedding & indexing pipeline
- [ ] Retriever + reranker
- [ ] LLM answer generation with citations
- [ ] Dependency graph builder
- [ ] Function call graph builder
- [ ] API server wiring
- [ ] Streamlit UI
- [ ] Architecture summary generation

---

## 📄 License

MIT
