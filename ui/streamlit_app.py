"""
streamlit_app.py — Streamlit UI for the GitHub Codebase Intelligence System.

Provides a simple interface to:
1. Input a GitHub repo URL and trigger ingestion.
2. Ask natural language questions about the codebase.
3. View answers with file citations and code snippets.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"


st.set_page_config(
    page_title="GitHub Codebase Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 GitHub Codebase Intelligence")
st.markdown("*Repo-level RAG with AST parsing, dependency graphs & LLM reasoning*")

# ---------- Sidebar: Repo Ingestion ----------
with st.sidebar:
    st.header("📦 Ingest Repository")
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")
    
    if st.button("Ingest Repo", type="primary"):
        if repo_url:
            with st.spinner("Cloning & indexing repository..."):
                try:
                    resp = requests.post(f"{API_URL}/ingest", json={"repo_url": repo_url})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ Indexed **{data['repo_name']}** — {data['chunks_indexed']} chunks")
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                except requests.ConnectionError:
                    st.error("Cannot connect to the API server. Is it running?")
        else:
            st.warning("Please enter a repo URL.")

# ---------- Main: Question Answering ----------
st.header("💬 Ask About the Codebase")

question = st.text_input("Your question", placeholder="Where is authentication implemented?")

if st.button("Ask", type="primary"):
    if question:
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    
                    st.markdown("### Answer")
                    st.markdown(data["answer"])
                    
                    st.markdown("### 📄 Sources")
                    for src in data["sources"]:
                        st.markdown(
                            f"- `{src['file']}` — **{src['symbol']}** "
                            f"({src['type']}, lines {src['lines']})"
                        )
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
            except requests.ConnectionError:
                st.error("Cannot connect to the API server. Is it running?")
    else:
        st.warning("Please enter a question.")
