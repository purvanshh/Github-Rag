"""
streamlit_app.py — Streamlit UI for the GitHub Codebase Intelligence System.

Provides an interface to:
1. Input a GitHub repo URL and trigger ingestion.
2. Ask natural language questions about the codebase.
3. View answers with file citations and code snippets.
4. Explore a repository dashboard (architecture, graphs, directory tree).
"""

from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000"


st.set_page_config(
    page_title="GitHub Codebase Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 GitHub Codebase Intelligence")
st.markdown("*Repo-level RAG with AST parsing, dependency graphs & LLM reasoning*")


if "current_repo" not in st.session_state:
    st.session_state["current_repo"] = ""


def _call_api(method: str, path: str, **kwargs):
    """Small helper to call the FastAPI backend with basic error handling."""
    url = f"{API_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=60, **kwargs)
    except requests.ConnectionError:
        st.error("Cannot connect to the API server. Is it running?")
        return None

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"Error ({resp.status_code}): {detail}")
        return None
    return resp.json()


# ---------- Sidebar: Repo Ingestion ----------
with st.sidebar:
    st.header("📦 Ingest Repository")
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")

    if st.button("Analyze Repository", type="primary"):
        if repo_url:
            with st.spinner("Cloning & indexing repository..."):
                data = _call_api("POST", "/ingest", json={"repo_url": repo_url})
                if data:
                    repo_name = data.get("repo_name") or ""
                    st.session_state["current_repo"] = repo_name
                    num_chunks = data.get("num_chunks", 0)
                    num_files = data.get("num_files", 0)
                    num_symbols = data.get("num_symbols", 0)
                    st.success(
                        f"✅ Indexed **{repo_name}** — "
                        f"{num_chunks} chunks "
                        f"({num_files} files, {num_symbols} symbols)"
                    )
        else:
            st.warning("Please enter a repo URL.")

    st.markdown("---")
    st.subheader("Current Repository")
    if st.session_state["current_repo"]:
        st.code(st.session_state["current_repo"])
    else:
        st.caption("No repository ingested yet.")


# ---------- Main layout ----------
tab_qa, tab_dashboard = st.tabs(["💬 Ask Questions", "📊 Repository Dashboard"])


with tab_qa:
    st.header("Ask About the Codebase")
    question = st.text_input(
        "Your question",
        placeholder="e.g. How does authentication work? Where is login_user used? Explain auth/service.py",
    )

    if st.button("Ask", type="primary", key="ask_button"):
        if not st.session_state["current_repo"]:
            st.warning("Please ingest a repository first.")
        elif not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                data = _call_api(
                    "POST",
                    "/query",
                    json={
                        "repo": st.session_state["current_repo"],
                        "query": question,
                    },
                )
                if data:
                    st.markdown("### Answer")
                    answer_text = data.get("answer") or "_No answer returned._"
                    st.markdown(answer_text)

                    sources = data.get("sources") or []
                    if sources:
                        st.markdown("### 📄 Sources")
                        for src in sources:
                            file_path = src.get("file", "<unknown>")
                            symbol = src.get("symbol", "<unknown>")
                            s_type = src.get("type", "<unknown>")
                            lines = src.get("lines", "?")
                            st.markdown(
                                f"- `{file_path}` — **{symbol}** "
                                f"({s_type}, lines {lines})"
                            )


with tab_dashboard:
    st.header("Repository Overview")

    if not st.session_state["current_repo"]:
        st.info("Ingest a repository first to see its dashboard.")
    else:
        repo_name = st.session_state["current_repo"]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Architecture Summary")
            overview = _call_api("GET", f"/repo/{repo_name}/overview")
            if overview:
                summary = overview.get("architecture_summary") or "No summary available."
                st.markdown(summary)

        with col2:
            if overview:
                st.subheader("Top Dependency Modules")
                hubs = overview.get("most_connected_modules", [])
                if hubs:
                    for name, degree in hubs:
                        st.markdown(f"- `{name}` (in-degree: {degree})")
                else:
                    st.caption("No dependency data.")

                st.subheader("Most Called Functions")
                most_called = overview.get("most_called_functions", [])
                if most_called:
                    for name, count in most_called:
                        st.markdown(f"- `{name}` (called {count} times)")
                else:
                    st.caption("No call-graph data.")

        st.markdown("---")
        st.subheader("Directory Tree")
        if overview:
            tree = overview.get("directory_tree", "")
            st.code(tree or "No directory information available.")

