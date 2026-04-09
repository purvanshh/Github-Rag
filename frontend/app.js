document.addEventListener("DOMContentLoaded", () => {
  const queryInput = document.getElementById("query-input");
  const askBtn = document.getElementById("ask-btn");
  const answerBox = document.getElementById("answer-box");
  const diagramSelect = document.getElementById("diagram-select");
  const diagramBox = document.getElementById("diagram-box");
  const statusIndicator = document.getElementById("connection-status");
  const modelIndicator = document.getElementById("model-indicator");

  const API_BASE = "http://127.0.0.1:8000";
  const TEST_REPO = "Github-Rag";

  // Initialize Mermaid
  mermaid.initialize({ startOnLoad: false, theme: 'dark' });

  // Test API server connectivity
  fetch(`${API_BASE}/`)
    .then(r => r.json())
    .then(() => {
      statusIndicator.textContent = "API: Connected";
      statusIndicator.style.borderColor = "var(--accent-glow)";
      loadRepoOverview();
    })
    .catch(() => {
      statusIndicator.textContent = "API: Offline (Local Mode)";
      statusIndicator.style.borderColor = "#ef4444";
    });

  // Query handler
  askBtn.addEventListener("click", performQuery);
  queryInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") performQuery();
  });

  async function performQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    answerBox.innerHTML = "";
    askBtn.disabled = true;
    askBtn.textContent = "Asking...";

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo: TEST_REPO,
          query: query,
          conversation_id: "dashboard-conv"
        })
      });

      if (!response.body) {
        throw new Error("No response body for streaming.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep partial line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6).trim();
            if (!dataStr) continue;

            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.type === "token") {
                answerBox.innerHTML += parsed.text;
                answerBox.scrollTop = answerBox.scrollHeight;
              } else if (parsed.type === "error") {
                answerBox.innerHTML += `\n[Error: ${parsed.message}]`;
              }
            } catch (e) {
              // Ignore invalid JSON chunks
            }
          }
        }
      }
    } catch (err) {
      answerBox.innerHTML = `System error: Failed to fetch stream. Make sure the API server is running locally on port 8000.\n\nDetails: ${err.message}`;
    } finally {
      askBtn.disabled = false;
      askBtn.textContent = "Ask";
      // Add cursor back
      answerBox.innerHTML += '<span class="token-cursor"></span>';
    }
  }

  // Load Repo Overview and charts
  async function loadRepoOverview() {
    try {
      const resp = await fetch(`${API_BASE}/repo/${TEST_REPO}/overview`);
      const data = await resp.json();
      
      if (data.most_connected_modules) {
        document.getElementById("stat-connected").textContent = data.most_connected_modules.length;
      }
      if (data.most_called_functions) {
        document.getElementById("stat-called").textContent = data.most_called_functions.length;
      }

      modelIndicator.textContent = `Model: ${data.architecture_metadata?.model || 'OpenAI'}`;
      updateDiagramView();
    } catch (err) {
      console.warn("Failed to fetch repository overview", err);
    }
  }

  diagramSelect.addEventListener("change", updateDiagramView);

  async function updateDiagramView() {
    const selected = diagramSelect.value;
    diagramBox.innerHTML = "Generating diagram...";

    try {
      let endpoint = "dependency";
      if (selected === "class") endpoint = "class-hierarchy";
      if (selected === "sequence") endpoint = "sequence/main";

      const res = await fetch(`${API_BASE}/repo/${TEST_REPO}/diagrams/${endpoint}`);
      const text = await res.text();

      // Render chart
      const uniqueId = `mermaid-${Date.now()}`;
      diagramBox.innerHTML = `<div class="mermaid" id="${uniqueId}">${text}</div>`;
      await mermaid.run({ nodes: [document.getElementById(uniqueId)] });
    } catch (err) {
      diagramBox.innerHTML = `<div style="color:#ef4444">Failed to load Mermaid diagram. Ensure FastAPI server is running on port 8000.</div>`;
    }
  }
});
