"""
prompt_templates.py — Prompt templates for LLM reasoning over code.

Contains structured prompts that guide the LLM to produce accurate,
well-cited answers about the codebase.
"""


SYSTEM_PROMPT = """You are a senior software architect with deep expertise in reading and understanding codebases.

Your task is to answer questions about a codebase using ONLY the provided code context.

Rules:
1. Base your answer strictly on the provided code context.
2. Reference specific files and functions in your answer.
3. Explain the architecture and how components interact when relevant.
4. If the context is insufficient to answer the question, say so clearly.
5. Use code snippets to support your explanations when helpful.
"""


QA_PROMPT_TEMPLATE = """## Code Context

{context}

---

## Question

{question}

---

## Instructions

Answer the question using the code context above. Structure your answer as:
1. **Summary** — Brief direct answer.
2. **Relevant Files & Functions** — List the key files and symbols involved.
3. **Explanation** — Detailed walkthrough of the implementation.
4. **Architecture Notes** — How this fits into the broader system (if applicable).

If you cannot determine the answer from the given context, state that clearly.
"""


ARCHITECTURE_PROMPT_TEMPLATE = """## Repository Structure

{file_tree}

## Code Context

{context}

---

Provide a high-level architecture summary of this codebase:
1. **System Overview** — What this project does.
2. **Main Modules** — Key packages/directories and their responsibilities.
3. **Data Flow** — How data moves through the system.
4. **Key Design Patterns** — Notable patterns used in the codebase.
"""
