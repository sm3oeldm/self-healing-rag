# Self-Healing RAG Pipeline 🤖🩹

A **Retrieval-Augmented Generation** pipeline that critiques its own outputs and retries on hallucinations — hence "self-healing."

Built with [LangGraph](https://langchain-ai.github.io/langgraph/), [Gemini](https://ai.google.dev/), [ChromaDB](https://www.trychroma.com/), and local HuggingFace embeddings.

---

## How It Works

The pipeline runs a **retrieve → generate → critique → (retry loop)** cycle powered by a LangGraph state graph:

```
        ┌──────────┐
        │ Retrieve │ ←── retry (reformulated query)
        └────┬─────┘
             ↓
        ┌──────────┐
        │ Generate │
        └────┬─────┘
             ↓
        ┌──────────┐
        │ Critique │
        └────┬─────┘
             │
    ┌────────┴────────┐
    │ PASS            │ FAIL (retries left)
    ↓                 ↓
   END          ──────┘
                (reformulate → Retrieve)
```

1. **Retrieve** — searches a ChromaDB vector store for relevant document chunks
2. **Generate** — answers using Gemini, grounded only in the retrieved chunks
3. **Critique** — a strict judge evaluates whether every claim is supported by the source chunks
4. **Self-Heal** — on failure, reformulates the query and retries (up to 1 retry by default)
5. **Fallback** — if retries are exhausted, returns an honest "I don't have enough information" response

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A free [Gemini API key](https://aistudio.google.com/) (set `GOOGLE_API_KEY` in `.env`)

### 2. Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd self-healing-rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY
```

### 3. Add your documents

Place `.txt` files in `data/docs/`. The project ships with sample docs about a fictional company called **NovaTech Inc.** (HR policy, company overview, product FAQ).

### 4. Run

**Interactive CLI:**

```bash
python main.py
```

The vector store builds automatically on first run. Then ask questions like:

```
Your question: How many vacation days do employees get?
Your question: What products does NovaTech offer?
Your question: quit
```

**Web API (FastAPI):**

```bash
python main.py --serve
```

Then hit `http://127.0.0.1:8000/docs` for the Swagger playground.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (vector store + LLM status) |
| `POST` | `/query` | Single RAG query, returns JSON |
| `POST` | `/query/stream` | Stream pipeline events + answer tokens via SSE |

---

## Project Structure

```
self-healing-rag/
├── main.py                    # Entry point (CLI or --serve)
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── data/
│   ├── docs/                  # Source .txt documents (your knowledge base)
│   └── chroma_db/             # Vector store (auto-built, git-ignored)
├── src/
│   ├── config.py              # Environment config & constants
│   ├── pipeline/
│   │   ├── graph.py           # LangGraph state graph definition
│   │   └── nodes.py           # Pipeline nodes (retrieve, generate, critique)
│   ├── agents/
│   │   └── critic.py          # LLM-based judge & query reformulator
│   ├── vectorstore/
│   │   └── store.py           # Document loading, chunking, embedding, retrieval
│   └── api/
│       ├── server.py          # FastAPI server & routes
│       └── schemas.py         # Pydantic request/response models
├── notebooks/                 # Jupyter notebooks (exploration)
└── test_score.py              # Vector store smoke test
```

---

## Configuration

Set these via `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | **Required.** Gemini API key |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model (e.g. `gemini-2.5-pro`, `gemini-2.0-flash`) |
| `SERVER_HOST` | `127.0.0.1` | FastAPI bind address |
| `SERVER_PORT` | `8000` | FastAPI port |
| `CORS_ORIGINS` | `*` | Comma-separated CORS origins |

---

## Key Design Features

- **Self-healing loop** — the critic catches hallucinations and triggers a reformulated retry before giving the user a bad answer
- **Local embeddings** — uses `all-MiniLM-L6-v2` via `sentence-transformers` (no embedding API key needed)
- **Graceful degradation** — when retries are exhausted, says "I don't have enough information" rather than guessing
- **Streaming** — SSE endpoint for real-time token output (great for chat UIs)
- **Module-level caching** — Graph, LLM, and vector store connections are built once and reused

---

## Testing

```bash
# Quick vector store smoke test
python test_score.py
```

---

## License

MIT
