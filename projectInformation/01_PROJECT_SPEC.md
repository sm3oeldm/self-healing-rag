# Self-Healing RAG Pipeline — Master Project Specification

## What This Project Is

A Retrieval-Augmented Generation (RAG) pipeline that doesn't just retrieve and generate — it critiques its own output and retries when it detects hallucinations. Built with LangGraph to model the pipeline as a stateful, cyclical graph (not a simple linear chain).

---

## The Problem This Solves

Standard RAG pipelines have a critical flaw: even when given relevant context chunks, the LLM can still hallucinate — it may ignore the retrieved documents and answer from its training data, fill gaps with made-up information, or sound confident while being completely wrong.

This project fixes that by adding a critic agent and a retry loop.

---

## How It Works (Full Flow)

```
User Question
     ↓
[NODE 1 — Retrieve]
Search ChromaDB vector store for the top 3 most semantically similar chunks
     ↓
[NODE 2 — Generate]
Gemini LLM generates an answer using the retrieved chunks as context
     ↓
[NODE 3 — Critique]
A second Gemini call acts as a critic/fact-checker:
"Is this answer grounded in the retrieved chunks, or did the model hallucinate?"
     ↓
         ┌────────────────────────────────────────┐
         │                                        │
      PASS ✅                                 FAIL ❌
         │                                        │
  Return answer                    Reformulate the query and
  to the user                      loop back to Node 1 (retry)
                                             │
                                   If it fails again →
                                   Return graceful fallback:
                                   "I don't have enough information
                                    to answer this question."
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Primary language |
| LangGraph 0.4.8 | Models the pipeline as a stateful, cyclical graph |
| LangChain 0.3.25 | Orchestration framework for LLM calls and document loading |
| langchain-community 0.3.24 | Document loaders, ChromaDB vectorstore wrapper |
| langchain-google-genai 2.1.4 | LangChain connector to Gemini LLM |
| langchain-text-splitters 0.3.8 | Document chunking |
| ChromaDB 1.0.12 | Local vector store for storing and searching embeddings |
| SentenceTransformers (all-MiniLM-L6-v2) | Local embedding model (no API key needed) |
| Gemini (gemini-1.5-flash) | LLM for both generation and critique |
| python-dotenv | Loads API keys from .env file |
| Google AI Studio API Key | Free tier Gemini API access |

---

## Project Folder Structure

```
self-healing-rag/
├── data/
│   ├── docs/                      ← Source .txt documents (NovaTech fake company docs)
│   │   ├── company_overview.txt
│   │   ├── hr_policy.txt
│   │   └── product_faq.txt
│   └── chroma_db/                 ← Auto-generated ChromaDB vector store (do not edit)
├── src/
│   ├── __init__.py
│   ├── config.py                  ← Loads env vars, defines shared constants
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── store.py               ← Load, chunk, embed, store, and retrieve documents
│   ├── agents/
│   │   ├── __init__.py
│   │   └── critic.py              ← Critic agent: evaluates if answer is grounded
│   └── pipeline/
│       ├── __init__.py
│       ├── nodes.py               ← Retrieve, Generate, Critique node functions
│       └── graph.py               ← LangGraph graph definition and conditional edges
├── test_store.py                  ← Test script for vectorstore only
├── main.py                        ← Entry point: run the full pipeline
├── .env                           ← Secret API keys (never committed to Git)
├── .env.example                   ← Template showing required env vars
├── requirements.txt               ← All pinned Python dependencies
└── README.md                      ← Project documentation
```

---

## Environment Variables Required

File: `.env` (copy from `.env.example`)

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get a free API key at: https://aistudio.google.com

---

## Source Documents

Three fake company documents for NovaTech Inc. stored in `data/docs/`:

- `company_overview.txt` — Company history, CEO, headcount, revenue, offices
- `hr_policy.txt` — Vacation days, remote work policy, health benefits, performance reviews
- `product_faq.txt` — DataPilot product description, integrations, pricing, support, security

These exist to provide realistic, queryable content for testing the RAG pipeline.

---

## What "Self-Healing" Means Specifically

1. The critic agent receives both the retrieved chunks AND the generated answer
2. It is prompted to act as a strict fact-checker: does every claim in the answer appear in the chunks?
3. If the answer contains information NOT found in the chunks → FAIL
4. On FAIL: the pipeline reformulates the query (makes it more specific) and retries retrieval
5. On second FAIL: the pipeline returns a graceful "I don't have enough information" response
6. Maximum retry attempts: 1 (so the graph can loop at most once before graceful fallback)
