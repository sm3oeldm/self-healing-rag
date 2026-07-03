"""
Pipeline nodes for the self-healing RAG graph.

Each node receives the current graph state, performs its job, and returns
an updated state dictionary.

Provides both synchronous variants (used by the LangGraph graph) and
async/streaming variants (used by the FastAPI server).
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.vectorstore.store import retrieve_chunks
from src.agents.critic import evaluate_answer, reformulate_query
from src.config import GOOGLE_API_KEY, LLM_MODEL, MAX_RETRIES

log = logging.getLogger("rag-api")

# ── Shared prompt strings ─────────────────────────────────────────────────────
GENERATE_SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided source documents.

Rules:
- Only use information from the provided sources
- If the sources don't contain enough information, say "I don't have enough information in the provided documents to answer this question."
- Do not use your general knowledge or training data
- Be concise and direct"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _format_context(chunks: list) -> str:
    """Format retrieved document chunks into a single context string."""
    return "\n\n".join([
        f"[Source {i + 1}]:\n{doc.page_content}"
        for i, doc in enumerate(chunks)
    ])


def _build_generate_messages(question: str, context_text: str) -> list:
    """Build the message list for the generate LLM call."""
    return [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=f"Sources:\n{context_text}\n\nQuestion: {question}\n\nAnswer based only on the sources above:"),
    ]


# ── Module-level LLM cache ────────────────────────────────────────────────────
_generator_llm = None


def _get_generator_llm():
    """Get or create the answer-generator LLM (low temperature for factual output)."""
    global _generator_llm
    if _generator_llm is None:
        _generator_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
    return _generator_llm


# ═══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS NODES (used by LangGraph sync graph)
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_node(state: dict) -> dict:
    """
    Node 1: Retrieve relevant chunks from ChromaDB for the current question.
    """
    print(f"\n[RETRIEVE] Searching for: '{state['question']}'")
    chunks = retrieve_chunks(state["question"], k=3)
    print(f"[RETRIEVE] Found {len(chunks)} chunk(s).")
    return {**state, "chunks": chunks}


def generate_node(state: dict) -> dict:
    """
    Node 2: Generate an answer using Gemini and the retrieved chunks as context.
    """
    print(f"\n[GENERATE] Generating answer...")

    llm = _get_generator_llm()
    context_text = _format_context(state["chunks"])
    messages = _build_generate_messages(state["question"], context_text)

    response = llm.invoke(messages)
    answer = response.content.strip()
    print(f"[GENERATE] Answer generated.")
    return {**state, "answer": answer}


def critique_node(state: dict) -> dict:
    """
    Node 3: Evaluate the generated answer against the retrieved chunks.
    - If PASS: set final_answer and finish
    - If FAIL with retries left: reformulate query and loop back
    - If FAIL with no retries left: return graceful fallback
    """
    print(f"\n[CRITIQUE] Evaluating answer...")

    result = evaluate_answer(
        question=state["question"],
        context_chunks=state["chunks"],
        answer=state["answer"]
    )

    verdict = result["verdict"]
    reason = result["reason"]

    print(f"[CRITIQUE] Verdict: {verdict}")
    print(f"[CRITIQUE] Reason: {reason}")

    # If PASS → set final answer and we're done
    if verdict == "PASS":
        return {
            **state,
            "verdict": verdict,
            "reason": reason,
            "final_answer": state["answer"]
        }

    # If FAIL and we have retries left → reformulate the query
    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRIES:
        new_query = reformulate_query(state["question"], reason)
        print(f"[CRITIQUE] Reformulated query: '{new_query}'")
        return {
            **state,
            "verdict": verdict,
            "reason": reason,
            "question": new_query,
            "retry_count": retry_count + 1
        }

    # If FAIL and no retries left → graceful fallback
    print(f"[CRITIQUE] Max retries reached. Returning fallback response.")
    return {
        **state,
        "verdict": "FAIL",
        "reason": reason,
        "final_answer": "I don't have enough information in the provided documents to answer this question accurately."
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC / STREAMING NODES (used by the FastAPI server)
# ═══════════════════════════════════════════════════════════════════════════════

async def aretrieve_node(state: dict) -> dict:
    """
    Async Node 1: Retrieve relevant chunks from ChromaDB.

    Currently wraps the synchronous retrieve; ChromaDB I/O is fast enough
    that this can run on the event loop without a thread-pool penalty.
    """
    log.info(f"[RETRIEVE] Searching for: '{state['question']}'")
    chunks = retrieve_chunks(state["question"], k=3)
    log.info(f"[RETRIEVE] Found {len(chunks)} chunk(s).")
    return {**state, "chunks": chunks}


async def astream_generate_tokens(state: dict) -> AsyncGenerator[str, None]:
    """
    Async Node 2 (streaming): Generate an answer, yielding tokens as they arrive.

    Yields each content token from the LLM and sets ``state["answer"]``
    to the full accumulated string when iteration finishes.

    Usage::

        async for token in astream_generate_tokens(state):
            print(token, end="", flush=True)
        # state["answer"] is now populated
    """
    log.info("[GENERATE] Streaming answer...")

    llm = _get_generator_llm()
    context_text = _format_context(state["chunks"])
    messages = _build_generate_messages(state["question"], context_text)

    full_answer: list[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content if hasattr(chunk, "content") and chunk.content else ""
        if content:
            full_answer.append(content)
            yield content

    state["answer"] = "".join(full_answer).strip()
    log.info(f"[GENERATE] Streamed {len(full_answer)} chunk(s), answer length={len(state['answer'])} chars.")


async def acritique_node(state: dict) -> dict:
    """
    Async Node 3: Evaluate the generated answer against the retrieved chunks.

    Same logic as ``critique_node`` but async-friendly (the LLM call inside
    ``evaluate_answer`` and ``reformulate_query`` is synchronous — this is
    fine since it runs in a short burst, not a stream).
    """
    log.info("[CRITIQUE] Evaluating answer...")

    result = evaluate_answer(
        question=state["question"],
        context_chunks=state["chunks"],
        answer=state["answer"]
    )

    verdict = result["verdict"]
    reason = result["reason"]

    log.info(f"[CRITIQUE] Verdict: {verdict}")
    log.info(f"[CRITIQUE] Reason: {reason}")

    if verdict == "PASS":
        return {
            **state,
            "verdict": verdict,
            "reason": reason,
            "final_answer": state["answer"],
        }

    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRIES:
        new_query = reformulate_query(state["question"], reason)
        log.info(f"[CRITIQUE] Reformulated query: '{new_query}'")
        return {
            **state,
            "verdict": verdict,
            "reason": reason,
            "question": new_query,
            "retry_count": retry_count + 1,
        }

    log.info("[CRITIQUE] Max retries reached. Returning fallback response.")
    return {
        **state,
        "verdict": "FAIL",
        "reason": reason,
        "final_answer": "I don't have enough information in the provided documents to answer this question accurately.",
    }
