"""
LangGraph state graph for the self-healing RAG pipeline.

Defines the state type, builds the graph with conditional edges,
and provides the run_pipeline entry point.

Also provides async and step-streaming entry points for the FastAPI server.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.pipeline.nodes import (
    acritique_node,
    aretrieve_node,
    astream_generate_tokens,
    critique_node,
    generate_node,
    retrieve_node,
)
log = logging.getLogger("rag-api")


# ── STATE DEFINITION ─────────────────────────────────────────────────────────


class RAGState(TypedDict):
    """
    The state object that flows through every node in the graph.

    Each node receives this, does its job, and returns an updated version.
    """
    question: str               # Current query (may be reformulated)
    original_question: str      # Original user question (never changes)
    chunks: List[Document]      # Retrieved document chunks
    answer: str                 # Generated answer
    verdict: str                # "PASS" or "FAIL" from critic
    reason: str                 # Critic's explanation
    retry_count: int            # Number of retries so far
    final_answer: Optional[str] # The answer to return to the user


# ── HELPER ────────────────────────────────────────────────────────────────────
def _make_initial_state(question: str) -> dict:
    """Return a fresh state dict for the given question."""
    return {
        "question": question,
        "original_question": question,
        "chunks": [],
        "answer": "",
        "verdict": "",
        "reason": "",
        "retry_count": 0,
        "final_answer": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS PIPELINE (LangGraph graph)
# ═══════════════════════════════════════════════════════════════════════════════

# ── CONDITIONAL EDGE FUNCTION ────────────────────────────────────────────────
def should_retry(state: RAGState) -> str:
    """
    Decides what happens after the critique node:
    - If final_answer is set → go to END
    - If verdict is FAIL and retries remain → go back to retrieve
    """
    if state.get("final_answer"):
        return "end"
    if state.get("verdict") == "FAIL":
        return "retrieve"
    return "end"


def build_graph():
    """
    Builds and compiles the LangGraph self-healing RAG graph.

    Graph structure:
        retrieve → generate → critique → (conditional)
                                              ↓ PASS → END
                                              ↓ FAIL → retrieve (retry loop)
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")

    graph.add_conditional_edges(
        "critique",
        should_retry,
        {
            "end": END,
            "retrieve": "retrieve",
        },
    )

    return graph.compile()


# Module-level cache — build the graph once, reuse for every question
_app = None


def run_pipeline(question: str) -> dict:
    """
    Run the full self-healing RAG pipeline for a given question.

    Args:
        question: The user's question

    Returns:
        The final state dict containing the answer and all intermediate info
    """
    global _app
    if _app is None:
        _app = build_graph()
    app = _app

    initial_state = _make_initial_state(question)
    final_state = app.invoke(initial_state)
    return dict(final_state)


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC PIPELINE (simple async wrapper around the sync graph via thread-pool)
# ═══════════════════════════════════════════════════════════════════════════════

async def arun_pipeline(question: str) -> dict:
    """
    Async entry point. Runs the compiled LangGraph in a thread-pool executor
    so the event loop is not blocked during LLM calls.

    Args:
        question: The user's question

    Returns:
        The final state dict containing the answer and all intermediate info
    """
    import asyncio
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_pipeline, question)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP-STREAMING PIPELINE (yields events; does NOT use LangGraph itself)
# ═══════════════════════════════════════════════════════════════════════════════

async def stream_pipeline(question: str) -> AsyncGenerator[dict, None]:
    """
    Run the pipeline one step at a time, yielding structured event dicts
    suitable for Server-Sent Events.

    This bypasses LangGraph's graph engine in favour of a manual loop so
    we can:
      1. Yield rich step-start / step-end events
      2. Stream answer tokens from the LLM as they arrive
      3. Yield a retry event before looping back

    Event types:
      ``{"type": "step", "data": {"step": "retrieve", "status": "start"}}``
      ``{"type": "step", "data": {"step": "retrieve", "status": "end", "chunk_count": 3}}``
      ``{"type": "step", "data": {"step": "generate", "status": "start"}}``
      ``{"type": "token", "data": {"content": "The"}}``
      ``{"type": "step", "data": {"step": "generate", "status": "end"}}``
      ``{"type": "step", "data": {"step": "critique", "status": "start"}}``
      ``{"type": "step", "data": {"step": "critique", "status": "end", "verdict": "PASS"}}``
      ``{"type": "retry", "data": {"retry_count": 1, "new_query": "…"}}``
      ``{"type": "result", "data": {"answer": "…", "verdict": "PASS", …}}``
      ``{"type": "error", "data": {"message": "…"}}``
    """

    def _event(etype: str, **kwargs) -> dict:
        return {"type": etype, "data": kwargs}

    state = _make_initial_state(question)

    while True:
        try:
            # ── RETRIEVE ──────────────────────────────────────────────
            yield _event("step", step="retrieve", status="start")
            state = await aretrieve_node(state)
            yield _event("step", step="retrieve", status="end",
                         chunk_count=len(state["chunks"]))

            # ── GENERATE (streaming) ──────────────────────────────────
            yield _event("step", step="generate", status="start")
            async for token in astream_generate_tokens(state):
                yield _event("token", content=token)
            yield _event("step", step="generate", status="end")

            # ── CRITIQUE ──────────────────────────────────────────────
            yield _event("step", step="critique", status="start")
            state = await acritique_node(state)
            yield _event("step", step="critique", status="end",
                         verdict=state.get("verdict", "FAIL"))

            # ── CHECK RESULT ──────────────────────────────────────────
            if state.get("final_answer") is not None:
                yield _event("result",
                             answer=state["final_answer"],
                             verdict=state.get("verdict", ""),
                             retry_count=state.get("retry_count", 0),
                             reason=state.get("reason"),
                             original_question=state.get("original_question", question))
                return

            # ── RETRY? ────────────────────────────────────────────────
            # acritique_node already decided whether to retry: if it wants
            # a retry, verdict is FAIL and final_answer is None (it won't
            # set final_answer until retries are exhausted).  If it has
            # exhausted retries it sets final_answer to the fallback, which
            # is caught by the final_answer check above.
            if state.get("verdict") == "FAIL" and state.get("final_answer") is None:
                yield _event("retry",
                             retry_count=state.get("retry_count", 0),
                             new_query=state["question"])
                continue  # loop back to retrieve

            # Safety net: should not normally be reached
            yield _event("result",
                         answer="I don't have enough information in the provided "
                                "documents to answer this question accurately.",
                         verdict="FAIL",
                         retry_count=retry_count,
                         reason=state.get("reason"),
                         original_question=state.get("original_question", question))
            return

        except Exception as exc:
            log.exception("Stream pipeline error")
            yield _event("error", message=str(exc))
            return
