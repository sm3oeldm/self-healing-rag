"""
LangGraph state graph for the self-healing RAG pipeline.

Defines the state type, builds the graph with conditional edges,
and provides the run_pipeline entry point.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from src.pipeline.nodes import retrieve_node, generate_node, critique_node


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


# ── BUILD THE GRAPH ──────────────────────────────────────────────────────────
def build_graph():
    """
    Builds and compiles the LangGraph self-healing RAG graph.

    Graph structure:
        retrieve → generate → critique → (conditional)
                                              ↓ PASS → END
                                              ↓ FAIL → retrieve (retry loop)
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Add linear edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")

    # Add conditional edge after critique
    graph.add_conditional_edges(
        "critique",
        should_retry,
        {
            "end": END,
            "retrieve": "retrieve"
        }
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

    initial_state = RAGState(
        question=question,
        original_question=question,
        chunks=[],
        answer="",
        verdict="",
        reason="",
        retry_count=0,
        final_answer=None
    )

    final_state = app.invoke(initial_state)
    return final_state
