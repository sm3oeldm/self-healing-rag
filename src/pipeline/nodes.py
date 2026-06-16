"""
Pipeline nodes for the self-healing RAG graph.

Each node receives the current graph state, performs its job, and returns
an updated state dictionary.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.vectorstore.store import retrieve_chunks
from src.agents.critic import evaluate_answer, reformulate_query
from src.config import GOOGLE_API_KEY, LLM_MODEL, MAX_RETRIES


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

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    # Format chunks into context string
    context_text = "\n\n".join([
        f"[Source {i+1}]:\n{doc.page_content}"
        for i, doc in enumerate(state["chunks"])
    ])

    system_prompt = """You are a helpful assistant that answers questions strictly based on the provided source documents.

Rules:
- Only use information from the provided sources
- If the sources don't contain enough information, say "I don't have enough information in the provided documents to answer this question."
- Do not use your general knowledge or training data
- Be concise and direct"""

    user_prompt = f"""Sources:
{context_text}

Question: {state['question']}

Answer based only on the sources above:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

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
