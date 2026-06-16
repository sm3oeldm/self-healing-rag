"""
Critic agent for the self-healing RAG pipeline.

Evaluates whether a generated answer is grounded in the retrieved source chunks,
and reformulates failed queries for retry.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import GOOGLE_API_KEY, LLM_MODEL


def evaluate_answer(question: str, context_chunks: list, answer: str) -> dict:
    """
    Evaluate whether the answer is grounded in the retrieved chunks.

    Args:
        question: The original user question
        context_chunks: List of LangChain Document objects retrieved from ChromaDB
        answer: The generated answer to evaluate

    Returns:
        dict with keys "verdict" ("PASS" or "FAIL") and "reason" (str)
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # Format the context chunks into a readable string
    context_text = "\n\n".join([
        f"[Chunk {i+1}]:\n{doc.page_content}"
        for i, doc in enumerate(context_chunks)
    ])

    system_prompt = """You are a strict fact-checking critic for a RAG (Retrieval-Augmented Generation) system.

Your job is to evaluate whether a generated answer is fully grounded in the provided source chunks.

Rules:
- If every claim in the answer can be traced back to the source chunks → respond with PASS
- If the answer contains ANY information not found in the source chunks → respond with FAIL
- If the answer says "I don't have enough information" or similar → respond with PASS
- Be strict. Even one unsupported claim = FAIL.

Respond in exactly this format:
VERDICT: PASS or FAIL
REASON: one sentence explaining your decision"""

    user_prompt = f"""Question: {question}

Source Chunks:
{context_text}

Generated Answer:
{answer}

Evaluate the answer against the source chunks."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()

    # Parse the structured response
    lines = response_text.split("\n")
    verdict = "FAIL"
    reason = "Could not parse critic response."

    for line in lines:
        if line.startswith("VERDICT:"):
            verdict = "PASS" if "PASS" in line else "FAIL"
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {"verdict": verdict, "reason": reason}


def reformulate_query(original_query: str, failed_reason: str) -> str:
    """
    Use the LLM to reformulate a failed query into a more specific one.

    Args:
        original_query: The original question that led to a hallucinated answer
        failed_reason: The critic's reason for rejecting the answer

    Returns:
        A reformulated query string
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    prompt = f"""A RAG system failed to retrieve good information for this question:

Original question: {original_query}
Failure reason: {failed_reason}

Reformulate the question to be more specific and likely to retrieve better results.
Return ONLY the reformulated question, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
