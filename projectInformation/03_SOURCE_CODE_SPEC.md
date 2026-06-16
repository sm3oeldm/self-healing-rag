# Source Code Specification — All Files

This document contains the exact code for every file in the project.
Build them in the order listed below.

---

## 1. src/config.py

Purpose: Loads environment variables and defines shared constants used across all modules.

```python
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCS_PATH = "data/docs"
CHROMA_DB_PATH = "data/chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"
MAX_RETRIES = 1
```

---

## 2. src/vectorstore/store.py

Purpose: Loads .txt documents, splits them into chunks, embeds them using a local
SentenceTransformer model, stores them in ChromaDB, and retrieves relevant chunks
for a given query.

IMPORTANT: Uses SentenceTransformerEmbeddings (local, no API key needed).
Do NOT use GoogleGenerativeAIEmbeddings — it causes v1beta API conflicts.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.config import DOCS_PATH, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def load_documents():
    """Load all .txt files from the docs folder."""
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    return documents


def chunk_documents(documents):
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks with local SentenceTransformer and store in ChromaDB."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print(f"Vectorstore built and saved to '{CHROMA_DB_PATH}'.")
    return vectorstore


def load_vectorstore():
    """Load an already-built ChromaDB from disk."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def retrieve_chunks(query: str, k: int = 3):
    """Search the vectorstore and return the top-k relevant chunks."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results
```

---

## 3. src/agents/critic.py

Purpose: A critic agent that evaluates whether a generated answer is grounded in
the retrieved source chunks, or whether it contains hallucinated information.

Returns a dict with:
- "verdict": "PASS" or "FAIL"
- "reason": a short explanation of why it passed or failed

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
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
```

---

## 4. src/pipeline/nodes.py

Purpose: Defines the three core node functions used in the LangGraph graph.
Each node receives the current graph state, performs its job, and returns
an updated state dict.

The state object passed between nodes contains:
- question (str): the current query (may be reformulated on retry)
- original_question (str): the original user question (never changes)
- chunks (list): retrieved document chunks
- answer (str): the generated answer
- verdict (str): "PASS" or "FAIL" from the critic
- reason (str): critic's explanation
- retry_count (int): how many times we've retried
- final_answer (str): the answer to return to the user

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
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
    If it fails and we haven't exceeded max retries, reformulate the query.
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
```

---

## 5. src/pipeline/graph.py

Purpose: Defines the LangGraph stateful graph. Connects nodes with edges,
including a conditional edge after the critique node that either ends the
pipeline (PASS) or loops back to retrieval (FAIL with retries left).

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain.schema import Document
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


def run_pipeline(question: str) -> dict:
    """
    Run the full self-healing RAG pipeline for a given question.

    Args:
        question: The user's question

    Returns:
        The final state dict containing the answer and all intermediate info
    """
    app = build_graph()

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
```

---

## 6. main.py (entry point)

Purpose: The main script to run the full pipeline from the command line.
Also handles building the vectorstore if it doesn't exist yet.

```python
import os
from src.vectorstore.store import load_documents, chunk_documents, build_vectorstore
from src.pipeline.graph import run_pipeline
from src.config import CHROMA_DB_PATH


def setup_vectorstore():
    """Build the vectorstore if it doesn't exist yet."""
    if not os.path.exists(CHROMA_DB_PATH):
        print("Vector store not found. Building it now...")
        docs = load_documents()
        chunks = chunk_documents(docs)
        build_vectorstore(chunks)
        print("Vector store ready.\n")
    else:
        print("Vector store already exists. Skipping build.\n")


def main():
    # Step 1: Make sure the vectorstore is ready
    setup_vectorstore()

    # Step 2: Ask questions in a loop
    print("=" * 60)
    print("Self-Healing RAG Pipeline")
    print("Ask questions about NovaTech Inc.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\n" + "-" * 40)
        result = run_pipeline(question)

        print("\n" + "=" * 40)
        print("FINAL ANSWER:")
        print(result["final_answer"])
        print("=" * 40)
        print(f"Verdict: {result['verdict']}")
        print(f"Retries used: {result['retry_count']}")
        if result.get("reason"):
            print(f"Critic note: {result['reason']}")


if __name__ == "__main__":
    main()
```

---

## 7. test_store.py (vectorstore test script)

Purpose: Tests the vectorstore in isolation before running the full pipeline.

```python
from src.vectorstore.store import load_documents, chunk_documents, build_vectorstore, retrieve_chunks

# Step 1: Load
docs = load_documents()

# Step 2: Chunk
chunks = chunk_documents(docs)

# Step 3: Build vectorstore
build_vectorstore(chunks)

# Step 4: Test retrieval
query = "How many vacation days do employees get?"
results = retrieve_chunks(query)

print("\n--- Retrieved Chunks ---")
for i, chunk in enumerate(results):
    print(f"\nChunk {i+1}:\n{chunk.page_content}")
```
