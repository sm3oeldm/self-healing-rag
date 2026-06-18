"""
Self-Healing RAG Pipeline — Main Entry Point

Run this script to start the interactive question-answering loop.
The vectorstore is automatically built on first run.
"""

import os
from src.vectorstore.store import load_documents, chunk_documents, build_vectorstore
from src.pipeline.graph import run_pipeline
from src.config import CHROMA_DB_PATH


def setup_vectorstore():
    """Build the vectorstore if it doesn't exist yet."""
    if not os.path.isdir(CHROMA_DB_PATH):
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
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\n" + "-" * 40)
        try:
            result = run_pipeline(question)
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            continue

        print("\n" + "=" * 40)
        print("FINAL ANSWER:")
        print(result.get("final_answer", "No answer generated."))
        print("=" * 40)
        print(f"Verdict: {result.get('verdict', 'N/A')}")
        print(f"Retries used: {result.get('retry_count', 0)}")
        if result.get("reason"):
            print(f"Critic note: {result['reason']}")


if __name__ == "__main__":
    main()
