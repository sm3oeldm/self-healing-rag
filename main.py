"""
Self-Healing RAG Pipeline — Main Entry Point

Run modes:
  python main.py           Interactive CLI (original behaviour)
  python main.py --serve   Start the FastAPI web server
  python main.py --help    Show usage
"""

from __future__ import annotations

import argparse
import os
import sys


def setup_vectorstore():
    """Build the vectorstore if it doesn't exist yet."""
    from src.config import CHROMA_DB_PATH
    from src.vectorstore.store import load_documents, chunk_documents, build_vectorstore

    if not os.path.isdir(CHROMA_DB_PATH):
        print("Vector store not found. Building it now...")
        docs = load_documents()
        chunks = chunk_documents(docs)
        build_vectorstore(chunks)
        print("Vector store ready.\n")
    else:
        print("Vector store already exists. Skipping build.\n")


def run_cli():
    """Original interactive CLI loop."""
    from src.pipeline.graph import run_pipeline

    setup_vectorstore()

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

        if question.lower() in ("quit", "exit", "q"):
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


def run_server():
    """Start the FastAPI web server."""
    from src.api.server import run_server as _start
    _start()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Healing RAG Pipeline",
    )
    parser.add_argument(
        "--serve", "-s",
        action="store_true",
        help="Start the FastAPI web server instead of the interactive CLI.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override the server host (default: from .env or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the server port (default: from .env or 8000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.serve:
        # Allow CLI overrides for host/port
        if args.host:
            os.environ.setdefault("SERVER_HOST", args.host)
        if args.port:
            os.environ.setdefault("SERVER_PORT", str(args.port))
        setup_vectorstore()
        run_server()
    else:
        run_cli()


if __name__ == "__main__":
    main()
