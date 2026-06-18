import os
from src.vectorstore.store import load_documents, chunk_documents, build_vectorstore, retrieve_chunks
from src.config import CHROMA_DB_PATH


def test_vectorstore():
    """Test the full vectorstore pipeline — load, chunk, build, and retrieve."""
    # Step 1-3: Only build if the vectorstore doesn't already exist
    if not os.path.isdir(CHROMA_DB_PATH):
        print("Building vectorstore from scratch...")
        docs = load_documents()
        chunks = chunk_documents(docs)
        build_vectorstore(chunks)
    else:
        print("Vectorstore already exists. Skipping build.\n")

    # Step 4: Test retrieval
    query = "How many vacation days do employees get?"
    results = retrieve_chunks(query)

    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(results):
        print(f"\nChunk {i+1}:\n{chunk.page_content}")


if __name__ == "__main__":
    test_vectorstore()