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