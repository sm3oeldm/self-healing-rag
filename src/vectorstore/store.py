# This file has 3 jobs:
#   1. Load your .txt files from data/docs/
#   2. Chunk them into smaller pieces
#   3. Embed those chunks and store them in ChromaDB
#
# Uses SentenceTransformer (local, no API key needed) for embeddings.


import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.config import DOCS_PATH, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


# ── 1. LOAD ──────────────────────────────────────────────────────────────────
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


# ── 2. CHUNK ─────────────────────────────────────────────────────────────────
def chunk_documents(documents):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks


# ── 3. EMBED & STORE ─────────────────────────────────────────────────────────
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


# ── 4. LOAD EXISTING STORE ───────────────────────────────────────────────────
def load_vectorstore():
    """Load an already-built ChromaDB from disk."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore


# ── 5. RETRIEVE ──────────────────────────────────────────────────────────────
def retrieve_chunks(query: str, k: int = 3):
    """Search the vectorstore and return the top-k relevant chunks."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results
