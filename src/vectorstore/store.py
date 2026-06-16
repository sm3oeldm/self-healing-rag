# This file has 3 jobs:
#   1. Loaad your .txt files from data/docs/
#   2. Chunks them into smaller pieces 
#   3. Embed those chunks and store them in ChromaDB


import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import GOOGLE_API_KEY, DOCS_PATH, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP


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
    """Embed chunks and store them in ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )
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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )
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


"""
Section 1 — load_documents()

DirectoryLoader walks through your data/docs/ folder and reads every .txt file. Each file becomes a LangChain Document object — basically
a wrapper that holds the text content plus metadata like the filename.


Section 2 — chunk_documents()

RecursiveCharacterTextSplitter is the smartest built-in splitter LangChain offers. It tries to split on paragraphs first, then sentences, then
words — always trying to keep meaning intact rather than cutting arbitrarily.


Section 3 — build_vectorstore()

This is the most important step. Two things happen here:

GoogleGenerativeAIEmbeddings sends each chunk to Google's embedding model, which returns a vector (list of numbers) for each chunk
Chroma.from_documents() stores all those vectors in ChromaDB on your disk at data/chroma_db/


Why persist_directory? Without this, ChromaDB only lives in memory and disappears when your script stops. By giving it a folder path, it saves
to disk so you only have to build it once — not every time you run the app.

Section 4 — load_vectorstore()

Next time you run the app, you don't rebuild from scratch. You just load the already-saved ChromaDB from disk. Much faster.
Section 5 — retrieve_chunks()

This is what gets called during the actual RAG pipeline. It takes a user's question, converts it to an embedding, searches ChromaDB for the
k most similar chunks, and returns them.

What does k=3 mean? It means "give me the 3 most relevant chunks." You can increase this to get more context, but too many chunks can confuse
the LLM or exceed its context window. 3 is a solid default to start with.
"""