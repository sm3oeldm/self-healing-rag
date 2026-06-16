# Dependencies & Environment Setup

## requirements.txt (exact contents)

```
langchain==0.3.25
langchain-community==0.3.24
langchain-google-genai==2.1.4
langchain-text-splitters==0.3.8
langgraph==0.4.8
chromadb==1.0.12
sentence-transformers
tiktoken
python-dotenv
jupyter
```

## Installation Steps (in order)

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
```

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

3. If any conflicts appear about langchain-classic or langchain-openai, uninstall them:
```bash
pip uninstall langchain-classic langchain-openai -y
```

## .env.example (exact contents)

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Python Version

Python 3.11 (tested and confirmed working on Windows with Git Bash)

## Important Notes for the Builder

- Do NOT install google-generativeai or google-genai — they conflict with langchain-google-genai
- Embeddings use SentenceTransformers locally (all-MiniLM-L6-v2) — no API key needed for embeddings
- Gemini API key is only needed for LLM calls (generation and critique nodes)
- The first run will auto-download the SentenceTransformer model (~80MB) and cache it
- ChromaDB data is stored at data/chroma_db/ and persists between runs
