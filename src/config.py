import os
import sys
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY is not set. Create a .env file with:")
    print("  GOOGLE_API_KEY=your_gemini_api_key_here")
    print("Get a free key at: https://aistudio.google.com")
    sys.exit(1)

DOCS_PATH = "data/docs"
CHROMA_DB_PATH = "data/chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Override in .env with LLM_MODEL=<model_name> to swap without touching code.
# Available: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, etc.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 1