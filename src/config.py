import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCS_PATH = "data/docs"
CHROMA_DB_PATH = "data/chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50