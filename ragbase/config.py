import os
from pathlib import Path


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

    class Model:
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        # EMBEDDINGS = "bge-large:latest"
        # EMBEDDINGS = "mxbai-embed-large:latest"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        # LOCAL_LLM = "gemma2:9b"
        LOCAL_LLM = "gemma3:12b-it-q8_0"
        # REMOTE_LLM = "llama-3.1-70b-versatile"
        TEMPERATURE = 0.0
        MAX_TOKENS = 16000
        USE_LOCAL = True

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False

    DEBUG = False
    CONVERSATION_MESSAGES_LIMIT = 100
