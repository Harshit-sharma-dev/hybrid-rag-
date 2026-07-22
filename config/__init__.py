# config/__init__.py — Export all config variables
from .settings import (
    BASE_DIR,
    DATA_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    EMBEDDING_MODEL,
    OLLAMA_MODEL
)

__all__ = [
    "BASE_DIR",
    "DATA_PATH",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "EMBEDDING_MODEL",
    "OLLAMA_MODEL"
]
