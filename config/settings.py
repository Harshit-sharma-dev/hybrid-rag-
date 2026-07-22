# config.py — Simple settings for Hybrid RAG
import os
from pathlib import Path

# Base directory (where this config.py lives)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"  # Points to New folder/data/

# Paths
DATA_PATH = DATA_DIR / "C:\\Users\\Harshit Sharma\\OneDrive\\Desktop\\New folder\\data\\financial_data.jsonl"  # Your dataset

# 🔑 Pinecone settings (get key from https://app.pinecone.io)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-key-here")
PINECONE_INDEX = "simple-rag-index"

# Embedding model (local, no API needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings (local Ollama)
OLLAMA_MODEL = "qwen2.5:7b"  # Better tool calling than 1b models
OLLAMA_BASE_URL = "http://localhost:11434"