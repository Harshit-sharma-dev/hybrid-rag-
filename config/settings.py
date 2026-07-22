# config.py — Sales RAG Configuration
# Local BM25 + Qwen 7B (CPU)
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────
# DATA PATHS
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data"

# Sales data files
SALES_CSV = DATA_PATH / "Financials.csv"
SALES_JSONL = DATA_PATH / "financial_data.jsonl"

# ──────────────────────────────────────────────────────────
# LOCAL MODEL - Qwen 7B on CPU
# ──────────────────────────────────────────────────────────
# No API keys needed - everything runs locally!
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"  # Free, open-source, CPU-optimized

# ──────────────────────────────────────────────────────────
# RAG RETRIEVAL - BM25 (CPU efficient, no embeddings)
# ──────────────────────────────────────────────────────────
RETRIEVAL_METHOD = "bm25"  # Sparse retrieval (super fast on CPU)
BM25_TOP_K = 5  # Number of results to retrieve
BM25_MIN_SCORE = 0.1  # Minimum relevance threshold

# ──────────────────────────────────────────────────────────
# LLM GENERATION SETTINGS
# ──────────────────────────────────────────────────────────
GENERATION_TEMPERATURE = 0.3  # Lower = more factual, deterministic
GENERATION_MAX_TOKENS = 150
GENERATION_TOP_P = 0.9

# ──────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
