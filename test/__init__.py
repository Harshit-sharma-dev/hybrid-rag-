# test/__init__.py
# Clean package-level imports
from .agent import run_query
from .smart_router import route_query
from .tool_calling import semantic_search, analytical_query, calculate
from .config import PINECONE_API_KEY, OLLAMA_MODEL, EMBEDDING_MODEL

__all__ = [
    "run_query",
    "route_query", 
    "semantic_search",
    "analytical_query",
    "calculate",
    "PINECONE_API_KEY",
    "OLLAMA_MODEL",
    "EMBEDDING_MODEL"
]