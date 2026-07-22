"""
smart_router.py — Local Query Router with Qwen 7B (CPU)
Simple intent detection for sales data RAG queries
Uses Qwen 7B locally for routing (no external APIs)
"""
import json
import logging
import re
from typing import Dict
import ollama

log = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        self.ollama_client = ollama.Client(host='http://localhost:11434')
        
    def route(self, query: str) -> Dict[str, any]:
        """
        Route query using Qwen 7B locally.
        Returns: {"intent": "search|summary|analysis", "confidence": 0.0-1.0}
        """
        routing_prompt = f"""You are a sales data assistant. Classify this query as one of:
- "search": Find specific sales records (e.g., "sales in Q3", "top customers")
- "summary": General statistics (e.g., "total revenue", "average order value")
- "analysis": Trend analysis (e.g., "growth rate", "customer retention")

Query: {query}

Respond ONLY with JSON: {{"intent": "search|summary|analysis", "confidence": 0.7-1.0}}"""
        
        try:
            response = self.ollama_client.generate(
                model=self.model,
                prompt=routing_prompt,
                stream=False,
                options={"temperature": 0.1, "num_predict": 50}
            )
            
            # Extract JSON from response
            content = response['response'].strip()
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                return result
            return {"intent": "search", "confidence": 0.5}
        except Exception as e:
            log.warning(f"Routing error: {e}")
            return {"intent": "search", "confidence": 0.5}


# Global router instance
_router = None

def get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router

def route_query(query: str) -> Dict[str, any]:
    """Simple routing wrapper for agent.py"""
    router = get_router()
    return router.route(query)
