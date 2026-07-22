"""
src/smart_router.py — 3-Tier Adaptive Router with API Support
Fixes routing mistakes, handles multi-intent, supports OpenAI/Cohere/Ollama
"""
import re
import json
import uuid
import asyncio
import logging
from enum import Enum
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import httpx

log = logging.getLogger(__name__)

class RouteIntent(BaseModel):
    primary: str
    secondary: Optional[List[str]] = []
    confidence: float
    reasoning: str
    trace_id: str
    api_used: str = "local"

# Intent reference examples for semantic matching
INTENT_EXAMPLES = {
    "SEMANTIC": ["explain how...", "what is the difference between...", "guide to...", "why does it work..."],
    "ANALYTICAL": ["count records where...", "show me totals by...", "filter by date...", "how many users signed up..."],
    "MATH": ["calculate...", "formula for...", "percent of...", "ROI if...", "solve 2x + 5 = ..."]
}

class SmartRouter:
    def __init__(
        self,
        api_provider: str = "ollama",  # "ollama" | "openai" | "cohere" | "custom"
        api_key: Optional[str] = None,
        model: str = "qwen2.5:7b",
        cohere_api_url: str = "https://api.cohere.ai/v1/classify",
        fallback_confidence: float = 0.6
    ):
        self.api_provider = api_provider
        self.api_key = api_key
        self.model = model
        self.cohere_url = cohere_api_url
        self.fallback_conf = fallback_confidence
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def route(self, query: str, trace_id: str) -> RouteIntent:
        # TIER 1: Fast semantic + rule scoring
        primary_fast, conf_fast = self._fast_classify(query)
        
        if conf_fast >= 0.85:
            return RouteIntent(
                primary=primary_fast, confidence=conf_fast,
                reasoning="tier1_fast_match", trace_id=trace_id, api_used="local"
            )
        
        # TIER 2: LLM/API classification with structured JSON output
        primary_api, conf_api, reasoning = await self._api_classify(query)
        weighted_conf = (conf_fast * 0.3) + (conf_api * 0.7)
        
        # TIER 3: Multi-intent detection + fallback
        if weighted_conf < self.fallback_conf:
            primary_api = "HYBRID"
            reasoning += "; auto_fallback_hybrid"
            
        secondary = self._detect_secondary(query, primary_api)
        
        return RouteIntent(
            primary=primary_api,
            secondary=secondary,
            confidence=weighted_conf,
            reasoning=reasoning,
            trace_id=trace_id,
            api_used=self.api_provider
        )

    def _fast_classify(self, query: str) -> Tuple[str, float]:
        q_lower = query.lower()
        scores = {}
        for intent, examples in INTENT_EXAMPLES.items():
            # Rule score
            rule_score = 0.0
            if any(re.search(pat, q_lower) for pat in self._get_rules(intent)):
                rule_score = 0.8
            # Semantic score
            q_vec = self.encoder.encode(query)
            ex_vecs = self.encoder.encode(examples)
            sem_score = float(max(util.cos_sim(q_vec, ex_vecs)[0]))
            scores[intent] = max(rule_score, sem_score)
        return max(scores, key=scores.get), scores[max(scores, key=scores.get)]

    async def _api_classify(self, query: str) -> Tuple[str, float, str]:
        prompt = f"""Classify query intent. Return ONLY valid JSON:
{{"primary": "SEMANTIC|ANALYTICAL|MATH", "confidence": 0.0-1.0, "reasoning": "1-sentence"}}
Query: {query}"""
        
        try:
            if self.api_provider == "ollama":
                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.post("http://localhost:11434/api/chat", json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    })
                    content = res.json()["message"]["content"]
                    
            elif self.api_provider == "openai":
                async with httpx.AsyncClient(timeout=10.0, headers={"Authorization": f"Bearer {self.api_key}"}) as client:
                    res = await client.post("https://api.openai.com/v1/chat/completions", json={
                        "model": self.model or "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "max_tokens": 50
                    })
                    content = res.json()["choices"][0]["message"]["content"]
                    
            elif self.api_provider == "cohere":
                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.post(self.cohere_url, headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }, json={"texts": [query], "examples": [
                        {"text": "Explain transformers", "label": "SEMANTIC"},
                        {"text": "Count active users", "label": "ANALYTICAL"},
                        {"text": "Calculate 15% of 200", "label": "MATH"}
                    ]})
                    # Cohere returns classification array
                    best = max(res.json()["classifications"], key=lambda x: x["confidence"])
                    return best["label"].upper(), best["confidence"], "cohere_api_match"
            
            # Parse JSON output
            data = json.loads(content)
            return data["primary"].upper(), data.get("confidence", 0.5), data.get("reasoning", "")
            
        except Exception as e:
            log.warning(f"API router failed ({self.api_provider}): {e}")
            return "SEMANTIC", 0.4, f"api_fallback: {str(e)}"

    def _detect_secondary(self, query: str, primary: str) -> List[str]:
        others = [i for i in ["SEMANTIC", "ANALYTICAL", "MATH"] if i != primary]
        secondaries = []
        for intent in others:
            if any(re.search(pat, query.lower()) for pat in self._get_rules(intent)):
                secondaries.append(intent)
        return secondaries

    def _get_rules(self, intent: str) -> List[str]:
        return {
            "SEMANTIC": [r'\b(explain|describe|how|why|what is|compare|guide|concept)\b'],
            "ANALYTICAL": [r'\b(count|sum|filter|show|list|how many|total|group by)\b'],
            "MATH": [r'\b(calculate|compute|formula|percent|ROI|interest|solve|[\d\+\-\*/=])\b']
        }[intent]


# ─────────────────────────────────────────────────────────────
# Module-level compatibility wrapper for agent.py
# ─────────────────────────────────────────────────────────────
_router_instance: Optional[SmartRouter] = None

def route_query(query: str) -> str:
    """Synchronous wrapper for SmartRouter.route() used by agent.py."""
    global _router_instance
    if _router_instance is None:
        _router_instance = SmartRouter()

    intent = asyncio.run(_router_instance.route(query, trace_id=str(uuid.uuid4())))
    return intent.primary.lower()
