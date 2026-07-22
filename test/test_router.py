# test_smart_router.py
import asyncio
from .smart_router import SmartRouter

router = SmartRouter(api_provider="ollama", model="qwen2.5:7b")

async def test():
    queries = [
        "Explain compound interest",           # → SEMANTIC (conf >0.8)
        "Count records where status=active",   # → ANALYTICAL (conf >0.8)
        "Calculate 15% of 200",                # → MATH (conf >0.9)
        "Walk me through ROI math",            # → MATH + SEMANTIC (conf ~0.7)
        "What's the deal with Model X?",       # → HYBRID fallback (conf <0.6)
    ]
    for q in queries:
        intent = await router.route(q, "test")
        print(f"Query: {q}")
        print(f"→ Primary: {intent.primary} | Conf: {intent.confidence:.2f} | API: {intent.api_used}")
        print(f"→ Secondary: {intent.secondary} | Reason: {intent.reasoning}\n")

asyncio.run(test())