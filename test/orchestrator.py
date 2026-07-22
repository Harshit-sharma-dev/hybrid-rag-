# agent.py — Runs tools + calls LLM ONCE for final answer
import json
import ollama
from .smart_router import route_query
from .tool_calling import semantic_search, analytical_query, calculate
import config

def run_query(user_query: str) -> str:
    # Step 1: Route query (fast, code-only)
    route = route_query(user_query)
    print(f"🔀 Route: {route}")
    
    # Step 2: Run appropriate tool(s)
    context = {}
    
    if route == "math":
        # Extract math expression from query
        import re
        expr = re.search(r'[\d\.\s\+\-\*/\(\)]+', user_query)
        if expr:
            context["calculation"] = calculate(expr.group().strip())
    
    if route in ("semantic", "analytical"):
        context["chunks"] = semantic_search(user_query, k=5)
    
    if route == "analytical":
        context["data"] = analytical_query(user_query)
    
    # Step 3: LLM synthesizes answer (ONE call only)
    system_prompt = f"""You are a helpful assistant. Route: {route}.
    Use ONLY the provided context. If context is missing, say so clearly.
    Be concise and accurate."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {json.dumps(context, indent=2)}\n\nQuestion: {user_query}"}
    ]
    
    response = ollama.chat(model=config.OLLAMA_MODEL, messages=messages, options={"temperature": 0.1})
    return response["message"]["content"]