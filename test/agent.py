# agent.py — Orchestrates routing, tool execution, and LLM synthesis
import json
import re
import ollama
from .smart_router import route_query
from .tool_calling import semantic_search, analytical_query, calculate
import config

def run_query(user_query: str) -> str:
    """
    Full data flow:
    1. Route query (code-only, <5ms)
    2. Execute appropriate tool(s) (deterministic)
    3. Format context
    4. Call LLM ONCE for final answer
    """
    
    # ── STEP 1: Fast Routing ──
    route = route_query(user_query)
    print(f"🔀 Route: {route}")
    
    # ── STEP 2: Run Tools Based on Route ──
    context = {}
    
    if route == "math":
        # Extract math expression from query
        match = re.search(r'[\d\.\s\+\-\*/\(\)]+', user_query)
        if match:
            expr = match.group().strip()
            # Remove common words that regex might catch
            expr = re.sub(r'\b(of|plus|minus|times|divided by|percent)\b', '', expr).strip()
            if any(c in expr for c in '+-*/'):
                context["calculation"] = calculate(expr)
            else:
                context["calculation"] = {"error": "No valid math expression detected"}
        else:
            context["calculation"] = {"error": "No numbers found"}
            
    if route in ("semantic", "analytical", "hybrid"):
        context["search"] = semantic_search(user_query, k=5)
        
    if route == "analytical":
        context["data"] = analytical_query(user_query)
    
    # ── STEP 3: Format Context for LLM ──
    context_str = json.dumps(context, indent=2, default=str)
    
    # 🔍 DEBUG: Verify exact path & data flow
    print("\n🔍 DEBUG - Context sent to LLM:")
    print(context_str[:300] + "..." if len(context_str) > 300 else context_str)
    print(f"📊 Tools triggered: {list(context.keys())}")
    
    # ── STEP 4: LLM Synthesis (ONE CALL ONLY) ──
    system_prompt = f"""You are a precise AI assistant. Query type: {route.upper()}.
    Use ONLY the provided context. If context is missing or has errors, say so clearly.
    Be concise, accurate, and never hallucinate numbers or facts."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
    ]
    
    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 500}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"