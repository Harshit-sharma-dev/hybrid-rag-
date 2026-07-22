# agent.py — Simple Sales RAG Agent
# Pipeline: Route → Retrieve → Generate
import json
import logging
from .smart_router import route_query
from .tool_calling import semantic_search, analytical_query, generate_summary
import config

log = logging.getLogger(__name__)

def run_query(user_query: str) -> dict:
    """
    Simple RAG pipeline for sales data queries:
    1. Route query (search | summary | analysis)
    2. Retrieve relevant data using BM25
    3. Generate answer using Qwen 7B
    """
    
    print(f"\n📝 Query: {user_query}")
    
    # ── STEP 1: Route Query ──
    routing_result = route_query(user_query)
    intent = routing_result.get("intent", "search")
    confidence = routing_result.get("confidence", 0.5)
    
    print(f"🔀 Intent: {intent} (confidence: {confidence:.1%})")
    
    # ── STEP 2: Retrieve Context ──
    if intent == "analysis":
        # Analytical queries - use DuckDB
        retrieval_result = analytical_query(user_query)
        context_type = "analytical"
    else:
        # Search/summary - use BM25
        retrieval_result = semantic_search(user_query, k=5)
        context_type = "bm25"
    
    print(f"📊 Retrieved {len(retrieval_result)} results using {context_type}")
    
    # ── STEP 3: Generate Answer ──
    answer = generate_summary(user_query, retrieval_result)
    
    # ── Return Result ──
    result = {
        "query": user_query,
        "intent": intent,
        "confidence": confidence,
        "answer": answer,
        "context_count": len(retrieval_result),
        "retrieval_method": context_type
    }
    
    print(f"✅ Answer: {answer}\n")
    
    return result


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Show me the top sales records",
        "What was our total revenue?",
        "How many transactions do we have?"
    ]
    
    for query in test_queries:
        result = run_query(query)
        print(f"Result: {json.dumps(result, indent=2)}\n")

        
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