# tools.py — Deterministic tools (no LLM)
import json
import duckdb
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
import config

# Initialize once
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

# ── Pinecone Tool ──
def semantic_search(query: str, k: int = 5) -> list:
    """Returns relevant text chunks from Pinecone"""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(name=config.PINECONE_INDEX, dimension=384, metric="cosine",
                       spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pc.Index(config.PINECONE_INDEX)
    
    vec = embeddings.embed_query(query)
    res = index.query(vector=vec, top_k=k, include_metadata=True)
    return [m["metadata"].get("text", "") for m in res["matches"]]

# ── DuckDB Tool ──
def analytical_query(query: str) -> dict:
    """Returns structured data results from your JSONL"""
    if not config.DATA_PATH.exists():
        return {"error": f"Data file not found: {config.DATA_PATH}"}
    
    # Load data
    df = pd.read_json(config.DATA_PATH, lines=True)
    conn = duckdb.connect()
    conn.register("data", df)
    
    # Simple SQL mapping (expand as needed)
    q_lower = query.lower()
    if "count" in q_lower or "how many" in q_lower:
        sql = "SELECT COUNT(*) as count FROM data"
    elif "sum" in q_lower or "total" in q_lower:
        # Auto-find first numeric column
        num_cols = df.select_dtypes(include='number').columns.tolist()
        col = num_cols[0] if num_cols else "1"
        sql = f"SELECT SUM({col}) as total FROM data"
    else:
        sql = "SELECT * FROM data LIMIT 10"
    
    try:
        result = conn.execute(sql).fetchdf()
        return {"data": result.to_dict(orient="records"), "sql": sql}
    except Exception as e:
        return {"error": str(e)}

# ── Calculator Tool ──
def calculate(expression: str) -> dict:
    """Safely evaluates math expressions"""
    allowed = set("0123456789.+-*/() ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters"}
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return {"result": float(result), "expression": expression}
    except Exception as e:
        return {"error": str(e)}