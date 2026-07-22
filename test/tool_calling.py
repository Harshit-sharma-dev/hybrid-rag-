# tool_calling.py — BM25 Retrieval + Local Qwen 7B
import json
import pandas as pd
from rank_bm25 import BM25Okapi
import duckdb
import ollama
import config

# Load sales data once
def load_sales_data():
    """Load sales data from CSV or JSONL"""
    try:
        df = pd.read_csv(config.DATA_PATH / "Financials.csv")
        return df
    except:
        # Fallback to JSONL
        records = []
        with open(config.DATA_PATH / "financial_data.jsonl") as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)

sales_df = load_sales_data()
ollama_client = ollama.Client(host='http://localhost:11434')

# ──────────────────────────────────────────────────────────
# BM25 Retrieval (No embeddings needed - efficient on CPU)
# ──────────────────────────────────────────────────────────

def build_bm25_index():
    """Build BM25 index from sales data"""
    # Combine all text columns into a single searchable field
    docs = []
    for idx, row in sales_df.iterrows():
        doc = " ".join([str(v) for v in row.values if v])
        docs.append(doc)
    
    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, docs, sales_df

# Build index once
bm25_index, bm25_docs, bm25_df = build_bm25_index()

def bm25_search(query: str, k: int = 5) -> list:
    """
    BM25 search on sales data
    Efficient on CPU, no embeddings needed
    """
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for idx in top_k_indices:
        if scores[idx] > 0:  # Only include relevant results
            results.append({
                "text": bm25_docs[idx],
                "row_index": idx,
                "score": scores[idx]
            })
    
    return results

# ──────────────────────────────────────────────────────────
# DuckDB for Analytical Queries
# ──────────────────────────────────────────────────────────

def analytical_query(query: str) -> dict:
    """
    Run SQL aggregation on sales data using DuckDB
    """
    try:
        conn = duckdb.connect()
        conn.register("sales", sales_df)
        
        # Simple analytics (user can extend with more queries)
        if "total" in query.lower():
            result = conn.execute(
                "SELECT SUM(CAST(revenue as FLOAT64)) as total FROM sales"
            ).fetchall()
            return {"result": f"Total Revenue: ${result[0][0]:,.2f}"}
        
        elif "average" in query.lower():
            result = conn.execute(
                "SELECT AVG(CAST(revenue as FLOAT64)) as avg_revenue FROM sales"
            ).fetchall()
            return {"result": f"Average Revenue: ${result[0][0]:,.2f}"}
        
        elif "count" in query.lower():
            result = conn.execute("SELECT COUNT(*) as cnt FROM sales").fetchall()
            return {"result": f"Total Records: {result[0][0]}"}
        
        else:
            # Fallback summary
            result = conn.execute(
                "SELECT COUNT(*) as cnt, SUM(CAST(revenue as FLOAT64)) as total FROM sales"
            ).fetchall()
            return {
                "count": result[0][0],
                "total_revenue": result[0][1]
            }
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────────────────────────────────────
# Semantic Search (using BM25, not embeddings)
# ──────────────────────────────────────────────────────────

def semantic_search(query: str, k: int = 5) -> list:
    """
    Multi-purpose sales data search using BM25
    Returns top-k relevant sales records
    """
    return bm25_search(query, k=k)

# ──────────────────────────────────────────────────────────
# Qwen 7B - Summary Generation
# ──────────────────────────────────────────────────────────

def generate_summary(query: str, context: list) -> str:
    """
    Use Qwen 7B to generate natural language summary from retrieved data
    """
    # Format context
    context_text = "\n".join([f"- {item.get('text', str(item))}" for item in context])
    
    prompt = f"""You are a sales analyst. Based on this sales data, answer the query concisely:

SALES DATA:
{context_text}

QUERY: {query}

ANSWER (2-3 sentences max):"""
    
    try:
        response = ollama_client.generate(
            model="qwen2.5:7b",
            prompt=prompt,
            stream=False,
            options={"temperature": 0.3, "num_predict": 100}
        )
        return response['response'].strip()
    except Exception as e:
        return f"Error generating summary: {e}"

    
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