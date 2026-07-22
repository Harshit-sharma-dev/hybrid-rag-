# 🏢 Sales Data RAG Agent

**Local, CPU-efficient Retrieval Augmented Generation for Sales Data**

- ✅ 100% Local (no external APIs)
- ✅ CPU Optimized (BM25 retrieval, Qwen 7B on CPU)
- ✅ Fast (sub-second queries)
- ✅ Easy Setup (just Ollama + Python)

---

## 📋 What This Does

This project is a **Retrieval Augmented Generation (RAG)** system for sales data:

1. **Route** your query (search, summary, analysis)
2. **Retrieve** relevant sales records using BM25
3. **Generate** natural language answers using Qwen 7B

**Example:**
```
Query: "What was our total revenue?"
→ Routes as: summary
→ Retrieves: Top 5 revenue records
→ Generates: "Based on sales data, total revenue was $X..."
```

---

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (https://ollama.ai)
2. **Pull Qwen 7B model:**
   ```bash
   ollama pull qwen2.5:7b
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r config/requirements.txt
   ```

### Run the Agent

**Terminal 1 - Start Ollama server:**
```bash
ollama serve
```

**Terminal 2 - Run the RAG agent:**
```bash
cd /path/to/project
python -m test.main
```

**Try queries:**
```
🤔 Your query: Show me sales data
🤔 Your query: What was total revenue?
🤔 Your query: How many transactions?
```

---

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────┐
│         User Query                      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  smart_router.py                        │
│  (Qwen 7B Route Detection)              │
│  ↓ Returns: search|summary|analysis     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  tool_calling.py                        │
│  ├─ BM25 Search (CPU efficient)         │
│  ├─ DuckDB Analytics                    │
│  └─ Generate Summary (Qwen 7B)          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│         Final Answer                    │
└─────────────────────────────────────────┘
```

### Key Files

- **`smart_router.py`** - Query intent detection (Qwen 7B)
- **`tool_calling.py`** - Retrieval (BM25) + Generation (Qwen 7B)
- **`agent.py`** - RAG orchestration
- **`main.py`** - CLI interface
- **`config/settings.py`** - Configuration

---

## 📊 Retrieval Methods

### BM25 (What We Use)

✅ **Fast** - No embeddings, instant search  
✅ **CPU** - Runs on any laptop  
✅ **Interpretable** - Keyword-based matching  
✅ **Small Data** - Perfect for CSVs/JSONLs  

**How it works:**
```python
# 1. Load data
df = pd.read_csv("sales.csv")

# 2. Tokenize documents
docs = [row1, row2, row3, ...]
tokenized = [doc.split() for doc in docs]

# 3. Create BM25 index
bm25 = BM25Okapi(tokenized)

# 4. Search
scores = bm25.get_scores(query.split())
top_results = sorted by scores
```

---

## 🧠 Model: Qwen 7B

Why Qwen 7B?

- 🎯 **7B parameters** - Fast inference on CPU
- 📚 **Strong reasoning** - Good at analysis
- 🏠 **Local** - No API calls
- 💰 **Free** - Open source

```bash
# Model info
Model: qwen2.5:7b
Size: ~4.4GB
Speed: ~3-5 tokens/second (CPU)
Context: 4K tokens
```

---

## 🔧 Configuration

Edit `config/settings.py`:

```python
# Retrieval
BM25_K = 5  # Number of results
BM25_MIN_SCORE = 0.1  # Relevance threshold

# Generation
GENERATION_TEMPERATURE = 0.3  # Lower = more factual
GENERATION_MAX_TOKENS = 150

# Data
SALES_CSV = DATA_PATH / "Financials.csv"
SALES_JSONL = DATA_PATH / "financial_data.jsonl"
```

---

## 📈 Performance

Typical query times (CPU):

- **Routing** (Qwen): 500-800ms
- **Retrieval** (BM25): 10-50ms
- **Generation** (Qwen): 1-3 seconds
- **Total**: 2-5 seconds

---

## 🔄 Example Flow

```
Input: "Show me Q3 sales"

Step 1: Route
  Qwen routes → "search" (confidence: 0.92)

Step 2: Retrieve
  BM25 searches for "Q3 sales"
  Returns: [record1, record2, record3, ...]

Step 3: Generate
  Qwen generates answer from context
  Output: "Based on sales data, Q3 had..."

Result: {"answer": "...", "intent": "search", ...}
```

---

## 📁 Data Format

Your sales data should be in `/data`:

**CSV Format:**
```
date,sales,region,product,revenue
2024-01-01,100,US,Widget A,5000
2024-01-02,150,EU,Widget B,7500
...
```

**JSONL Format:**
```json
{"date": "2024-01-01", "sales": 100, "revenue": 5000}
{"date": "2024-01-02", "sales": 150, "revenue": 7500}
```

---

## 🐛 Troubleshooting

### "Connection refused" error
```
Solution: Make sure Ollama is running
$ ollama serve
```

### "Model not found"
```
Solution: Pull the model
$ ollama pull qwen2.5:7b
```

### "No results from BM25"
```
Solution: Check your query and data format
- Verify data is in /data folder
- Check column names
```

---

## 🚀 Next Steps

1. **Add more sales data** to `/data`
2. **Customize routing** in `smart_router.py`
3. **Add analytics queries** in `tool_calling.py`
4. **Deploy** as API using FastAPI (if needed)

---

## 📝 License

Open Source - MIT License

---

## 🤝 Contributing

Got improvements? Submit a PR!

---

**Made with ❤️ for Sales Analytics**
