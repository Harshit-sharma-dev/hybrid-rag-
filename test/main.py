#!/usr/bin/env python3
"""
main.py — Sales RAG Demo
Local BM25 + Qwen 7B (CPU)

Setup:
1. ollama serve (in terminal 1)
2. ollama pull qwen2.5:7b (pull model)
3. python test/main.py (run this)
"""

from agent import run_query
import config

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║      Sales Data RAG - Qwen 7B + BM25 (CPU)                ║
║  100% Local • No APIs • No Embeddings • Fast              ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("📚 Try these:")
    print("  • Show me sales data")
    print("  • What's the total revenue?")
    print("  • How many records?")
    print("  • type 'exit' to quit\n")
    
    while True:
        try:
            query = input("🤔 Your query: ").strip()
            
            if query.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Run RAG pipeline
            result = run_query(query)
            print(f"\n{'─'*60}")
            print(f"✅ {result['answer']}")
            print(f"   (Intent: {result['intent']}, Records: {result['context_count']})")
            print(f"{'─'*60}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
