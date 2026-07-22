# main.py — Simple entry point
from .agent import run_query
import config

def main():
    print("🚀 Simple Hybrid RAG Ready")
    print(f"📁 Data: {config.DATA_PATH}")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("❓ You: ").strip()
        if query.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        if not query:
            continue
        
        try:
            answer = run_query(query)
            print(f"\n🤖 Answer:\n{answer}\n{'─'*50}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()