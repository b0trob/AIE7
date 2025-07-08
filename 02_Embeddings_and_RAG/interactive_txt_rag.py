import os
from rag_engine.core import RAGSystem

def main():
    txt_path = "data/test-scan.txt"
    
    if not os.path.exists(txt_path):
        print(f"Text file not found: {txt_path}")
        print("Please make sure the text file is in the data/ directory.")
        return
    
    try:
        rag_system = RAGSystem(txt_path)
        print("✅ RAG system initialized successfully!")


        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if question.lower() == 'clear':
                    rag_system.clear_conversation_history()
                    continue
                if not question:
                    print("Please enter a question.")
                    continue
                
                print("\n🔍 Searching for relevant information...")
                result = rag_system.ask_question(question, k=3)
                
                print(f"\n📝 Answer:")
                print(f"{result['response']}")
                print(f"\n📊 Sources: {result['context_sources']} chunks found")
                print(f"🎯 Relevance scores: {[f'{s:.3f}' for s in result['relevance_scores']]}")
                print(f"💬 Conversation length: {result['conversation_length']} exchanges")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again with a different question.")

    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")

if __name__ == "__main__":
    main()