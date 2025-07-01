#!/usr/bin/env python3
"""
Interactive PDF RAG System for RFC 1918
Run this script to interactively ask questions about the RFC 1918 document.
"""

import os
from pdf_rag_demo import PDFRAGSystem

def main():
    pdf_path = "data/RFC 1918_ Address Allocation for Private Internets.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please make sure the RFC 1918 PDF is in the data/ directory.")
        return
    
    print("🚀 Initializing PDF RAG System for RFC 1918...")
    print("This may take a few minutes to process the PDF and create embeddings...")
    
    try:
        rag_system = PDFRAGSystem(pdf_path)
        print("✅ RAG system initialized successfully!")
        
        # Show database stats
        stats = rag_system.get_database_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   PDF pages: {stats['pdf_metadata']['total_pages']}")
        print(f"   Average chunk size: {sum(stats['chunk_sizes']) // len(stats['chunk_sizes'])} characters")
        
        print(f"\n{'='*60}")
        print("🎯 INTERACTIVE RFC 1918 Q&A SYSTEM")
        print("Ask questions about private IP address allocation!")
        print("Type 'quit' or 'exit' to stop.")
        print(f"{'='*60}")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                print("\n🔍 Searching for relevant information...")
                result = rag_system.ask_question(question, k=3)
                
                print(f"\n📝 Answer:")
                print(f"{result['response']}")
                print(f"\n📊 Sources: {result['context_sources']} chunks found")
                print(f"🎯 Relevance scores: {[f'{s:.3f}' for s in result['relevance_scores']]}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again with a different question.")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("Please check your OpenAI API key and internet connection.")

if __name__ == "__main__":
    main()
