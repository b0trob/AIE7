#!/usr/bin/env python3
"""
Test script to demonstrate follow-up question functionality
"""

import os
from pdf_rag_demo import PDFRAGSystem

def test_followup_questions():
    """Test the follow-up question functionality."""
    
    pdf_path = "data/Anonymised-Web-and-Infrastructure-Penetration-Testing-Report_2019.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print("🚀 Initializing PDF RAG System for Follow-up Question Testing...")
    
    try:
        rag_system = PDFRAGSystem(pdf_path)
        print("✅ RAG system initialized successfully!")
        
        # Test questions that build on each other
        test_questions = [
            "What types of vulnerabilities were found in this penetration test?",
            "How many of those were critical?",
            "What were the most common critical vulnerabilities?",
            "Were any of these vulnerabilities related to web applications?",
            "What recommendations were given for the critical vulnerabilities?"
        ]
        
        print(f"\n{'='*80}")
        print("🧪 TESTING FOLLOW-UP QUESTION FUNCTIONALITY")
        print("These questions build on each other to test conversation context:")
        print(f"{'='*80}")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'🔍'*20} Question {i} {'🔍'*20}")
            print(f"❓ Question: {question}")
            print(f"💬 Conversation history length: {len(rag_system.conversation_history)}")
            
            try:
                result = rag_system.ask_question(question, k=3)
                
                print(f"\n📝 Answer:")
                print(f"{result['response']}")
                print(f"\n📊 Sources: {result['context_sources']} chunks found")
                print(f"🎯 Relevance scores: {[f'{s:.3f}' for s in result['relevance_scores']]}")
                print(f"💬 Conversation length: {result['conversation_length']} exchanges")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print(f"\n{'─'*60}")
        
        print(f"\n✅ Follow-up question testing completed!")
        print(f"Final conversation history length: {len(rag_system.conversation_history)}")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")

if __name__ == "__main__":
    test_followup_questions() 