#!/usr/bin/env python3
"""
Basic test of PDF processing without API calls
"""

import os
from pdf_rag_demo import SimplePDFLoader, SimpleTextSplitter

def test_pdf_processing():
    pdf_path = "data/RFC 1918_ Address Allocation for Private Internets.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print("🔍 Testing PDF processing...")
    
    # Test PDF loading
    try:
        loader = SimplePDFLoader(pdf_path)
        text = loader.extract_text()
        metadata = loader.extract_metadata()
        
        print(f"✅ PDF loaded successfully")
        print(f"   Pages: {metadata['total_pages']}")
        print(f"   Text length: {len(text)} characters")
        print(f"   File size: {metadata['file_size']} bytes")
        
        # Test text splitting
        splitter = SimpleTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        print(f"✅ Text split successfully")
        print(f"   Chunks created: {len(chunks)}")
        print(f"   Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
        
        # Show sample chunks
        print(f"\n📄 Sample chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   Chunk {i}: {chunk[:100]}...")
        
        # Test text search (basic)
        print(f"\n🔍 Testing basic text search...")
        search_terms = ["private", "address", "192.168", "10.0", "172.16"]
        for term in search_terms:
            count = text.lower().count(term.lower())
            print(f"   '{term}' appears {count} times")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pdf_processing()
    if success:
        print(f"\n✅ All basic tests passed! The PDF is ready for RAG processing.")
        print(f"💡 To test the full RAG system, run: python3 interactive_pdf_rag.py")
    else:
        print(f"\n❌ Tests failed. Please check the PDF file and dependencies.")
