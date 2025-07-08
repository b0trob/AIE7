#!/usr/bin/env python3
# this script does not use OpenAI API, it is just a basic test of the PDFLoader and TextSplitter classes.

import os
from rag_processors.pdf_files import PDFLoader, TextSplitter

def test_pdf_processing():
    pdf_path = "data/Anonymised-Web-and-Infrastructure-Penetration-Testing-Report_2019.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return False
    
    print("Testing PDF processing...")
    
    try:
        loader = PDFLoader(pdf_path)
        text = loader.extract_text()
        metadata = loader.extract_metadata()
        
        print(f"PDF loaded successfully")
        print(f"Pages: {metadata['total_pages']}")
        print(f"Text length: {len(text)} characters")
        print(f"File size: {metadata['file_size']} bytes")
        
        splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        print(f"Text split successfully")
        print(f"Chunks created: {len(chunks)}")
        print(f"Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
        
        print(f"\nSample chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   Chunk {i}: {chunk[:500].replace('\n', ' ')}...")
        
        print(f"\n🔍 Testing basic text search...")
        search_terms = ["cross", "high", "medium", "low", "payload", "vulnerabil"]
        for term in search_terms:
            count = text.lower().count(term.lower())
            print(f"   '{term}' appears {count} times")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pdf_processing()
    if success:
        print(f"\nAll basic tests passed!")
        print(f"To test the full RAG system, run: python3 interactive_pdf_rag.py")
    else:
        print(f"\nTests failed. Please check the PDF file and dependencies.")
