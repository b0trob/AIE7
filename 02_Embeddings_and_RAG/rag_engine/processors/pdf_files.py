import os
import PyPDF2
from datetime import datetime
from typing import Dict, Any


class DigestPDFDocument:    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text(self) -> str:
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {e}")
    
    def extract_metadata(self) -> Dict[str, Any]:
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = {
                    "filename": os.path.basename(self.pdf_path),
                    "total_pages": len(pdf_reader.pages),
                    "file_size": os.path.getsize(self.pdf_path),
                    "extracted_at": datetime.now().isoformat()
                }
                
                # Try to get document info
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "producer": pdf_reader.metadata.get('/Producer', '')
                    })
                
                return metadata
        except Exception as e:
            return {"error": str(e)}