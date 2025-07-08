import os
import fitz  # PyMuPDF
import PyPDF2
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class PDFProcessor:
    """Comprehensive PDF processing utility for RAG applications."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = None
        self.metadata = {}
        
    def __enter__(self):
        self.doc = fitz.open(self.pdf_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.doc:
            self.doc.close()
    
    def extract_text_by_pages(self) -> List[Dict[str, Any]]:
        """Extract text from each page with metadata."""
        pages_data = []
        
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            text = page.get_text()
            
            page_data = {
                "page_number": page_num + 1,
                "text": text,
                "text_length": len(text),
                "page_rect": {
                    "width": page.rect.width,
                    "height": page.rect.height
                },
                "rotation": page.rotation
            }
            pages_data.append(page_data)
        
        return pages_data
    
    def extract_document_metadata(self) -> Dict[str, Any]:
        """Extract document-level metadata."""
        try:
            doc_metadata = self.doc.metadata
            return {
                "title": doc_metadata.get("title", ""),
                "author": doc_metadata.get("author", ""),
                "subject": doc_metadata.get("subject", ""),
                "creator": doc_metadata.get("creator", ""),
                "producer": doc_metadata.get("producer", ""),
                "creation_date": doc_metadata.get("creationDate", ""),
                "modification_date": doc_metadata.get("modDate", ""),
                "total_pages": len(self.doc),
                "file_size": os.path.getsize(self.pdf_path),
                "file_path": self.pdf_path
            }
        except Exception as e:
            return {"error": str(e)}
    
    def extract_images(self, output_dir: str = "extracted_images") -> List[Dict[str, Any]]:
        """Extract images from all pages."""
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(self.doc, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    pix.save(img_path)
                    
                    images.append({
                        "page": page_num + 1,
                        "image_index": img_index + 1,
                        "filename": img_filename,
                        "path": img_path,
                        "width": pix.width,
                        "height": pix.height,
                        "colorspace": pix.colorspace.name
                    })
                
                pix = None
        
        return images
    
    def extract_tables(self) -> List[Dict[str, Any]]:
        """Extract tables from PDF pages."""
        tables = []
        
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            
            # Try to extract tables (this is a simplified approach)
            # In production, you might want to use more sophisticated table extraction
            try:
                # This is a basic table detection - you might want to enhance this
                text = page.get_text("dict")
                # Look for table-like structures in the text
                # This is a placeholder for more sophisticated table extraction
                pass
            except Exception as e:
                print(f"Could not extract tables from page {page_num + 1}: {e}")
        
        return tables


class PDFDocumentLoader:
    """High-level PDF document loader for RAG applications."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def load_with_metadata(self) -> Dict[str, Any]:
        """Load PDF with comprehensive metadata."""
        with PDFProcessor(self.pdf_path) as processor:
            pages_data = processor.extract_text_by_pages()
            doc_metadata = processor.extract_document_metadata()
            
            return {
                "document_metadata": doc_metadata,
                "pages": pages_data,
                "total_pages": len(pages_data),
                "loaded_at": datetime.now().isoformat()
            }
    
    def load_for_rag(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Load PDF optimized for RAG processing."""
        from .text_utils import CharacterTextSplitter
        
        # Load PDF data
        pdf_data = self.load_with_metadata()
        
        # Extract all text
        all_text = ""
        for page in pdf_data["pages"]:
            all_text += page["text"] + "\n"
        
        # Split into chunks
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split(all_text)
        
        # Create chunks with metadata
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source_file": self.pdf_path,
                "file_type": "pdf",
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks),
                "title": pdf_data["document_metadata"].get("title", ""),
                "author": pdf_data["document_metadata"].get("author", ""),
                "total_pages": pdf_data["total_pages"],
                "created_at": datetime.now().isoformat()
            }
            
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": metadata
            })
        
        return chunks_with_metadata


def process_pdf_directory(directory_path: str, output_file: str = None) -> List[Dict[str, Any]]:
    """Process all PDF files in a directory for RAG."""
    all_chunks = []
    
    for pdf_file in Path(directory_path).glob("**/*.pdf"):
        try:
            loader = PDFDocumentLoader(str(pdf_file))
            chunks = loader.load_for_rag()
            all_chunks.extend(chunks)
            print(f"Processed {pdf_file}: {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
    
    return all_chunks


if __name__ == "__main__":
    # Example usage
    pdf_path = "data/sample.pdf"
    
    if os.path.exists(pdf_path):
        # Load PDF for RAG
        loader = PDFDocumentLoader(pdf_path)
        chunks = loader.load_for_rag()
        
        print(f"Extracted {len(chunks)} chunks from PDF")
        print("Sample chunk metadata:", chunks[0]["metadata"])
    else:
        print(f"PDF file {pdf_path} not found") 