import os
from typing import List, Dict, Any
from datetime import datetime
import hashlib
import PyPDF2
import fitz  # PyMuPDF
import io
from pathlib import Path


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_file()
            elif self.path.endswith(".pdf"):
                self.load_pdf()
            else:
                raise ValueError(f"Unsupported file type: {self.path}")
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a supported file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf(self):
        """Load text from PDF file using PyMuPDF for better text extraction."""
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(self.path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            self.documents.append(text)
        except ImportError:
            # Fallback to PyPDF2 if PyMuPDF is not available
            try:
                with open(self.path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    self.documents.append(text)
            except Exception as e:
                raise ValueError(f"Failed to read PDF file: {e}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding=self.encoding) as f:
                        self.documents.append(f.read())
                elif file.endswith(".pdf"):
                    try:
                        # Load PDF using PyMuPDF
                        doc = fitz.open(file_path)
                        text = ""
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            text += page.get_text()
                        doc.close()
                        self.documents.append(text)
                    except Exception as e:
                        print(f"Warning: Could not load PDF {file_path}: {e}")

    def load_documents(self):
        self.load()
        return self.documents


class PDFLoader:
    """Specialized loader for PDF files with metadata extraction."""
    
    def __init__(self, path: str):
        self.path = path
        self.documents = []
        self.metadata = []

    def load_pdf_with_metadata(self) -> List[Dict[str, Any]]:
        """Load PDF with extracted metadata."""
        try:
            doc = fitz.open(self.path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Extract page metadata
                page_metadata = {
                    "source_file": self.path,
                    "file_type": "pdf",
                    "page_number": page_num + 1,
                    "total_pages": len(doc),
                    "page_rect": page.rect,
                    "rotation": page.rotation,
                    "text_length": len(text),
                    "extracted_at": datetime.now().isoformat()
                }
                
                # Try to extract document-level metadata
                if page_num == 0:  # Only on first page
                    try:
                        doc_metadata = doc.metadata
                        page_metadata.update({
                            "title": doc_metadata.get("title", ""),
                            "author": doc_metadata.get("author", ""),
                            "subject": doc_metadata.get("subject", ""),
                            "creator": doc_metadata.get("creator", ""),
                            "producer": doc_metadata.get("producer", ""),
                            "creation_date": doc_metadata.get("creationDate", ""),
                            "modification_date": doc_metadata.get("modDate", "")
                        })
                    except:
                        pass
                
                self.documents.append(text)
                self.metadata.append(page_metadata)
            
            doc.close()
            return [{"text": text, "metadata": meta} for text, meta in zip(self.documents, self.metadata)]
            
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {e}")

    def extract_images(self, output_dir: str = None) -> List[Dict[str, Any]]:
        """Extract images from PDF pages."""
        if output_dir is None:
            output_dir = "extracted_images"
        
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        try:
            doc = fitz.open(self.path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
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
                    
                    pix = None  # Free memory
            
            doc.close()
            return images
            
        except Exception as e:
            print(f"Warning: Could not extract images: {e}")
            return []


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

    def split_with_metadata(self, texts: List[str], source_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split texts and generate metadata for each chunk.
        
        Args:
            texts: List of texts to split
            source_info: Additional source information to include in metadata
            
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        chunks_with_metadata = []
        chunk_id = 0
        
        for doc_idx, text in enumerate(texts):
            chunks = self.split(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "chunk_id": chunk_id,
                    "document_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks),
                    "created_at": datetime.now().isoformat(),
                    "text_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                }
                
                # Add source information if provided
                if source_info:
                    metadata.update(source_info)
                
                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": metadata
                })
                
                chunk_id += 1
        
        return chunks_with_metadata

    def split_pdf_pages_with_metadata(self, pdf_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split PDF pages and preserve page-level metadata.
        
        Args:
            pdf_data: List of dictionaries with 'text' and 'metadata' from PDFLoader
            
        Returns:
            List of dictionaries with 'text' and 'metadata' for each chunk
        """
        chunks_with_metadata = []
        chunk_id = 0
        
        for page_data in pdf_data:
            text = page_data["text"]
            page_metadata = page_data["metadata"]
            
            chunks = self.split(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "chunk_id": chunk_id,
                    "page_number": page_metadata["page_number"],
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks),
                    "created_at": datetime.now().isoformat(),
                    "text_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                }
                
                # Preserve PDF-specific metadata
                metadata.update({
                    "source_file": page_metadata["source_file"],
                    "file_type": page_metadata["file_type"],
                    "total_pages": page_metadata["total_pages"],
                    "title": page_metadata.get("title", ""),
                    "author": page_metadata.get("author", ""),
                    "subject": page_metadata.get("subject", "")
                })
                
                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": metadata
                })
                
                chunk_id += 1
        
        return chunks_with_metadata


class MultiFormatLoader:
    """Loader that supports multiple file formats including PDF."""
    
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding
        self.documents = []
        self.metadata = []
    
    def load(self):
        """Load documents from various file formats."""
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            self.load_single_file()
        else:
            raise ValueError(f"Path does not exist: {self.path}")
    
    def load_single_file(self):
        """Load a single file based on its extension."""
        file_ext = Path(self.path).suffix.lower()
        
        if file_ext == ".txt":
            self.load_text_file()
        elif file_ext == ".pdf":
            self.load_pdf_file()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def load_text_file(self):
        """Load text file."""
        with open(self.path, "r", encoding=self.encoding) as f:
            text = f.read()
            self.documents.append(text)
            self.metadata.append({
                "source_file": self.path,
                "file_type": "text",
                "encoding": self.encoding,
                "text_length": len(text)
            })
    
    def load_pdf_file(self):
        """Load PDF file using PDFDocumentLoader."""
        try:
            from .pdf_utils import PDFDocumentLoader
            loader = PDFDocumentLoader(self.path)
            pdf_data = loader.load_with_metadata()
            
            # Extract text from all pages
            all_text = ""
            for page in pdf_data["pages"]:
                all_text += page["text"] + "\n"
            
            self.documents.append(all_text)
            self.metadata.append({
                "source_file": self.path,
                "file_type": "pdf",
                "document_metadata": pdf_data["document_metadata"],
                "total_pages": pdf_data["total_pages"],
                "text_length": len(all_text)
            })
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install PyMuPDF")
    
    def load_directory(self):
        """Load all supported files from directory."""
        for file_path in Path(self.path).rglob("*"):
            if file_path.is_file():
                try:
                    file_ext = file_path.suffix.lower()
                    if file_ext in [".txt", ".pdf"]:
                        # Create a temporary loader for this file
                        temp_loader = MultiFormatLoader(str(file_path), self.encoding)
                        temp_loader.load_single_file()
                        self.documents.extend(temp_loader.documents)
                        self.metadata.extend(temp_loader.metadata)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
    
    def load_documents(self):
        """Load and return documents."""
        self.load()
        return self.documents
    
    def load_documents_with_metadata(self):
        """Load and return documents with metadata."""
        self.load()
        return [{"text": doc, "metadata": meta} for doc, meta in zip(self.documents, self.metadata)]


if __name__ == "__main__":
    # Test with text file
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(f"Text file chunks: {len(chunks)}")
    
    # Test with PDF (if available)
    try:
        pdf_loader = PDFLoader("data/sample.pdf")
        pdf_data = pdf_loader.load_pdf_with_metadata()
        pdf_chunks = splitter.split_pdf_pages_with_metadata(pdf_data)
        print(f"PDF chunks: {len(pdf_chunks)}")
        print("Sample PDF chunk metadata:", pdf_chunks[0]["metadata"])
    except FileNotFoundError:
        print("No sample.pdf found for testing")
