import os
import numpy as np
from typing import List, Dict, Any, Tuple
import PyPDF2
import openai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class SimplePDFLoader:
    """Simple PDF loader that avoids the import issues."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text(self) -> str:
        """Extract text from PDF using PyPDF2."""
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
        """Extract basic metadata from PDF."""
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

class SimpleTextSplitter:
    """Simple text splitter for chunking documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
        return chunks

class SimpleEmbeddingModel:
    """Simple embedding model using OpenAI."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai.api_key = self.api_key
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            return response['data'][0]['embedding']
        except Exception as e:
            raise ValueError(f"Failed to get embedding: {e}")

class SimpleVectorDatabase:
    """Simple vector database for storing and searching embeddings."""
    
    def __init__(self, embedding_model: SimpleEmbeddingModel):
        self.embedding_model = embedding_model
        self.vectors = {}
        self.metadata = {}
    
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)
    
    def insert(self, text: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Insert a text chunk with its vector and metadata."""
        self.vectors[text] = vector
        self.metadata[text] = metadata or {}
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar texts."""
        query_vector = self.embedding_model.get_embedding(query)
        
        scores = []
        for text, vector in self.vectors.items():
            similarity = self.cosine_similarity(query_vector, vector)
            scores.append((text, similarity, self.metadata[text]))
        
        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class SimpleChatModel:
    """Simple chat model using OpenAI."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai.api_key = self.api_key
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI chat completion."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

class PDFRAGSystem:
    """Complete PDF RAG system for the RFC 1918 document."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.loader = SimplePDFLoader(pdf_path)
        self.splitter = SimpleTextSplitter(chunk_size=800, chunk_overlap=100)
        self.embedding_model = SimpleEmbeddingModel()
        self.vector_db = SimpleVectorDatabase(self.embedding_model)
        self.chat_model = SimpleChatModel()
        
        # Load and process the PDF
        self._load_and_process_pdf()
    
    def _load_and_process_pdf(self):
        """Load PDF and process it into the vector database."""
        print(f"Loading PDF: {self.pdf_path}")
        
        # Extract text and metadata
        text = self.loader.extract_text()
        metadata = self.loader.extract_metadata()
        
        print(f"Extracted {len(text)} characters from {metadata['total_pages']} pages")
        
        # Split text into chunks
        chunks = self.splitter.split_text(text)
        print(f"Created {len(chunks)} text chunks")
        
        # Create embeddings and store in vector database
        print("Creating embeddings...")
        for i, chunk in enumerate(chunks):
            try:
                vector = self.embedding_model.get_embedding(chunk)
                chunk_metadata = {
                    **metadata,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
                self.vector_db.insert(chunk, vector, chunk_metadata)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
        
        print(f"Successfully processed {len(self.vector_db.vectors)} chunks")
    
    def ask_question(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Ask a question about the PDF content."""
        print(f"Searching for relevant context...")
        
        # Search for relevant chunks
        search_results = self.vector_db.search(question, k=k)
        
        # Prepare context
        context_parts = []
        for i, (text, score, metadata) in enumerate(search_results, 1):
            context_parts.append(f"[Source {i} - Relevance: {score:.3f}]:\n{text[:500]}...")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        system_message = {
            "role": "system",
            "content": """You are a helpful assistant that answers questions based on the provided context from RFC 1918 (Address Allocation for Private Internets). 
            Only answer using information from the provided context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
            Be accurate and cite specific parts of the context when possible."""
        }
        
        user_message = {
            "role": "user", 
            "content": f"""Context from RFC 1918:
{context}

Question: {question}

Please provide your answer based solely on the context above."""
        }
        
        # Generate response
        print("Generating response...")
        response = self.chat_model.generate_response([system_message, user_message])
        
        return {
            "question": question,
            "response": response,
            "context_sources": len(search_results),
            "relevance_scores": [score for _, score, _ in search_results],
            "context_preview": context[:500] + "..." if len(context) > 500 else context
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        return {
            "total_chunks": len(self.vector_db.vectors),
            "pdf_metadata": self.loader.extract_metadata(),
            "chunk_sizes": [len(chunk) for chunk in self.vector_db.vectors.keys()]
        }

def test_pdf_rag():
    """Test the PDF RAG system with RFC 1918."""
    
    # Initialize the RAG system
    pdf_path = "data/RFC 1918_ Address Allocation for Private Internets.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print("Initializing PDF RAG system...")
    rag_system = PDFRAGSystem(pdf_path)
    
    # Get database stats
    stats = rag_system.get_database_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"PDF pages: {stats['pdf_metadata']['total_pages']}")
    print(f"Average chunk size: {np.mean(stats['chunk_sizes']):.0f} characters")
    
    # Test questions
    test_questions = [
        "What are the private IP address ranges defined in RFC 1918?",
        "What is the purpose of private address allocation?",
        "How many private address blocks are defined?",
        "What are the specific address ranges for each private block?",
        "What is the difference between private and public IP addresses?"
    ]
    
    print(f"\n{'='*60}")
    print("TESTING PDF RAG SYSTEM")
    print(f"{'='*60}")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 50)
        
        try:
            result = rag_system.ask_question(question)
            print(f"Response: {result['response']}")
            print(f"Sources found: {result['context_sources']}")
            print(f"Relevance scores: {[f'{s:.3f}' for s in result['relevance_scores']]}")
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    test_pdf_rag()
