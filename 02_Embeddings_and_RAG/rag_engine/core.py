import os
import numpy as np
from typing import Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from rag_engine.processors.txt_files import DigestTxtFile
from rag_engine.processors.pdf_files import DigestPDFDocument
from rag_engine.processors.text_splitter import TextSplitter
from rag_engine.vector_db.client import EmbeddingModel, VectorDatabase
from rag_engine.open_ai_connector import ChatInterface

load_dotenv()

class RAGSystem:
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.splitter = TextSplitter(chunk_size=1024, chunk_overlap=256)
        self.embedding_model = EmbeddingModel()
        self.vector_db = VectorDatabase(self.embedding_model)
        self.chat_model = ChatInterface()
        
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges

        if file_path.endswith(".pdf"):
            self.loader = DigestPDFDocument(file_path)
            self.metadata = self.loader.extract_metadata()
            self.pdf_text = self.loader.extract_text()
            self._load_and_process_pdf()

        elif file_path.endswith(".txt"):
            self.loader = DigestTxtFile(file_path)
            self.documents = self.loader.load_documents()
            self.metadata = self.loader.extract_metadata()
            self._load_and_process_txt()

        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
    def _load_and_process_txt(self):
        print(f"Loading TXT: {self.file_path}")
        chunks = self.splitter.split_text(self.documents)

        for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Creating embeddings", unit="chunk"):
            try:
                vector = self.embedding_model.get_embedding(chunk)
                chunk_metadata = {
                    **self.metadata,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
                self.vector_db.insert(chunk, vector, chunk_metadata)
            except Exception as e:
                print(f"\nError processing chunk {i}: {e}")

        print(f"Successfully processed {len(self.vector_db.vectors)} chunks")

    def _load_and_process_pdf(self):
        print(f"Loading PDF: {self.file_path}")
        
        print("Extracting text...")
        text = self.pdf_text
        
        print(f"Extracted {len(text)} characters from {self.metadata['total_pages']} pages")
        
        print("Splitting text into chunks...")
        chunks = self.splitter.split_text(text)
        print(f"Created {len(chunks)} text chunks")
        
        for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Creating embeddings", unit="chunk"):
            try:
                vector = self.embedding_model.get_embedding(chunk)
                chunk_metadata = {
                    **self.metadata,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
                self.vector_db.insert(chunk, vector, chunk_metadata)
            except Exception as e:
                print(f"\nError processing chunk {i}: {e}")
        
        print(f"Successfully processed {len(self.vector_db.vectors)} chunks")
    
    def _build_conversation_context(self, current_question: str) -> str:
        """Build conversation context from recent history."""
        if not self.conversation_history:
            return current_question
        
        # Include recent conversation history
        context_parts = []
        for i, (question, answer) in enumerate(self.conversation_history[-3:], 1):  # Last 3 exchanges
            context_parts.append(f"Previous Q{i}: {question}")
            context_parts.append(f"Previous A{i}: {answer[:200]}...")  # Truncate long answers
        
        context_parts.append(f"Current question: {current_question}")
        return "\n".join(context_parts)
    
    def _update_conversation_history(self, question: str, answer: str):
        """Update conversation history."""
        self.conversation_history.append((question, answer))
        
        # Keep only the most recent exchanges
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared!")
    
    def ask_question(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Ask a question about the PDF content with conversation context."""
        print(f"Searching for relevant context...")
        
        # Build conversation context for better search
        conversation_context = self._build_conversation_context(question)
        
        # Search for relevant chunks using both current question and conversation context
        search_results = self.vector_db.search(conversation_context, k=k)
        
        # Prepare context
        context_parts = []
        for i, (text, score, metadata) in enumerate(search_results, 1):
            context_parts.append(f"[Source {i} - Relevance: {score:.3f}]:\n{text[:500]}...")
        
        context = "\n\n".join(context_parts)
        
        # Create conversation-aware prompt
        system_message = {
            "role": "system",
            "content": """You are a helpful assistant that answers questions based on the provided context from a Penetration Testing Report. 
            You can handle follow-up questions by considering the conversation history provided.
            Only answer using information from the provided context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
            Be accurate and cite specific parts of the context when possible.
            For follow-up questions, refer to previous answers when relevant."""
        }
        
        # Build conversation history for the prompt
        conversation_text = ""
        if self.conversation_history:
            conversation_text = "\n\nPrevious conversation:\n"
            for i, (prev_q, prev_a) in enumerate(self.conversation_history[-3:], 1):
                conversation_text += f"Q{i}: {prev_q}\nA{i}: {prev_a[:300]}...\n\n"
        
        user_message = {
            "role": "user", 
            "content": f"""{conversation_text}Context from Penetration Testing Report:
        {context}

        Current Question: {question}

        Please provide your answer based on the context above and any relevant previous conversation."""
        }
        
        # Generate response
        print("Generating response...")
        response = self.chat_model.generate_response([system_message, user_message])
        
        # Update conversation history
        self._update_conversation_history(question, response)
        
        return {
            "question": question,
            "response": response,
            "context_sources": len(search_results),
            "relevance_scores": [score for _, score, _ in search_results],
            "context_preview": context[:500] + "..." if len(context) > 500 else context,
            "conversation_length": len(self.conversation_history)
        }
