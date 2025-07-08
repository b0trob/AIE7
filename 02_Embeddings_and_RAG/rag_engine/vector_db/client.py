import os
import numpy as np
from typing import List, Dict, Any, Tuple
import openai

class EmbeddingModel:    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Failed to get embedding: {e}")

class VectorDatabase:    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.vectors = {}
        self.metadata = {}
    
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)
    
    def insert(self, text: str, vector: List[float], metadata: Dict[str, Any] = None):
        self.vectors[text] = vector
        self.metadata[text] = metadata or {}
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        query_vector = self.embedding_model.get_embedding(query)
        
        scores = []
        for text, vector in self.vectors.items():
            similarity = self.cosine_similarity(query_vector, vector)
            scores.append((text, similarity, self.metadata[text]))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]