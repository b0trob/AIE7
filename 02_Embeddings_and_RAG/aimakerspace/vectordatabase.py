import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import json
from datetime import datetime


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the euclidean distance between two vectors."""
    return np.linalg.norm(vector_a - vector_b)


def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the manhattan distance between two vectors."""
    return np.sum(np.abs(vector_a - vector_b))


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # Store metadata for each vector
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert a vector with optional metadata."""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata
        else:
            self.metadata[key] = {}

    def insert_with_metadata(self, text: str, vector: np.array, metadata: Dict[str, Any]) -> None:
        """Insert a text chunk with its vector and metadata."""
        self.vectors[text] = vector
        self.metadata[text] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors with metadata filtering.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            distance_measure: Distance function to use
            filter_metadata: Optional metadata filters (e.g., {"source": "blog", "date": "2023"})
        """
        scores = []
        
        for key, vector in self.vectors.items():
            # Apply metadata filtering if specified
            if filter_metadata:
                if not self._matches_metadata_filter(key, filter_metadata):
                    continue
            
            score = distance_measure(query_vector, vector)
            scores.append((key, score, self.metadata[key]))
        
        # Sort by score (higher is better for similarity, lower is better for distance)
        reverse_sort = distance_measure == cosine_similarity
        sorted_results = sorted(scores, key=lambda x: x[1], reverse=reverse_sort)[:k]
        
        return sorted_results

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search by text with metadata support.
        
        Returns:
            List of tuples: (text, score, metadata)
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, filter_metadata)
        
        if return_as_text:
            return [(result[0], result[1]) for result in results]
        return results

    def _matches_metadata_filter(self, key: str, filter_metadata: Dict[str, Any]) -> bool:
        """Check if a vector's metadata matches the filter criteria."""
        vector_metadata = self.metadata[key]
        
        for filter_key, filter_value in filter_metadata.items():
            if filter_key not in vector_metadata:
                return False
            
            # Handle different types of filters
            if isinstance(filter_value, (list, tuple)):
                # Check if any value in the list matches
                if vector_metadata[filter_key] not in filter_value:
                    return False
            elif isinstance(filter_value, dict):
                # Handle range filters like {"min": 0, "max": 100}
                if "min" in filter_value and vector_metadata[filter_key] < filter_value["min"]:
                    return False
                if "max" in filter_value and vector_metadata[filter_key] > filter_value["max"]:
                    return False
            else:
                # Exact match
                if vector_metadata[filter_key] != filter_value:
                    return False
        
        return True

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict[str, Any]]:
        """Retrieve vector and metadata by key."""
        vector = self.vectors.get(key, None)
        metadata = self.metadata.get(key, {})
        return vector, metadata

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a specific key."""
        return self.metadata.get(key, {})

    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific key."""
        if key in self.vectors:
            self.metadata[key].update(metadata)

    def filter_by_metadata(self, filter_metadata: Dict[str, Any]) -> List[str]:
        """Get all keys that match the metadata filter."""
        matching_keys = []
        for key in self.vectors.keys():
            if self._matches_metadata_filter(key, filter_metadata):
                matching_keys.append(key)
        return matching_keys

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics including metadata insights."""
        total_vectors = len(self.vectors)
        
        # Analyze metadata
        metadata_keys = set()
        metadata_values = defaultdict(set)
        
        for metadata in self.metadata.values():
            for key, value in metadata.items():
                metadata_keys.add(key)
                metadata_values[key].add(str(value))
        
        return {
            "total_vectors": total_vectors,
            "metadata_fields": list(metadata_keys),
            "metadata_summary": {key: list(values)[:10] for key, values in metadata_values.items()}
        }

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> "VectorDatabase":
        """Build database from list of texts with optional metadata."""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self.insert_with_metadata(text, np.array(embedding), metadata)
        
        return self

    def save_to_file(self, filename: str) -> None:
        """Save the vector database to a file (vectors and metadata)."""
        # Note: This is a simplified save - in production you'd want a more robust solution
        data = {
            "vectors": {key: vector.tolist() for key, vector in self.vectors.items()},
            "metadata": dict(self.metadata)
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str) -> None:
        """Load the vector database from a file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)
        
        for key, vector_list in data["vectors"].items():
            self.vectors[key] = np.array(vector_list)
        
        for key, metadata in data["metadata"].items():
            self.metadata[key] = metadata


if __name__ == "__main__":
    # Example usage
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
