from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import os
import asyncio


class EmbeddingModel:
    def __init__(self, embeddings_model_name: str = "text-embedding-3-small"):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)

        if self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key."
            )
        self.embeddings_model_name = embeddings_model_name

    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        embedding_response = self.client.embeddings.create(
            input=list_of_text, model=self.embeddings_model_name
        )

        return [embeddings.embedding for embeddings in embedding_response.data]

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.client.embeddings.create(
            input=text, model=self.embeddings_model_name
        )

        return embedding.data[0].embedding


if __name__ == "__main__":
    embedding_model = EmbeddingModel()