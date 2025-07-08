import os
from dotenv import load_dotenv
import openai

load_dotenv()

class ChatInterface:    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_response(self, messages) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=list(messages)
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            return f"Error generating response: {e}"