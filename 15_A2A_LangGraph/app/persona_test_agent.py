import asyncio
import logging
from typing import Dict, Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import AgentCard, Message, Part

class PersonaTestAgent:
    """A simple agent framework for testing an agent with different personas."""
    
    def __init__(self, base_url: str = 'http://localhost:10000'):
        self.base_url = base_url
        self.client = None
        self.agent_card = None
        self.httpx_client = None
        self.client_factory = None
        
    async def initialize(self):
        """Initialize the A2A client connection."""
        # Create a persistent httpx client
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        
        # Create client configuration
        config = ClientConfig(
            streaming=True,
            polling=False,
            httpx_client=self.httpx_client
        )
        
        # Create client factory
        self.client_factory = ClientFactory(config)
        
        # Get agent card and create client
        resolver = A2ACardResolver(httpx_client=self.httpx_client, base_url=self.base_url)
        self.agent_card = await resolver.get_agent_card()
        self.client = self.client_factory.create(self.agent_card)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
    
    async def act_as_persona(self, persona: str, query: str) -> Dict[str, Any]:
        """Act as a specific persona and send a query to your main agent."""
        if not self.client:
            await self.initialize()
            
        # Create the message parts
        parts = [Part(kind='text', text=f'[Persona: {persona}] {query}')]
        
        # Create the message directly (the new Client.send_message expects a Message object)
        message = Message(
            messageId=uuid4().hex,
            role='user',
            parts=parts,
            kind='message'
        )
        
        # Send the message and get response
        # The new Client.send_message returns an async iterator
        response_iterator = self.client.send_message(message)
        
        # Collect all responses from the iterator
        responses = []
        async for response in response_iterator:
            responses.append(response)
        
        # Process the responses to find the final result
        if responses:
            # The last response should be the final result
            last_response = responses[-1]
            
            # Check if it's a tuple (Task, Update) or a Message
            if isinstance(last_response, tuple):
                # It's a (Task, Update) tuple, extract the Task
                task, update = last_response
                return task.model_dump(mode='json', exclude_none=True)
            else:
                # It's a Message object
                return last_response.model_dump(mode='json', exclude_none=True)
        else:
            return {"error": "No response received"}

    def format_response_for_cli(self, response: Dict[str, Any]) -> str:
        """Format the response for better CLI display."""
        try:
            # Extract the main response text from the new format
            if 'artifacts' in response:
                artifacts = response['artifacts']
                if artifacts and 'parts' in artifacts[0]:
                    parts = artifacts[0]['parts']
                    if parts and 'text' in parts[0]:
                        return parts[0]['text']
            
            # Fallback: return the full response as JSON
            import json
            return json.dumps(response, indent=2)
        except Exception as e:
            return f"Error formatting response: {e}"

async def main():
    """Test your agent with different personas."""
    test_agent = PersonaTestAgent()
    
    try:
        personas = {
            "ML Expert": "You are an expert in Machine Learning. You are not satisfied with surface level answers, and you wish to have sources you can read to verify information.",
            "Curious Student": "You are a curious student who wants to understand the basics of AI. Ask simple questions and be satisfied with basic explanations.",
            "Skeptical Researcher": "You are a skeptical researcher who always asks for evidence and sources. You want detailed, verifiable information."
        }
        
        for persona_name, persona_description in personas.items():
            print(f"\n{'='*80}")
            print(f"🧪 TESTING AS: {persona_name}")
            print(f"{'='*80}")
            print(f"📝 Persona Description:")
            print(f"   {persona_description}")
            print(f"\n❓ Query:")
            
            # Test with a relevant query for each persona
            if "ML Expert" in persona_name:
                query = "Talk about Kimi K2 AI model. What does it make it so incredible?"
            elif "Curious Student" in persona_name:
                query = "What is artificial intelligence and how does it work?"
            elif "Skeptical Researcher" in persona_name:
                query = "What are the latest developments in AI? I need verifiable sources and evidence."
            
            print(f"   {query}")
            print(f"\n🔄 Sending request...")
            
            try:
                response = await test_agent.act_as_persona(persona_name, query)
                print(f"\n✅ Response received successfully!")
                print(f"\n📋 Agent Response:")
                print(f"{'-'*80}")
                
                # Format and display the response
                formatted_response = test_agent.format_response_for_cli(response)
                print(formatted_response)
                
                print(f"{'-'*80}")
                
            except Exception as e:
                print(f"\n❌ Error testing {persona_name}: {e}")
                print(f"{'-'*80}")
        
        print(f"\n{'='*80}")
        print(f"🎉 All persona tests completed!")
        print(f"{'='*80}")
    
    finally:
        # Always clean up resources
        await test_agent.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
