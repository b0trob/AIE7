import os

from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.agent_graph_with_helpfulness import build_agent_graph_with_helpfulness


memory = MemorySaver()

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class SplitAgent:
    """Agent - a general-purpose assistant with access to web search, academic papers, and RAG."""

    SYSTEM_INSTRUCTION = (
        'You are Kevin Wendell Crumb, a man with dissociative identity disorder (DID) who has 23 distinct personalities. '
        'You are currently in a conversation with a user, and you are going to answer the user\'s questions, but '
        'change your personality to the one that is most appropriate to answer the question, or just answer the question in the style of Kevin Wendell Crumb. '
        'If you cannot find relevant information using the available tools, clearly state that you were unable to find the requested information.'
    )

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self):
        self.model = ChatOpenAI(
            model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
            temperature=0,
        )
        # Use the new graph with helpfulness evaluation for A2A protocol compatibility
        self.graph = build_agent_graph_with_helpfulness(
            self.model,
            self.SYSTEM_INSTRUCTION,
            self.FORMAT_INSTRUCTION,
            checkpointer=memory
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        # Track if we've seen tool execution
        tool_executed = False
        
        try:
            for item in self.graph.stream(inputs, config, stream_mode='values'):
                message = item['messages'][-1]
                if (
                    isinstance(message, AIMessage)
                    and message.tool_calls
                    and len(message.tool_calls) > 0
                ):
                    tool_executed = True
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': 'Searching for information...',
                    }
                elif isinstance(message, ToolMessage):
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': 'Processing the results...',
                    }

            # Get the final state after the graph completes
            final_state = self.graph.get_state(config)
            final_response = self.get_agent_response_from_state(final_state)
            yield final_response
            
        except Exception as e:
            # Yield an error response
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f'An error occurred: {str(e)}',
            }

    def get_agent_response_from_state(self, state):
        """Extract the final response from the graph state."""
        structured_response = state.values.get('structured_response')
        
        if structured_response and hasattr(structured_response, 'status'):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            elif structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            elif structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }
        
        # If no structured response, try to get the last message
        messages = state.values.get('messages', [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                # Check if it's a helpfulness message
                if 'HELPFULNESS:' in last_message.content:
                    # Get the actual response from the previous message
                    if len(messages) > 1:
                        actual_response = messages[-2]
                        if hasattr(actual_response, 'content'):
                            return {
                                'is_task_complete': True,
                                'require_user_input': False,
                                'content': actual_response.content,
                            }
                
                # Return the last message content
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': last_message.content,
                }
        
        # Fallback response
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    def get_agent_response(self, config):
        """Legacy method - kept for compatibility but not used."""
        current_state = self.graph.get_state(config)
        return self.get_agent_response_from_state(current_state)

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
