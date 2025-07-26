"""
Modern Chat Engine
Advanced chat patterns with streaming, function calling, and multi-modal support
"""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
import logging
from pathlib import Path

from modern_llm_core import (
    ChatMessage, ChatResponse, ConversationMemory, LLMProvider, 
    RetrievalContext, ConfigManager, RateLimiter
)
from modern_retrieval import ModernRAGPipeline, HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Represents a function call from LLM"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class FunctionResult:
    """Result of function execution"""
    call_id: str
    result: Any
    error: Optional[str] = None


class FunctionRegistry:
    """Registry for LLM-callable functions"""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict] = {}
    
    def register(self, name: str, func: Callable, schema: Dict):
        """Register a function with its schema"""
        self.functions[name] = func
        self.schemas[name] = schema
        logger.info(f"Registered function: {name}")
    
    async def call(self, function_call: FunctionCall) -> FunctionResult:
        """Execute a function call"""
        try:
            if function_call.name not in self.functions:
                return FunctionResult(
                    call_id=function_call.call_id,
                    result=None,
                    error=f"Function {function_call.name} not found"
                )
            
            func = self.functions[function_call.name]
            
            # Check if function is async
            if asyncio.iscoroutinefunction(func):
                result = await func(**function_call.arguments)
            else:
                result = func(**function_call.arguments)
            
            return FunctionResult(
                call_id=function_call.call_id,
                result=result
            )
            
        except Exception as e:
            logger.error(f"Error executing function {function_call.name}: {e}")
            return FunctionResult(
                call_id=function_call.call_id,
                result=None,
                error=str(e)
            )
    
    def get_schemas(self) -> List[Dict]:
        """Get all function schemas for LLM"""
        return list(self.schemas.values())


class CitationManager:
    """Manages citations and source attribution"""
    
    def __init__(self):
        self.citations: Dict[str, RetrievalContext] = {}
        self.citation_counter = 0
    
    def add_citation(self, context: RetrievalContext) -> str:
        """Add citation and return citation ID"""
        self.citation_counter += 1
        citation_id = f"[{self.citation_counter}]"
        self.citations[citation_id] = context
        return citation_id
    
    def format_citations(self) -> str:
        """Format all citations as references"""
        if not self.citations:
            return ""
        
        references = ["## References"]
        for citation_id, context in self.citations.items():
            source = Path(context.source).name
            page_info = ""
            if context.metadata and 'page_number' in context.metadata:
                page_info = f", p. {context.metadata['page_number']}"
            
            references.append(f"{citation_id} {source}{page_info}")
        
        return "\n".join(references)
    
    def clear(self):
        """Clear all citations"""
        self.citations.clear()
        self.citation_counter = 0


class ModernChatEngine:
    """Modern chat engine with RAG, streaming, and function calling"""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        rag_pipeline: ModernRAGPipeline,
        config: ConfigManager,
        function_registry: Optional[FunctionRegistry] = None
    ):
        self.llm_provider = llm_provider
        self.rag_pipeline = rag_pipeline
        self.config = config
        self.function_registry = function_registry or FunctionRegistry()
        self.rate_limiter = RateLimiter(config.get('rate_limiting.calls_per_minute', 60))
        self.citation_manager = CitationManager()
        
        # Initialize conversation memory
        self.memory = ConversationMemory(
            max_messages=config.get('conversation.max_messages', 20),
            strategy=config.get('conversation.memory_strategy', 'sliding_window')
        )
        
        # Add system message
        system_msg = ChatMessage(
            role="system",
            content=self._get_system_prompt()
        )
        self.memory.add_message(system_msg)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the chat"""
        return """You are an AI assistant specialized in helping users understand and discuss academic papers. 

Key guidelines:
1. Provide accurate, well-sourced information based on the retrieved document contexts
2. When referencing information from documents, use citations like [1], [2], etc.
3. If you're unsure about something, clearly state your uncertainty
4. Break down complex concepts into understandable explanations
5. Suggest follow-up questions when appropriate

You have access to document retrieval capabilities and can call functions when needed."""
    
    async def chat(
        self,
        user_message: str,
        document_ids: List[str] = None,
        use_rag: bool = True,
        stream: bool = False
    ) -> AsyncGenerator[str, None] if stream else ChatResponse:
        """Main chat function"""
        await self.rate_limiter.acquire()
        
        # Add user message to memory
        user_msg = ChatMessage(role="user", content=user_message)
        self.memory.add_message(user_msg)
        
        # Clear previous citations
        self.citation_manager.clear()
        
        # Retrieve relevant contexts if RAG is enabled
        contexts = []
        if use_rag:
            retriever = HybridRetriever(self.rag_pipeline)
            contexts = await retriever.retrieve(
                query=user_message,
                document_ids=document_ids,
                k=self.config.get('retrieval.max_contexts', 5)
            )
        
        # Build enhanced prompt with contexts
        enhanced_messages = self._build_rag_messages(user_message, contexts)
        
        if stream:
            return self._stream_response(enhanced_messages, contexts)
        else:
            return await self._generate_response(enhanced_messages, contexts)
    
    async def _stream_response(
        self, 
        messages: List[Dict[str, str]], 
        contexts: List[RetrievalContext]
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        full_response = ""
        
        try:
            async for chunk in self.llm_provider.stream_chat(
                messages=messages,
                temperature=self.config.get('llm.temperature', 0.7),
                max_tokens=self.config.get('llm.max_tokens', 4000)
            ):
                full_response += chunk
                yield chunk
            
            # Add assistant message to memory
            assistant_msg = ChatMessage(role="assistant", content=full_response)
            self.memory.add_message(assistant_msg)
            
            # Add citations at the end
            citations = self.citation_manager.format_citations()
            if citations:
                yield f"\n\n{citations}"
                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
    
    async def _generate_response(
        self, 
        messages: List[Dict[str, str]], 
        contexts: List[RetrievalContext]
    ) -> ChatResponse:
        """Generate non-streaming response"""
        try:
            response = await self.llm_provider.chat(
                messages=messages,
                temperature=self.config.get('llm.temperature', 0.7),
                max_tokens=self.config.get('llm.max_tokens', 4000)
            )
            
            # Add citations to response
            citations = self.citation_manager.format_citations()
            if citations:
                response.message.content += f"\n\n{citations}"
            
            # Add assistant message to memory
            self.memory.add_message(response.message)
            
            # Add sources to response
            response.sources = contexts
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = ChatMessage(role="assistant", content=f"Sorry, I encountered an error: {str(e)}")
            return ChatResponse(message=error_msg)
    
    def _build_rag_messages(
        self, 
        user_message: str, 
        contexts: List[RetrievalContext]
    ) -> List[Dict[str, str]]:
        """Build messages with RAG context"""
        messages = self.memory.get_context()
        
        if contexts:
            # Add context information
            context_text = "Here are relevant excerpts from the documents:\n\n"
            
            for i, context in enumerate(contexts):
                citation_id = self.citation_manager.add_citation(context)
                context_text += f"{citation_id} {context.content}\n\n"
            
            # Insert context before the last user message
            if messages and messages[-1]["role"] == "user":
                # Replace the last user message with context + user message
                enhanced_user_msg = f"{context_text}User question: {user_message}"
                messages[-1]["content"] = enhanced_user_msg
        
        return messages
    
    async def ask_with_followup(
        self,
        initial_question: str,
        document_ids: List[str] = None,
        max_followups: int = 3
    ) -> List[ChatResponse]:
        """Ask question with automatic follow-up suggestions"""
        responses = []
        
        # Initial response
        response = await self.chat(initial_question, document_ids, stream=False)
        responses.append(response)
        
        # Generate follow-up questions
        followup_count = 0
        while followup_count < max_followups:
            followup_questions = await self._generate_followup_questions(response.message.content)
            
            if not followup_questions:
                break
            
            # For demo, just ask the first follow-up
            # In practice, you'd present these to the user
            followup = followup_questions[0]
            response = await self.chat(followup, document_ids, stream=False)
            responses.append(response)
            
            followup_count += 1
        
        return responses
    
    async def _generate_followup_questions(self, assistant_response: str) -> List[str]:
        """Generate follow-up questions based on response"""
        followup_prompt = f"""Based on this response about academic papers:

{assistant_response}

Generate 2-3 insightful follow-up questions that would help the user understand the topic better. 
Return only the questions, one per line."""
        
        try:
            response = await self.llm_provider.chat(
                messages=[{"role": "user", "content": followup_prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            questions = [q.strip() for q in response.message.content.split('\n') if q.strip()]
            return questions[:3]  # Return max 3 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    async def summarize_conversation(self) -> str:
        """Summarize the current conversation"""
        if len(self.memory.messages) <= 2:  # Just system + user message
            return "No conversation to summarize yet."
        
        conversation_text = ""
        for msg in self.memory.messages[1:]:  # Skip system message
            conversation_text += f"{msg.role.capitalize()}: {msg.content}\n\n"
        
        summary_prompt = f"""Summarize this conversation about academic papers:

{conversation_text}

Provide a concise summary highlighting:
1. Main topics discussed
2. Key insights or findings
3. Any remaining questions or areas for further exploration"""
        
        try:
            response = await self.llm_provider.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.5,
                max_tokens=500
            )
            
            return response.message.content
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
            return "Error generating summary."
    
    async def save_conversation(self, filepath: Path):
        """Save conversation to file"""
        try:
            # Add summary to conversation
            summary = await self.summarize_conversation()
            
            conversation_data = {
                "summary": summary,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata
                    }
                    for msg in self.memory.messages
                ],
                "citations": {
                    citation_id: {
                        "source": context.source,
                        "content": context.content,
                        "relevance_score": context.relevance_score,
                        "metadata": context.metadata
                    }
                    for citation_id, context in self.citation_manager.citations.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            logger.info(f"Conversation saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def load_conversation(self, filepath: Path):
        """Load conversation from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore messages
            self.memory.messages = []
            for msg_data in data["messages"]:
                msg = ChatMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    metadata=msg_data["metadata"]
                )
                self.memory.messages.append(msg)
            
            # Restore citations
            self.citation_manager.clear()
            for citation_id, context_data in data.get("citations", {}).items():
                context = RetrievalContext(
                    content=context_data["content"],
                    source=context_data["source"],
                    relevance_score=context_data["relevance_score"],
                    metadata=context_data["metadata"]
                )
                self.citation_manager.citations[citation_id] = context
            
            logger.info(f"Conversation loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
    
    def clear_conversation(self):
        """Clear current conversation"""
        self.memory = ConversationMemory(
            max_messages=self.config.get('conversation.max_messages', 20),
            strategy=self.config.get('conversation.memory_strategy', 'sliding_window')
        )
        
        # Re-add system message
        system_msg = ChatMessage(
            role="system",
            content=self._get_system_prompt()
        )
        self.memory.add_message(system_msg)
        
        self.citation_manager.clear()
        logger.info("Conversation cleared")


# Example function for function calling
async def search_papers_by_topic(topic: str, max_results: int = 5) -> Dict[str, Any]:
    """Example function to search papers by topic"""
    # This would integrate with ArXiv API
    return {
        "topic": topic,
        "results": [
            {"title": f"Paper about {topic} #{i}", "arxiv_id": f"2024.{i:04d}"}
            for i in range(1, max_results + 1)
        ]
    }


# Function schema for the above function
SEARCH_PAPERS_SCHEMA = {
    "name": "search_papers_by_topic",
    "description": "Search for academic papers by topic",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic to search for"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            }
        },
        "required": ["topic"]
    }
}