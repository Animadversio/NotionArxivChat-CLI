"""
Modern LLM Application Core
Provides abstract interfaces and modern patterns for LLM applications
"""
import abc
import asyncio
import json
import time
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class ChatMessage:
    """Standardized chat message format"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalContext:
    """Context from document retrieval"""
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = None


@dataclass
class ChatResponse:
    """Standardized response format"""
    message: ChatMessage
    sources: List[RetrievalContext] = None
    usage: Dict[str, int] = None
    processing_time: float = None


class ConversationMemory:
    """Manages conversation history with different strategies"""
    
    def __init__(self, max_messages: int = 20, strategy: str = "sliding_window"):
        self.max_messages = max_messages
        self.strategy = strategy
        self.messages: List[ChatMessage] = []
    
    def add_message(self, message: ChatMessage):
        """Add message to memory"""
        self.messages.append(message)
        self._apply_strategy()
    
    def _apply_strategy(self):
        """Apply memory management strategy"""
        if self.strategy == "sliding_window" and len(self.messages) > self.max_messages:
            # Keep system messages + recent messages
            system_msgs = [msg for msg in self.messages if msg.role == "system"]
            recent_msgs = [msg for msg in self.messages if msg.role != "system"][-self.max_messages:]
            self.messages = system_msgs + recent_msgs
    
    def get_context(self) -> List[Dict[str, str]]:
        """Get conversation context for LLM"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    def save(self, filepath: Path):
        """Save conversation to file"""
        data = {
            "messages": [asdict(msg) for msg in self.messages],
            "metadata": {
                "max_messages": self.max_messages,
                "strategy": self.strategy
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ConversationMemory':
        """Load conversation from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        memory = cls(
            max_messages=data["metadata"]["max_messages"],
            strategy=data["metadata"]["strategy"]
        )
        memory.messages = [ChatMessage(**msg_data) for msg_data in data["messages"]]
        return memory


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers"""
    
    @abc.abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse:
        """Generate chat response"""
        pass
    
    @abc.abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        pass
    
    @abc.abstractmethod
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", embedding_model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required for OpenAI provider")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse:
        """Generate chat response"""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}
            
            return ChatResponse(
                message=ChatMessage(role="assistant", content=content),
                usage=usage,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required for Anthropic provider")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse:
        """Generate chat response"""
        start_time = time.time()
        
        try:
            # Convert messages format for Anthropic
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            chat_messages = [msg for msg in messages if msg["role"] != "system"]
            
            response = await self.client.messages.create(
                model=self.model,
                system=system_msg,
                messages=chat_messages,
                max_tokens=kwargs.get("max_tokens", 4000),
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            content = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            
            return ChatResponse(
                message=ChatMessage(role="assistant", content=content),
                usage=usage,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Anthropic doesn't provide embeddings - use a different provider"""
        raise NotImplementedError("Anthropic doesn't provide embeddings")
    
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        try:
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            chat_messages = [msg for msg in messages if msg["role"] != "system"]
            
            async with self.client.messages.stream(
                model=self.model,
                system=system_msg,
                messages=chat_messages,
                max_tokens=kwargs.get("max_tokens", 4000),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(provider_type: ModelProvider, **kwargs) -> LLMProvider:
        """Create LLM provider instance"""
        if provider_type == ModelProvider.OPENAI:
            return OpenAIProvider(**kwargs)
        elif provider_type == ModelProvider.ANTHROPIC:
            return AnthropicProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def acquire(self):
        """Acquire rate limit permission"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)


class ConfigManager:
    """Configuration management with validation"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path.exists():
            return self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix == '.json':
                return json.load(f)
            elif self.config_path.suffix in ['.yml', '.yaml']:
                import yaml
                return yaml.safe_load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4000
            },
            "embeddings": {
                "model": "text-embedding-3-small",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "conversation": {
                "max_messages": 20,
                "memory_strategy": "sliding_window"
            },
            "storage": {
                "pdf_root": "./pdfs",
                "embed_root": "./embeddings",
                "conversations_root": "./conversations"
            },
            "rate_limiting": {
                "calls_per_minute": 60
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)