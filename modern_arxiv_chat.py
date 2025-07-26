"""
Modern ArXiv Chat Application
Modernized version of the ArXiv chat system using new architecture
"""
import asyncio
import os
from pathlib import Path
from typing import Optional, List
import questionary
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
import logging

from modern_llm_core import (
    ModelProvider, LLMProviderFactory, ConfigManager, 
    OpenAIProvider, ConversationMemory
)
from modern_retrieval import ModernRAGPipeline
from modern_chat_engine import ModernChatEngine, FunctionRegistry, search_papers_by_topic, SEARCH_PAPERS_SCHEMA
from arxiv_browse_lib import arxiv_paper_download, fetch_K_results
import arxiv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModernArxivChatApp:
    """Modern ArXiv Chat Application"""
    
    def __init__(self, config_path: str = "modern_config.json"):
        self.config = ConfigManager(Path(config_path))
        self.llm_provider = None
        self.rag_pipeline = None
        self.chat_engine = None
        self.current_paper_id = None
        
        # Setup prompt session for chat
        self.chat_history = FileHistory("modern_qa_chat_history.txt")
        self.chat_session = PromptSession(history=self.chat_history)
        
        # Setup prompt session for search
        self.search_history = FileHistory("modern_arxiv_search_history.txt")
        self.search_session = PromptSession(history=self.search_history)
    
    async def initialize(self):
        """Initialize the application"""
        logger.info("Initializing Modern ArXiv Chat Application...")
        
        # Check for required environment variables
        self._check_environment()
        
        # Initialize LLM provider
        await self._initialize_llm_provider()
        
        # Initialize RAG pipeline
        await self._initialize_rag_pipeline()
        
        # Initialize chat engine
        await self._initialize_chat_engine()
        
        logger.info("Application initialized successfully!")
    
    def _check_environment(self):
        """Check required environment variables"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Please set the following environment variables: {missing_vars}")
    
    async def _initialize_llm_provider(self):
        """Initialize LLM provider"""
        provider_type = self.config.get('llm.provider', 'openai')
        
        if provider_type == 'openai':
            self.llm_provider = OpenAIProvider(
                api_key=os.getenv('OPENAI_API_KEY'),
                model=self.config.get('llm.model', 'gpt-4'),
                embedding_model=self.config.get('embeddings.model', 'text-embedding-3-small')
            )
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
        
        logger.info(f"LLM provider initialized: {provider_type}")
    
    async def _initialize_rag_pipeline(self):
        """Initialize RAG pipeline"""
        self.rag_pipeline = ModernRAGPipeline(
            config=self.config,
            embedding_function=self.llm_provider  # OpenAIProvider can be used as embedding function
        )
        logger.info("RAG pipeline initialized")
    
    async def _initialize_chat_engine(self):
        """Initialize chat engine with function registry"""
        function_registry = FunctionRegistry()
        
        # Register search function
        function_registry.register(
            name="search_papers_by_topic",
            func=search_papers_by_topic,
            schema=SEARCH_PAPERS_SCHEMA
        )
        
        self.chat_engine = ModernChatEngine(
            llm_provider=self.llm_provider,
            rag_pipeline=self.rag_pipeline,
            config=self.config,
            function_registry=function_registry
        )
        logger.info("Chat engine initialized with function registry")
    
    async def run(self):
        """Main application loop"""
        print("🚀 Welcome to Modern ArXiv Chat!")
        print("Type 'help' for available commands or start by searching for papers.")
        
        while True:
            try:
                action = await self._get_main_action()
                
                if action == "search_papers":
                    await self._search_and_select_paper()
                elif action == "chat_with_paper":
                    if self.current_paper_id:
                        await self._chat_with_current_paper()
                    else:
                        print("❌ No paper selected. Please search and select a paper first.")
                elif action == "view_conversation":
                    await self._view_conversation_summary()
                elif action == "save_conversation":
                    await self._save_conversation()
                elif action == "load_conversation":
                    await self._load_conversation()
                elif action == "clear_conversation":
                    await self._clear_conversation()
                elif action == "settings":
                    await self._show_settings()
                elif action == "help":
                    self._show_help()
                elif action == "exit":
                    print("👋 Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"❌ An error occurred: {e}")
    
    async def _get_main_action(self) -> str:
        """Get main action from user"""
        choices = [
            "search_papers",
            "chat_with_paper",
            "view_conversation", 
            "save_conversation",
            "load_conversation",
            "clear_conversation",
            "settings",
            "help",
            "exit"
        ]
        
        return questionary.select(
            "What would you like to do?",
            choices=choices,
            instruction="(Use arrow keys to navigate)"
        ).ask()
    
    async def _search_and_select_paper(self):
        """Search for papers and select one"""
        try:
            query = self.search_session.prompt("Enter ArXiv search query: ")
            if not query.strip():
                return
            
            print(f"🔍 Searching for: {query}")
            
            # Search ArXiv
            search = arxiv.Search(
                query=query,
                max_results=self.config.get('search.max_results', 10),
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = list(search.results())
            
            if not results:
                print("❌ No papers found for your query.")
                return
            
            # Display results
            print(f"\n📄 Found {len(results)} papers:")
            choices = []
            for i, paper in enumerate(results):
                title = paper.title[:80] + "..." if len(paper.title) > 80 else paper.title
                arxiv_id = paper.entry_id.split("/")[-1]
                choices.append(f"[{arxiv_id}] {title}")
            
            choices.append("🔙 Back to main menu")
            
            selection = questionary.select(
                "Select a paper to chat with:",
                choices=choices
            ).ask()
            
            if selection == "🔙 Back to main menu":
                return
            
            # Extract ArXiv ID
            selected_idx = choices.index(selection)
            selected_paper = results[selected_idx]
            self.current_paper_id = selected_paper.entry_id.split("/")[-1]
            
            print(f"📑 Selected paper: {selected_paper.title}")
            print(f"🆔 ArXiv ID: {self.current_paper_id}")
            
            # Download and process paper
            await self._download_and_process_paper(selected_paper)
            
        except Exception as e:
            logger.error(f"Error in paper search: {e}")
            print(f"❌ Search failed: {e}")
    
    async def _download_and_process_paper(self, paper: arxiv.Result):
        """Download and process the selected paper"""
        try:
            print(f"📥 Downloading paper: {self.current_paper_id}")
            
            # Create directories
            pdf_root = Path(self.config.get('storage.pdf_root', './pdfs'))
            pdf_root.mkdir(exist_ok=True)
            
            # Download paper (reuse existing function but make it async-compatible)
            loop = asyncio.get_event_loop()
            pages = await loop.run_in_executor(
                None, 
                arxiv_paper_download, 
                self.current_paper_id, 
                str(pdf_root)
            )
            
            if not pages:
                print("❌ Failed to download paper content")
                return
            
            print(f"✅ Downloaded {len(pages)} pages")
            
            # Process with RAG pipeline
            print("🧠 Processing paper for chat...")
            paper_path = pdf_root / f"{self.current_paper_id}.pdf"
            if not paper_path.exists():
                paper_path = pdf_root / f"{self.current_paper_id}.html"
            
            success = await self.rag_pipeline.process_document(
                file_path=paper_path,
                document_id=self.current_paper_id
            )
            
            if success:
                print("✅ Paper processed successfully! Ready for chat.")
            else:
                print("❌ Failed to process paper for chat")
                
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            print(f"❌ Processing failed: {e}")
    
    async def _chat_with_current_paper(self):
        """Chat with the currently selected paper"""
        if not self.current_paper_id:
            print("❌ No paper selected")
            return
        
        print(f"💬 Chatting with paper: {self.current_paper_id}")
        print("Type 'exit' to return to main menu, 'stream' to toggle streaming mode")
        
        streaming_mode = self.config.get('chat.streaming', True)
        
        while True:
            try:
                question = self.chat_session.prompt("❓ Your question: ")
                
                if not question.strip():
                    continue
                
                if question.lower() == 'exit':
                    break
                elif question.lower() == 'stream':
                    streaming_mode = not streaming_mode
                    print(f"🔄 Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                    continue
                
                print("🤔 Thinking...")
                
                if streaming_mode:
                    print("💭 Response:")
                    async for chunk in self.chat_engine.chat(
                        user_message=question,
                        document_ids=[self.current_paper_id],
                        stream=True
                    ):
                        print(chunk, end='', flush=True)
                    print("\n")
                else:
                    response = await self.chat_engine.chat(
                        user_message=question,
                        document_ids=[self.current_paper_id],
                        stream=False
                    )
                    print(f"💭 Response:\n{response.message.content}\n")
                
            except KeyboardInterrupt:
                print("\n🔙 Returning to main menu...")
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                print(f"❌ Chat error: {e}")
    
    async def _view_conversation_summary(self):
        """View conversation summary"""
        try:
            summary = await self.chat_engine.summarize_conversation()
            print(f"\n📋 Conversation Summary:\n{summary}\n")
        except Exception as e:
            logger.error(f"Error viewing summary: {e}")
            print(f"❌ Failed to generate summary: {e}")
    
    async def _save_conversation(self):
        """Save current conversation"""
        try:
            conversations_dir = Path(self.config.get('storage.conversations_root', './conversations'))
            conversations_dir.mkdir(exist_ok=True)
            
            filename = questionary.text(
                "Enter filename for conversation:",
                default=f"conversation_{self.current_paper_id or 'general'}.json"
            ).ask()
            
            if filename:
                filepath = conversations_dir / filename
                await self.chat_engine.save_conversation(filepath)
                print(f"💾 Conversation saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            print(f"❌ Failed to save conversation: {e}")
    
    async def _load_conversation(self):
        """Load a saved conversation"""
        try:
            conversations_dir = Path(self.config.get('storage.conversations_root', './conversations'))
            if not conversations_dir.exists():
                print("❌ No saved conversations found")
                return
            
            # List available conversations
            conversation_files = list(conversations_dir.glob("*.json"))
            if not conversation_files:
                print("❌ No saved conversations found")
                return
            
            choices = [f.name for f in conversation_files] + ["🔙 Cancel"]
            
            selection = questionary.select(
                "Select conversation to load:",
                choices=choices
            ).ask()
            
            if selection == "🔙 Cancel":
                return
            
            filepath = conversations_dir / selection
            self.chat_engine.load_conversation(filepath)
            print(f"📂 Conversation loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            print(f"❌ Failed to load conversation: {e}")
    
    async def _clear_conversation(self):
        """Clear current conversation"""
        confirm = questionary.confirm("Are you sure you want to clear the conversation?").ask()
        if confirm:
            self.chat_engine.clear_conversation()
            print("🗑️ Conversation cleared")
    
    async def _show_settings(self):
        """Show and modify settings"""
        settings_menu = [
            "View current settings",
            "Change LLM model",
            "Change temperature",
            "Change max tokens",
            "Toggle streaming mode",
            "🔙 Back to main menu"
        ]
        
        selection = questionary.select("Settings:", choices=settings_menu).ask()
        
        if selection == "View current settings":
            print("\n⚙️ Current Settings:")
            print(f"LLM Provider: {self.config.get('llm.provider')}")
            print(f"Model: {self.config.get('llm.model')}")
            print(f"Temperature: {self.config.get('llm.temperature')}")
            print(f"Max Tokens: {self.config.get('llm.max_tokens')}")
            print(f"Streaming: {self.config.get('chat.streaming')}")
            
        elif selection == "Change LLM model":
            models = ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
            new_model = questionary.select("Select model:", choices=models).ask()
            self.config.update('llm.model', new_model)
            print(f"✅ Model changed to {new_model}")
            
        elif selection == "Change temperature":
            temp = questionary.text(
                "Enter temperature (0.0-2.0):",
                default=str(self.config.get('llm.temperature', 0.7))
            ).ask()
            try:
                temp_float = float(temp)
                if 0.0 <= temp_float <= 2.0:
                    self.config.update('llm.temperature', temp_float)
                    print(f"✅ Temperature changed to {temp_float}")
                else:
                    print("❌ Temperature must be between 0.0 and 2.0")
            except ValueError:
                print("❌ Invalid temperature value")
                
        elif selection == "Toggle streaming mode":
            current = self.config.get('chat.streaming', True)
            self.config.update('chat.streaming', not current)
            print(f"✅ Streaming mode: {'ON' if not current else 'OFF'}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
🚀 Modern ArXiv Chat Help

Available Actions:
📄 Search Papers    - Search ArXiv and select a paper to chat with
💬 Chat with Paper  - Ask questions about the selected paper
📋 View Summary     - See a summary of your conversation
💾 Save Conversation - Save current chat to file
📂 Load Conversation - Load a previously saved chat
🗑️  Clear Conversation - Start a new conversation
⚙️  Settings        - Modify application settings
❓ Help            - Show this help message
🚪 Exit            - Quit the application

Chat Features:
🔄 Type 'stream' during chat to toggle streaming mode
🚪 Type 'exit' during chat to return to main menu
📚 Citations are automatically included in responses
🧠 Conversations have persistent memory

Tips:
• Use specific questions for better responses
• The system remembers your conversation context
• Citations link back to specific parts of the paper
• You can save and resume conversations anytime
        """
        print(help_text)


async def main():
    """Main entry point"""
    app = ModernArxivChatApp()
    
    try:
        await app.initialize()
        await app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application failed to start: {e}")


if __name__ == "__main__":
    asyncio.run(main())