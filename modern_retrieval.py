"""
Modern Document Retrieval and Processing
Advanced RAG patterns with async processing and better document handling
"""
import asyncio
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("Warning: LangChain dependencies not found. Some features may not work.")

from modern_llm_core import RetrievalContext, ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    source: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    @abstractmethod
    async def process(self, file_path: Path) -> List[DocumentChunk]:
        """Process document and return chunks"""
        pass
    
    @abstractmethod
    def supports_format(self, file_path: Path) -> bool:
        """Check if processor supports file format"""
        pass


class PDFProcessor(DocumentProcessor):
    """PDF document processor"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def supports_format(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'
    
    async def process(self, file_path: Path) -> List[DocumentChunk]:
        """Process PDF document"""
        try:
            # Load PDF in executor to avoid blocking
            loop = asyncio.get_event_loop()
            loader = PyPDFLoader(str(file_path))
            documents = await loop.run_in_executor(None, loader.load)
            
            chunks = []
            chunk_index = 0
            
            for doc in documents:
                page_chunks = self.text_splitter.split_text(doc.page_content)
                page_num = doc.metadata.get('page', None)
                
                for chunk_text in page_chunks:
                    chunk = DocumentChunk(
                        content=chunk_text,
                        source=str(file_path),
                        page_number=page_num,
                        chunk_index=chunk_index,
                        metadata=doc.metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise


class HTMLProcessor(DocumentProcessor):
    """HTML document processor"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def supports_format(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.html', '.htm']
    
    async def process(self, file_path: Path) -> List[DocumentChunk]:
        """Process HTML document"""
        try:
            loop = asyncio.get_event_loop()
            loader = BSHTMLLoader(str(file_path), open_encoding="utf8")
            documents = await loop.run_in_executor(None, loader.load)
            
            chunks = []
            chunk_index = 0
            
            for doc in documents:
                page_chunks = self.text_splitter.split_text(doc.page_content)
                
                for chunk_text in page_chunks:
                    chunk = DocumentChunk(
                        content=chunk_text,
                        source=str(file_path),
                        chunk_index=chunk_index,
                        metadata=doc.metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing HTML {file_path}: {e}")
            raise


class DocumentProcessorFactory:
    """Factory for document processors"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.processors = [
            PDFProcessor(
                chunk_size=config.get('embeddings.chunk_size', 1000),
                chunk_overlap=config.get('embeddings.chunk_overlap', 200)
            ),
            HTMLProcessor(
                chunk_size=config.get('embeddings.chunk_size', 1000),
                chunk_overlap=config.get('embeddings.chunk_overlap', 200)
            )
        ]
    
    def get_processor(self, file_path: Path) -> Optional[DocumentProcessor]:
        """Get appropriate processor for file"""
        for processor in self.processors:
            if processor.supports_format(file_path):
                return processor
        return None


class VectorStore:
    """Enhanced vector store with caching and async operations"""
    
    def __init__(self, persist_directory: Path, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._store = None
        self._cache: Dict[str, List[DocumentChunk]] = {}
    
    async def initialize(self):
        """Initialize vector store"""
        if self.persist_directory.exists():
            self._store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_function
            )
        else:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    async def add_documents(self, chunks: List[DocumentChunk], document_id: str):
        """Add document chunks to vector store"""
        if not chunks:
            return
        
        # Create cache key
        cache_key = self._get_cache_key(document_id, chunks)
        
        # Check if already cached
        if cache_key in self._cache:
            logger.info(f"Using cached embeddings for {document_id}")
            return
        
        # Convert chunks to LangChain documents
        documents = []
        metadatas = []
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    'source': chunk.source,
                    'chunk_index': chunk.chunk_index,
                    'page_number': chunk.page_number,
                    'document_id': document_id
                }
            )
            documents.append(doc)
            metadatas.append(doc.metadata)
        
        # Add to vector store
        if self._store is None:
            self._store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=str(self.persist_directory)
            )
        else:
            self._store.add_documents(documents)
        
        # Cache the chunks
        self._cache[cache_key] = chunks
        
        # Persist the store
        self._store.persist()
        
        logger.info(f"Added {len(chunks)} chunks for document {document_id}")
    
    async def similarity_search(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[RetrievalContext]:
        """Perform similarity search"""
        if self._store is None:
            return []
        
        try:
            # Perform search
            results = self._store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # Convert to RetrievalContext
            contexts = []
            for doc, score in results:
                context = RetrievalContext(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'Unknown'),
                    relevance_score=score,
                    metadata=doc.metadata
                )
                contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _get_cache_key(self, document_id: str, chunks: List[DocumentChunk]) -> str:
        """Generate cache key for document chunks"""
        content_hash = hashlib.md5()
        for chunk in chunks:
            content_hash.update(chunk.content.encode('utf-8'))
        return f"{document_id}_{content_hash.hexdigest()}"
    
    def get_document_count(self) -> int:
        """Get number of documents in store"""
        if self._store is None:
            return 0
        return self._store._collection.count()


class ModernRAGPipeline:
    """Modern RAG pipeline with async processing"""
    
    def __init__(self, config: ConfigManager, embedding_function):
        self.config = config
        self.embedding_function = embedding_function
        self.processor_factory = DocumentProcessorFactory(config)
        self.vector_stores: Dict[str, VectorStore] = {}
    
    async def process_document(self, file_path: Path, document_id: str = None) -> bool:
        """Process document and add to vector store"""
        if document_id is None:
            document_id = file_path.stem
        
        # Get appropriate processor
        processor = self.processor_factory.get_processor(file_path)
        if processor is None:
            logger.error(f"No processor found for {file_path}")
            return False
        
        try:
            # Process document
            logger.info(f"Processing document: {file_path}")
            chunks = await processor.process(file_path)
            
            if not chunks:
                logger.warning(f"No chunks extracted from {file_path}")
                return False
            
            # Get or create vector store for this document
            vector_store = await self._get_vector_store(document_id)
            
            # Add chunks to vector store
            await vector_store.add_documents(chunks, document_id)
            
            logger.info(f"Successfully processed {file_path} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return False
    
    async def query_documents(
        self, 
        query: str, 
        document_ids: List[str] = None, 
        k: int = 5
    ) -> List[RetrievalContext]:
        """Query documents and return relevant contexts"""
        all_contexts = []
        
        # If no specific documents specified, search all
        if document_ids is None:
            document_ids = list(self.vector_stores.keys())
        
        # Search each document
        for doc_id in document_ids:
            if doc_id in self.vector_stores:
                contexts = await self.vector_stores[doc_id].similarity_search(query, k)
                all_contexts.extend(contexts)
        
        # Sort by relevance score and return top k
        all_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_contexts[:k]
    
    async def _get_vector_store(self, document_id: str) -> VectorStore:
        """Get or create vector store for document"""
        if document_id not in self.vector_stores:
            embed_dir = Path(self.config.get('storage.embed_root', './embeddings')) / document_id
            vector_store = VectorStore(embed_dir, self.embedding_function)
            await vector_store.initialize()
            self.vector_stores[document_id] = vector_store
        
        return self.vector_stores[document_id]
    
    async def get_document_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all documents"""
        stats = {}
        for doc_id, store in self.vector_stores.items():
            stats[doc_id] = {
                'chunk_count': store.get_document_count(),
                'store_path': str(store.persist_directory)
            }
        return stats


class HybridRetriever:
    """Hybrid retriever combining different search strategies"""
    
    def __init__(self, rag_pipeline: ModernRAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    async def retrieve(
        self, 
        query: str, 
        document_ids: List[str] = None,
        strategies: List[str] = None,
        k: int = 5
    ) -> List[RetrievalContext]:
        """Retrieve using multiple strategies"""
        if strategies is None:
            strategies = ['semantic', 'keyword']
        
        all_contexts = []
        
        # Semantic search
        if 'semantic' in strategies:
            semantic_contexts = await self.rag_pipeline.query_documents(query, document_ids, k)
            all_contexts.extend(semantic_contexts)
        
        # Keyword search (simplified - could use BM25 or similar)
        if 'keyword' in strategies:
            keyword_contexts = await self._keyword_search(query, document_ids, k)
            all_contexts.extend(keyword_contexts)
        
        # Remove duplicates and re-rank
        unique_contexts = self._deduplicate_contexts(all_contexts)
        return self._rerank_contexts(unique_contexts, query)[:k]
    
    async def _keyword_search(self, query: str, document_ids: List[str], k: int) -> List[RetrievalContext]:
        """Simple keyword-based search"""
        # This is a simplified implementation
        # In practice, you'd want to use BM25 or similar
        keywords = query.lower().split()
        contexts = []
        
        for doc_id in (document_ids or self.rag_pipeline.vector_stores.keys()):
            if doc_id in self.rag_pipeline.vector_stores:
                # This would need to be implemented based on your specific needs
                pass
        
        return contexts
    
    def _deduplicate_contexts(self, contexts: List[RetrievalContext]) -> List[RetrievalContext]:
        """Remove duplicate contexts"""
        seen = set()
        unique_contexts = []
        
        for context in contexts:
            # Create a simple hash based on content
            content_hash = hash(context.content[:100])  # Use first 100 chars
            if content_hash not in seen:
                seen.add(content_hash)
                unique_contexts.append(context)
        
        return unique_contexts
    
    def _rerank_contexts(self, contexts: List[RetrievalContext], query: str) -> List[RetrievalContext]:
        """Re-rank contexts (simplified implementation)"""
        # This could use more sophisticated re-ranking models
        return sorted(contexts, key=lambda x: x.relevance_score, reverse=True)