# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NotionArxivChat-CLI is a collection of Python CLI tools for browsing ArXiv papers, downloading PDFs, chatting with papers using OpenAI/LangChain, and saving results to Notion databases with optional Zotero integration.

## Core Architecture

### Main Entry Points
- `arxiv_browser.py` - Basic ArXiv paper browser with search functionality
- `notion_arxiv_browse.py` - ArXiv browser with Notion database integration for saving papers
- `notion_arxiv_browse_chat.py` - Full-featured tool with PDF download, chat capabilities, and Notion/Zotero integration
- `notion_arxiv_vector_search.py` - Semantic vector search for ArXiv abstracts
- `notion_arxiv_kNN_explorer.py` - k-NN exploration of paper embeddings
- `notion_database_browser.py` - Browse existing Notion database entries
- `upgrade_arxiv_database.py` - Build and update ArXiv abstract embedding database

### Core Libraries
- `arxiv_browse_lib.py` - Core functionality for ArXiv integration, PDF downloading, and paper chat
- `notion_tools.py` - Notion API utilities for creating pages, blocks, and managing Q&A history
- `zotero_lib.py` - Zotero integration for adding papers to reference manager

### Configuration
- `config.yaml` - Main configuration file with settings for:
  - `MAX_RESULTS_PER_PAGE` - Number of search results per page
  - `database_id` - Notion database ID for saving papers
  - `PDF_DOWNLOAD_ROOT` - Directory for downloaded PDFs
  - `EMBED_ROOTDIR` - Directory for embeddings storage

### Environment Variables
- `NOTION_TOKEN` - Required for Notion database integration
- `OPENAI_API_KEY` - Required for chat functionality
- `ZOTERO_API` - Required for Zotero integration

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Tools
```bash
# Basic ArXiv browsing
python arxiv_browser.py

# ArXiv browsing with Notion integration
python notion_arxiv_browse.py

# Full-featured paper chat with Notion/Zotero
python notion_arxiv_browse_chat.py

# Vector search in ArXiv abstracts
python notion_arxiv_vector_search.py

# Build/update ArXiv embedding database
python upgrade_arxiv_database.py
```

## Key Technical Details

### Chat System Architecture
- Uses LangChain's `ConversationalRetrievalChain` for document Q&A
- ChromaDB for vector storage of paper embeddings
- OpenAI embeddings for semantic search
- Supports both PDF and HTML document loading

### Notion Integration
- Creates structured pages with paper metadata (title, authors, abstract, etc.)
- Saves Q&A chat history as nested blocks in Notion pages
- Handles long content by splitting into multiple blocks (max 1950 chars per block)

### ArXiv Query System
- Supports ArXiv API query strings (see ArXiv API documentation)
- Pagination support for large result sets
- Persistent command history using `prompt_toolkit.FileHistory`

### File Organization
- PDFs stored in `PDF_DOWNLOAD_ROOT` directory
- Embeddings cached in `EMBED_ROOTDIR` for reuse across sessions
- Chat histories saved to local files (`notion_arxiv_history.txt`, `qa_chat_history.txt`)

### Zotero Integration
- Converts ArXiv results to Zotero items (preprint or journal article format)
- Supports collection selection and duplicate checking
- Uses PyZotero library for API interactions

## Important Dependencies
- `pydantic < 2` (version constraint required)
- LangChain ecosystem for document processing and chat
- ChromaDB for vector storage
- Notion SDK for database operations
- PyZotero for reference management