# NotionArxivChat-CLI

A few CLI tools to immersively, browse, read Arxiv paper, save to notion, chat with them and save the chat history

![](media/arxiv_chat_demo.gif)

Resulting Notion database:
![](media/Notion_record.png)

## Updates 
* Feb.1st, 2024 updates. 
  * Add functionality to fetch and build arxiv abstract database `upgrade_arxiv_database.py`
  * Support semantic vector search for arxiv abstracts and chat `notion_arxiv_vector_search.py`
  * Notion saving failure bug fixed. 
* Mar. 2nd 2024 updates. 
  * Add pure Arxiv keyword paper scrapting CLI
* July 22nd 2025 updates. 
  * Add integration to Zotero, enabling direct adding arxiv papers to Zotero via `pyzotero`. 
  * Add collection selection functionality in CLI. 

## Usage
* `arxiv_browser.py` - simple CLI tool to browse Arxiv papers, search with arxiv query string; read abstracts in a terminal. 
  * Details for query string can be found [here](https://info.arxiv.org/help/api/user-manual.html#Appendices)
* `notion_arxiv_browse.py` - browse Arxiv papers in a terminal, save to Notion database (if notion API key and database id are provided)
* `notion_arxiv_browse_chat.py` - more powerful version of `notion_arxiv_browse` that allows you to download and chat with the papers and save the chat history to Notion database (if notion API key and database id are provided)

## Configuration

In the `config.yaml` file, you can configure the following parameters:
* `MAX_RESULTS_PER_PAGE` - number of results per page when browsing Arxiv papers
* `database_id` - Notion database id. Set this id when you want to save the papers to Notion database. Default is `"PUT_YOUR_DATABASE_ID_HERE"` (string) "None" or "" (empty string) will disable saving to Notion database.
* `PDF_DOWNLOAD_ROOT`: path to the root directory where the PDFs will be downloaded. Default is `'./pdfs'`
* `EMBED_ROOTDIR`: path to the root directory where the embeddings will be saved. Default is `'./embeddings'`
  * Note for windows path please use single quote string to avoid escape character issues. e.g. `'C:/Users/username/pdf'`
  * **Pro Tips**: put the embedding and pdf in a synced folder, so you can access the embedding and chat history from other devices.   

Environment variables:
* `NOTION_TOKEN` necessary for saving to Notion database. See [Notion API](https://developers.notion.com/docs/getting-started) for more details.
* `OPENAI_API_KEY` necessary for chatting with the papers. See [OpenAI API](https://platform.openai.com/account/api-keys) for more details.

## Requirements

- Python 3.6+
- pip install "pydantic<2"
- If you want to save to Notion database, you need:
  - [Notion](https://www.notion.so/) account, API key
- If you want to chat with the papers, you need:
  - OpenAI API key
