# NotionArxivChat-CLI
A few CLI tools to immersively, browse, read Arxiv paper, save to notion, chat with them and save the chat history

![](media/arxiv_chat_demo.gif)

Resulting Notion database:
![](media/Notion_record.png)

## Usage
* `notion_arxiv_browse.py` - browse Arxiv papers in a terminal, save to Notion database (if notion API key is provided)
* `notion_arxiv_browse_chat.py` - more powerful version of `notion_arxiv_browse` that allows you to download and chat with the papers and save the chat history to Notion database (if notion API key is provided)

## Requirements

- Python 3.6+
- If you want to save to Notion database, you need:
  - [Notion](https://www.notion.so/) account, API key
- If you want to chat with the papers, you need:
  - OpenAI API key
