import os
from os.path import join
from notion_client import Client
import arxiv
import questionary
import textwrap
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory
import yaml
import requests
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from notion_tools import QA_notion_blocks, clean_metadata, print_entries, save_qa_history, load_qa_history, print_qa_result
from arxiv_browse_lib import add_to_notion, notion_paper_chat, fetch_K_results, print_arxiv_entry, arxiv_paper_download
# file to save the arxiv query history
history = FileHistory("notion_arxiv_history.txt")
session = PromptSession(history=history)
# file to save the Q&A chat history
chathistory = FileHistory("qa_chat_history.txt")
chatsession = PromptSession(history=chathistory)

with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

MAX_RESULTS_PER_PAGE = int(config["MAX_RESULTS_PER_PAGE"])
PDF_DOWNLOAD_ROOT = config["PDF_DOWNLOAD_ROOT"]
EMBED_ROOTDIR = config["EMBED_ROOTDIR"]
os.makedirs(PDF_DOWNLOAD_ROOT, exist_ok=True)
os.makedirs(EMBED_ROOTDIR, exist_ok=True)
print(f"PDFs will be downloaded to {PDF_DOWNLOAD_ROOT}")
print(f"Computed embeddings will be saved to {EMBED_ROOTDIR}")

if "NOTION_TOKEN" in os.environ:
    notion = Client(auth=os.environ["NOTION_TOKEN"])
    database_id = config["database_id"]
    # notion.databases.query(database_id, filter={"property": "Name", "text": {"is_not_empty": True}}, )
    if database_id == "PUT_YOUR_DATABASE_ID_HERE" or database_id == "" or database_id == "None":
        print("Please set the database_id in config.yaml.")
        save2notion = False
    else:
        save2notion = True
else:
    print("Please set the NOTION_TOKEN environment variable.")
    save2notion = False

default_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200)

# query = "2106.05963"
# query = "au:Yann LeCun"
# Logic:
# Ctrl-C in the navigation loop to exit and start a new query
# Ctrl-C in the query prompt to exit the program
# Up/Down to navigate through prompts and query history
# main loop
while True:
    try:
        cnt = 0
        query = session.prompt("Enter arXiv ID or query str: ", multiline=False)
        search_obj = arxiv.Search(query, )
        results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS_PER_PAGE, offset=cnt)
        if len(results_arxiv) == 0:
            print("No results found.")
            continue
        elif len(results_arxiv) == 1:
            paper = results_arxiv[0]
            arxiv_id = paper.entry_id.split("/")[-1]
            print_arxiv_entry(paper)
            # Add the entry if confirmed
            if questionary.confirm("Add this entry?").ask():
                page_id, _ = add_to_notion(notion, database_id, paper)
                if questionary.confirm("Q&A Chatting with this file?").ask():
                    pages = arxiv_paper_download(arxiv_id, pdf_download_root=PDF_DOWNLOAD_ROOT)
                    notion_paper_chat(arxiv_id=arxiv_id, pages=pages, save_page_id=page_id,
                                        notion_client=notion, embed_rootdir=EMBED_ROOTDIR,
                                        chatsession=chatsession, )
        elif len(results_arxiv) > 1:
            # multiple results found, complex logic to navigate through results
            last_selection = None  # last selected result to highlight
            while True:
                # looping of results and pages, navigating through search results
                print("Multiple results found. Please select one:")
                choices = [f"{i + 1}: [{paper.entry_id.split('/')[-1]}] {paper.title} " for i, paper in enumerate(results_arxiv)]
                if len(results_arxiv) == MAX_RESULTS_PER_PAGE:
                    choices.append("0: Next page")
                if cnt > 0:
                    choices.append("-1: Prev page")
                selection = questionary.select("Select paper:", choices=choices, default=None if last_selection is None
                                               else choices[last_selection]).ask()
                selection = int(selection.split(":")[0])
                if selection == 0:
                    cnt += MAX_RESULTS_PER_PAGE
                    results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS_PER_PAGE, offset=cnt)
                    continue
                if selection == -1:
                    cnt -= MAX_RESULTS_PER_PAGE
                    results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS_PER_PAGE, offset=cnt)
                    continue
                else:
                    paper = results_arxiv[int(selection) - 1]
                    last_selection = int(selection) - 1
                    arxiv_id = paper.entry_id.split("/")[-1]
                    print_arxiv_entry(paper)
                    if questionary.confirm("Add this entry?").ask():
                        # Add the entry if confirmed
                        page_id, _ = add_to_notion(notion, database_id, paper)
                        if questionary.confirm("Q&A Chatting with this file?").ask():
                            pages = arxiv_paper_download(arxiv_id, pdf_download_root=PDF_DOWNLOAD_ROOT)
                            notion_paper_chat(arxiv_id=arxiv_id, pages=pages, save_page_id=page_id,
                                              notion_client=notion, embed_rootdir=EMBED_ROOTDIR,
                                              chatsession=chatsession, )

    except KeyboardInterrupt as e:
        break
    except Exception as e:
        print("Chat loop failed with exception:")
        print(e)
        continue
