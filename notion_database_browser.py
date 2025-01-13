
import yaml
import os
import arxiv
from notion_client import Client
from arxiv_browse_lib import fetch_K_results, add_to_notion, notion_paper_chat
from arxiv_browse_lib import print_arxiv_entry, arxiv_paper_download
import questionary
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory


with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

notion_client = Client(auth=os.environ["NOTION_TOKEN"])
database_id = config["database_id"]
MAX_RESULTS_PER_PAGE = int(config["MAX_RESULTS_PER_PAGE"])
PDF_DOWNLOAD_ROOT = config["PDF_DOWNLOAD_ROOT"]
EMBED_ROOTDIR = config["EMBED_ROOTDIR"]
chathistory = FileHistory("qa_chat_history.txt")
chatsession = PromptSession(history=chathistory)

last_selection = None
cnt = 0
results_notion = notion_client.databases.query(database_id=database_id,
                                            filter={"property": "Link", "url": {"contains": "arxiv"}},
                                            sorts=[{"property": "Created Time", "direction": "descending"}],
                                            page_size=25, start_cursor=None)
if results_notion.get("has_more"):
    next_cursor = results_notion["next_cursor"]
else:
    next_cursor = None

print("Most recent Notion paper database entries:")
while True:
    title_list = []
    arxiv_id_list = []
    for result_i in range(len(results_notion['results'])):
        result_page = results_notion['results'][result_i]
        page_id = result_page["id"]
        title = result_page["properties"]["Name"]["title"][0]["plain_text"]
        url = result_page["properties"]["Link"]["url"]
        arxiv_id = url.split("/")[-1]
        arxiv_id = arxiv_id.split("v")[0]
        title_list.append(title)
        arxiv_id_list.append(arxiv_id)

    choices = [f"{i + 1}: {title} " for i, title in enumerate(title_list)]
    if len(title_list) == MAX_RESULTS_PER_PAGE:
        choices.append("0: Next page")
    if cnt > 0:
        choices.append("-1: Prev page")
    selection = questionary.select("Select paper:", choices=choices, default=None if last_selection is None
                                    else choices[last_selection]).ask()
    selection = int(selection.split(":")[0])
    if selection == 0:
        cnt += MAX_RESULTS_PER_PAGE
        results_notion = notion_client.databases.query(database_id=database_id,
                                            filter={"property": "Link", "url": {"contains": "arxiv"}},
                                            sorts=[{"property": "Created Time", "direction": "descending"}],
                                            page_size=25, start_cursor=next_cursor)
        next_cursor = results_notion["next_cursor"]
        continue
    elif selection == -1:
        cnt -= MAX_RESULTS_PER_PAGE
        results_notion = notion_client.databases.query(database_id=database_id,
                                            filter={"property": "Link", "url": {"contains": "arxiv"}},
                                            sorts=[{"property": "Created Time", "direction": "descending"}],
                                            page_size=25, start_cursor=None)
        next_cursor = results_notion["next_cursor"]
        continue
    else:
        last_selection = int(selection) - 1
        notion_page = results_notion['results'][int(selection) - 1]
        page_id = notion_page["id"]
        title = notion_page["properties"]["Name"]["title"][0]["plain_text"]
        url = notion_page["properties"]["Link"]["url"]
        arxiv_id = url.split("/")[-1]
        arxiv_id = arxiv_id.split("v")[0]
        search_obj = arxiv.Search(id_list=[arxiv_id], max_results=1)
        results_arxiv = fetch_K_results(search_obj, K=1, offset=0)
        if len(results_arxiv) == 0:
            print(f"No paper found for {arxiv_id}")
            continue
        paper = results_arxiv[0]
        print_arxiv_entry(paper)
        if questionary.confirm("Add this entry?").ask():
            # Add the entry if confirmed
            page_id, _ = add_to_notion(notion_client, database_id, paper)
            if questionary.confirm("Q&A Chatting with this file?").ask():
                pages = arxiv_paper_download(arxiv_id, pdf_download_root=PDF_DOWNLOAD_ROOT)
                notion_paper_chat(arxiv_id=arxiv_id, pages=pages, save_page_id=page_id,
                                    notion_client=notion_client, embed_rootdir=EMBED_ROOTDIR,
                                    chatsession=chatsession, )