import os
from os.path import join
import arxiv
import textwrap
import questionary
import requests
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader, BSHTMLLoader, UnstructuredURLLoader
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.callbacks import get_openai_callback
except:
    from langchain.document_loaders import PDFMinerLoader, PyPDFLoader, BSHTMLLoader, UnstructuredURLLoader # for loading the pdf
    from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
    from langchain.vectorstores import Chroma  # for the vectorization part
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks import get_openai_callback
from notion_tools import QA_notion_blocks, clean_metadata, print_entries, save_qa_history, load_qa_history, print_qa_result


def print_arxiv_entry(paper: arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    print(f"[{arxiv_id}] {title}")
    print("Authors:", ", ".join(authors))
    print("Published:", pubyear.date().isoformat())
    print("Abstract:")
    print(textwrap.fill(abstract, width=100))
    print("comments:", paper.comment)
    print("URL:", abs_url)


def fetch_K_results(search_obj, K=10, offset=0):
    """Fetches K results from the search object, starting from offset, and returns a list of results."""
    results = []
    try:
        for entry in search_obj.results(offset=offset):
            results.append(entry)
            if len(results) >= K:
                break
    except StopIteration:
        pass
    return results


def blocks2text(blocks):
    if "results" in blocks:
        blocks = blocks["results"]
    for block in blocks:
        if block["type"] == "paragraph":
            for parts in block["paragraph"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "heading_2":
            for parts in block["heading_2"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "quote":
            for parts in block["quote"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))
        else:
            print(block["type"])


def arxiv_entry2page_blocks(paper: arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    page_prop = {
        'Name': {
            "title": [
                {
                    "text": {
                        "content": f"[{arxiv_id}] {title}"
                    }
                }],
        },
        "Author": {
            "multi_select": [
                {'name': name} for name in authors
            ]
        },
        'Publishing/Release Date': {
            'date': {'start': pubyear.date().isoformat(), }
        },
        'Link': {
            'url': abs_url
        }
    }
    content_block = [{'quote': {"rich_text": [{"text": {"content": abstract}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Related Work"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Techniques"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     ]
    return page_prop, content_block


def arxiv_entry2page(notion_client, database_id, paper: arxiv.Result):
    """Creates a new page in the Notion database with the arxiv entry. Returns the page_id and page."""
    page_prop, content_block = arxiv_entry2page_blocks(paper)
    new_page = notion_client.pages.create(parent={"database_id": database_id}, properties=page_prop)
    notion_client.blocks.children.append(new_page["id"], children=content_block)
    return new_page["id"], new_page


def add_to_notion(notion_client, database_id, paper: arxiv.Result, print_existing=False):
    """Higher level function to add the arxiv entry to the Notion database.
    
    If the entry already exists, it will skip adding the entry and return the page_id and page.
    If the entry does not exist, it will create a new page and return the page_id and page.
    """
    title = paper.title
    arxiv_id = paper.entry_id.split("/")[-1]
    # check if entry already exists in Notion database
    results_notion = notion_client.databases.query(database_id=database_id,
                                            filter={"property": "Link", "url": {"contains": arxiv_id}})
    if len(results_notion["results"]) == 0:
        # page does not exist, create a new page
        print(f"Adding entry paper {arxiv_id}: {title}")
        page_id, page = arxiv_entry2page(notion_client, database_id, paper)
        print(f"Added entry {page_id} for arxiv paper {arxiv_id}: {title}")
        print_entries([page], print_prop=("url",))
        return page_id, page
    else:
        # page already exists, ask user if they want to update the page
        print_entries(results_notion, print_prop=("url",))
        print("Entry already exists as above. ")
        if print_existing:
            # print the existing pages
            for page in results_notion["results"]:
                print_entries([page], print_prop=("url",))
                try:
                    blocks = notion_client.blocks.children.list(page["id"])
                    blocks2text(blocks)
                except Exception as e:
                    print(e)
        if len(results_notion["results"]) == 1:
            page_id, page = results_notion["results"][0]["id"], results_notion["results"][0]
            #TODO: update page with entry
            return page_id, page
        else:
            page_id = questionary.select("Select paper:",
                         choices=[page["id"] for page in results_notion["results"]]).ask()
            page = [page for page in results_notion["results"] if page["id"] == page_id][0]
            #TODO: update page with entry
            return page_id, page


def arxiv_paper_download(arxiv_id, pdf_download_root="", text_splitter=None):
    """Downloads the arxiv paper with the given arxiv_id, and returns the path to the downloaded pdf file."""
    ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}" # older ar5iv page
    arxiv_html_url = f"https://arxiv.org/html/{arxiv_id}"  # newer arxiv page, after 2024
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # try getting ar5iv page first
    for url in [ar5iv_url, arxiv_html_url]:
        r = requests.get(url, allow_redirects=True, )
        if r.url.startswith(url.rsplit('/', 1)[0]):
            # if not redirected, then ar5iv page exists
            # then download html to parse
            print(f"Downloading {r.url}...")
            open(join(pdf_download_root, f"{arxiv_id}.html"), 'wb').write(r.content)
            print("Saved to", join(pdf_download_root, f"{arxiv_id}.html"))
            loader = BSHTMLLoader(join(pdf_download_root, f"{arxiv_id}.html"),
                                  open_encoding="utf8", bs_kwargs={"features": "html.parser"})
            pages = loader.load_and_split(text_splitter=text_splitter)
            return pages

    # if redirected, then ar5iv page does not exist, save pdf instead
    print(f"redirected to {r.url}")
    print("ar5iv not found, downloading pdf instead ")
    r = requests.get(pdf_url, allow_redirects=True, )
    open(join(pdf_download_root, f"{arxiv_id}.pdf"), 'wb').write(r.content)
    print("Saved to", join(pdf_download_root, f"{arxiv_id}.pdf"))
    loader = PyPDFLoader(join(pdf_download_root, f"{arxiv_id}.pdf"))
    # loader = PDFMinerLoader(pdf_path)
    pages = loader.load_and_split(text_splitter=text_splitter)
    return pages


def notion_paper_chat(arxiv_id, pages=None, notion_client=None, save_page_id=None, 
                      embed_rootdir="", pdf_download_rootdir="", chatsession=None):
    # TODO: add the default rootdir 
    if save_page_id is None or notion_client is None:
        print("No page id provided, no chat history will be saved to Notion.")

    if pages is None:
        print("No pages provided, downloading paper from arxiv...")
        pages = arxiv_paper_download(arxiv_id, pdf_download_root=pdf_download_rootdir)

    # create embedding directory
    embed_persist_dir = join(embed_rootdir, arxiv_id)
    qa_path = embed_persist_dir + "_qa_history"
    os.makedirs(qa_path, exist_ok=True)
    # create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", disallowed_special=()) # "text-embedding-3-small" is not found yet. 
    if os.path.exists(embed_persist_dir):
        print("Loading embeddings from", embed_persist_dir)
        vectordb = Chroma(persist_directory=embed_persist_dir, embedding_function=embeddings)
        if vectordb._collection.count() == 0:
            print("No embeddings loaded, creating new embeddings...")
            vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                             persist_directory=embed_persist_dir, )
            vectordb.persist()
    else:
        print("Creating embeddings and saving to", embed_persist_dir)
        vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                         persist_directory=embed_persist_dir, )
        vectordb.persist()
    print(f"Embeddings created. {vectordb._collection.count()} vectors loaded.")
    if os.path.exists(qa_path):
        print("Loading Q&A history from", qa_path)
        chat_history, queries, results = load_qa_history(qa_path)
        while True:
            question = questionary.select("Select Q&A history:", choices=["New query"] + queries,
                                          default="New query").ask()
            if question == "New query":
                break
            else:
                print("Q:", question)
                result = results[queries.index(question)]
                print_qa_result(result, )
    model_version = questionary.select("Select ChatGPT Model", 
                                       choices=["gpt-3.5-turbo", 
                                                "gpt-4-turbo-preview"], 
                                       default="gpt-3.5-turbo").ask()
    chat_temperature = questionary.text("Sampling temperature for ChatGPT?", default="0.3").ask()
    chat_temperature = float(chat_temperature)
    # ref_maxlen = questionary.text("Max length of reference document?", default="300").ask()
    ref_maxlen = 200
    pdf_qa_new = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=chat_temperature, model_name=model_version),
        vectordb.as_retriever(), return_source_documents=True, max_tokens_limit=2500)
    # max_tokens_limit is the max token limit for the sum of all retrieved documents

    # Q&A loop with ChatOpenAI
    with get_openai_callback() as cb:
        while True:
            try:
                if chatsession is None:
                    # no prompt session provided, fall back to questionary, 
                    # no history will be saved in this case
                    query = questionary.text("Question: ", multiline=True).ask()
                else:
                    query = chatsession.prompt("Question: ", multiline=False)
                # query = "For robotics purpose, which algorithm did they used, PPO, Q-learning, etc.?"
                if query == "" or query is None:
                    if questionary.confirm("Exit?").ask():
                        break
                    else:
                        continue

                result = pdf_qa_new({"question": query, "chat_history": ""})

                print_qa_result(result)
                # local save qa history
                save_qa_history(query, result, qa_path)
                # save to notion
                if save_page_id is not None and notion_client is not None:
                    answer = result["answer"]
                    refdocs = result['source_documents']
                    refstrs = [str(refdoc.metadata) + refdoc.page_content[:ref_maxlen] for refdoc in refdocs]
                    try:
                        notion_client.blocks.children.append(save_page_id, children=QA_notion_blocks(query, answer, refstrs))
                    except Exception as e:
                        print("Failed to save to notion")
                        print(e)
                        refstrs_meta = [str(refdoc.metadata) for refdoc in refdocs]
                        notion_client.blocks.children.append(save_page_id, children=QA_notion_blocks(query, answer, refstrs_meta))
            except KeyboardInterrupt:
                break
        # End of chat loop
        print(f"Finish conversation")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

