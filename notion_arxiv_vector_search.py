import os
from os.path import join
import yaml
import numpy as np
import pickle as pkl
import arxiv
import openai
from notion_client import Client
import questionary
import textwrap
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory
from notion_tools import print_entries
from arxiv_browse_lib import add_to_notion, notion_paper_chat, fetch_K_results, print_arxiv_entry, arxiv_paper_download

#%%
abstr_embed_dir = "/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/openai-emb-database/Embed_arxiv_abstr"
database_catalog = {# "diffusion_7k": "arxiv_embedding_arr_diffusion_7k.pkl",
                    # "LLM_5k": "arxiv_embedding_arr_LLM_5k.pkl",
                    "diffusion_10k": "arxiv_embedding_arr_diffusion_10k.pkl",
                    "LLM_18k": "arxiv_embedding_arr_LLM_18k.pkl",
                    "GAN_6k": "arxiv_embedding_arr_GAN_6k.pkl",
                    "VAE_2k": "arxiv_embedding_arr_VAE_2k.pkl",
                    "flow_100": "arxiv_embedding_arr_flow_100.pkl",
                    "normflow_800": "arxiv_embedding_arr_normflow_800.pkl",
}
# database_name = "diffusion_7k"
database_name = questionary.select("Select database to browse:", choices=["All"]+list(database_catalog.keys())).ask()
if not (database_name == "All"):
    database_file = database_catalog[database_name]
    embed_arr, paper_collection = pkl.load(open(join(abstr_embed_dir, database_file), "rb"))
else:
    embed_arr = []
    paper_collection = []
    for database_file in database_catalog.values():
        embed_arr_cur, paper_collection_cur = pkl.load(open(join(abstr_embed_dir, database_file), "rb"))
        embed_arr.append(embed_arr_cur)
        paper_collection.extend(paper_collection_cur)
    embed_arr = np.concatenate(embed_arr, axis=0)

assert embed_arr.shape[0] == len(paper_collection)
print(f"Loaded {embed_arr.shape[0]} papers from {database_name}, embed shape {embed_arr.shape}")
#%%
with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

MAX_RESULTS_PER_PAGE = int(config["MAX_RESULTS_PER_PAGE"])
PDF_DOWNLOAD_ROOT = config["PDF_DOWNLOAD_ROOT"]
EMBED_ROOTDIR = config["EMBED_ROOTDIR"]
database_id = config["database_id"]

client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
notion = Client(auth=os.environ["NOTION_TOKEN"])
history = FileHistory("notion_abstract_history.txt")
session = PromptSession(history=history)
MAX_RESULTS = 25

def fetch_K_vector_neighbor(cossim, paper_collection, K=10, offset=0):
    """Fetch K nearest neighbors from the paper collection based on the cosine similarity.
    Parameters
    cossim: np.ndarray
        The cosine similarity array between the query and the paper collection.
    paper_collection: list
        The list of paper entries.
    K: int (default 10)
        The number of nearest neighbors to fetch.
    offset: int (default 0)
        offset to start fetching the nearest neighbors.
    """
    sort_idx = np.argsort(cossim)
    sort_idx = sort_idx[::-1]
    sort_idx = sort_idx[offset:offset+K]
    return [paper_collection[idx] for idx in sort_idx]


while True:
    try:
        query = session.prompt("Enter query str to search arxiv database: ",
                               multiline=False)
        response_query = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embed = np.array(response_query.data[0].embedding)
        sim = embed_arr @ query_embed
        cossim = (sim / np.linalg.norm(embed_arr, axis=1)
                  / np.linalg.norm(query_embed))
        offset_cur = 0
        results_arxiv = fetch_K_vector_neighbor(cossim, paper_collection, K=MAX_RESULTS, offset=offset_cur)
        last_selection = None  # last selected result to highlight
        while True:
            # looping of results and pages, navigating through search results
            choices = [f"{i + 1}: [{paper.entry_id.split('/')[-1]}] {paper.title} "
                        for i, paper in enumerate(results_arxiv)]
            if len(results_arxiv) == MAX_RESULTS:
                choices.append("0: Next page")
            if offset_cur > 0:
                choices.append("-1: Prev page")
            selection = questionary.select("Select paper:", choices=choices, default=None if last_selection is None
                                           else choices[last_selection]).ask()
            selection = int(selection.split(":")[0])
            if selection == 0:
                offset_cur += MAX_RESULTS
                results_arxiv = fetch_K_vector_neighbor(cossim, paper_collection, K=MAX_RESULTS, offset=offset_cur)
                continue
            if selection == -1:
                offset_cur -= MAX_RESULTS
                results_arxiv = fetch_K_vector_neighbor(cossim, paper_collection, K=MAX_RESULTS, offset=offset_cur)
                continue
            else:
                paper = results_arxiv[int(selection) - 1]
                last_selection = int(selection) - 1
                print_arxiv_entry(paper)
                arxiv_id = paper.entry_id.split("/")[-1]
                if questionary.confirm("Add this entry?").ask():
                    # Add the entry if confirmed
                    page_id, _ = add_to_notion(notion, database_id, paper)
                    if questionary.confirm("Q&A Chatting with this file?").ask():
                        pages = arxiv_paper_download(arxiv_id, pdf_download_root=PDF_DOWNLOAD_ROOT)
                        notion_paper_chat(arxiv_id=arxiv_id, pages=pages, save_page_id=page_id,
                                            notion_client=notion, embed_rootdir=EMBED_ROOTDIR,
                                            chatsession=None, )

    except KeyboardInterrupt as e:
        break
    except Exception as e:
        print("Chat loop failed with exception:")
        print(e)
        continue

#%%