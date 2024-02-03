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
from sklearn.neighbors import NearestNeighbors
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
database_name = questionary.select("Select database to browse:", choices=list(database_catalog.keys())+["All"]).ask()
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
# 
knn = NearestNeighbors(n_neighbors=25, metric="cosine")
knn.fit(embed_arr)
#%%
with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

MAX_RESULTS = int(config["MAX_RESULTS_PER_PAGE"]) # 25
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

def fetch_K_vector_neighbor(cossim, paper_collection, K=10, offset=0):
    sort_idx = np.argsort(cossim)
    sort_idx = sort_idx[::-1]
    sort_idx = sort_idx[offset:offset+K]
    return sort_idx, [paper_collection[idx] for idx in sort_idx]


cur_anchor_idx = None
cossim_vec = None
cur_idx_list = np.random.randint(0, embed_arr.shape[0], (MAX_RESULTS))
results_arxiv = [paper_collection[idx] for idx in cur_idx_list]
while True:
    try:
        # results_arxiv = fetch_K_vector_neighbor(cossim, paper_collection, K=MAX_RESULTS, offset=offset_cur)
        last_selection = None  # last selected result to highlight
        offset_cur = 0
        # while True:
        # looping of results and pages, navigating through search results
        if cur_anchor_idx is None:
            print("Shuffled Recommendations: ")
            choices = [f"{i + 1}: [{paper.entry_id.split('/')[-1]}] {paper.title} "
                       for i, paper in enumerate(results_arxiv)]
        else:
            anchor_paper = paper_collection[cur_anchor_idx]
            print(f"Anchor paper: [{anchor_paper.entry_id.split('/')[-1]}] {anchor_paper.title}")
            choices = [f"{i + 1}: (Cos: {cossim_vec[idx]:.3f}) [{paper.entry_id.split('/')[-1]}] {paper.title} "
                       for i, (idx, paper) in enumerate(zip(cur_idx_list, results_arxiv))]
            if len(results_arxiv) == MAX_RESULTS:
                choices.append("0: Next page")
            if offset_cur > 0:
                choices.append("-1: Prev page")
        choices.append("-2: Randomize")
        choices.append("-3: Exit")
        selection = (questionary.select("Select paper:", choices=choices,
               default=None if last_selection is None
                            else choices[last_selection]).
               ask())
        selection = int(selection.split(":")[0])
        if selection == 0:
            offset_cur += MAX_RESULTS
            cur_idx_list, results_arxiv = fetch_K_vector_neighbor(cossim_vec, paper_collection, K=MAX_RESULTS, offset=offset_cur)
            continue
        if selection == -1:
            offset_cur -= MAX_RESULTS
            cur_idx_list, results_arxiv = fetch_K_vector_neighbor(cossim_vec, paper_collection, K=MAX_RESULTS, offset=offset_cur)
            continue
        if selection == -2:
            cur_anchor_idx = None
            cossim_vec = None
            cur_idx_list = np.random.randint(0, embed_arr.shape[0], (MAX_RESULTS))
            results_arxiv = [paper_collection[idx] for idx in cur_idx_list]
            continue
        if selection == -3:
            raise KeyboardInterrupt
        else:
            paper_global_idx = cur_idx_list[int(selection) - 1]
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

            if questionary.confirm("Explore kNN of this paper?").ask():
                cur_anchor_idx = paper_global_idx
                # Add the entry if confirmed\
                query_embed = embed_arr[cur_anchor_idx]
                sim = embed_arr @ query_embed
                cossim_vec = (sim / np.linalg.norm(embed_arr, axis=1)
                          / np.linalg.norm(query_embed))
                # cur_idx_list = cossim_vec.argsort()[::-1][:MAX_RESULTS]
                offset_cur = 0
                cur_idx_list, results_arxiv = fetch_K_vector_neighbor(cossim_vec, paper_collection,
                                            K=MAX_RESULTS, offset=offset_cur)

    except KeyboardInterrupt as e:
        break
    except Exception as e:
        print("Chat loop failed with exception:")
        print(e)
        continue

#%%