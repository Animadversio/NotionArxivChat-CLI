#%%
"""
Pipeline for preparing the arxiv embedding database
"""
import os
from os.path import join
import arxiv
import openai
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

#%
# class names of arxiv
# https://gist.github.com/jozefg/c2542f51a0b9b3f6efe528fcec90e334
CS_CLASSES = [
    'cs.' + cat for cat in [
        'AI', 'AR', 'CC', 'CE', 'CG', 'CL', 'CR', 'CV', 'CY', 'DB',
        'DC', 'DL', 'DM', 'DS', 'ET', 'FL', 'GL', 'GR', 'GT', 'HC',
        'IR', 'IT', 'LG', 'LO', 'MA', 'MM', 'MS', 'NA', 'NE', 'NI',
        'OH', 'OS', 'PF', 'PL', 'RO', 'SC', 'SD', 'SE', 'SI', 'SY',
    ]
]

MATH_CLASSES = [
    'math.' + cat for cat in [
        'AC', 'AG', 'AP', 'AT', 'CA', 'CO', 'CT', 'CV', 'DG', 'DS',
        'FA', 'GM', 'GN', 'GR', 'GT', 'HO', 'IT', 'KT', 'LO',
        'MG', 'MP', 'NA', 'NT', 'OA', 'OC', 'PR', 'QA', 'RA',
        'RT', 'SG', 'SP', 'ST', 'math-ph'
    ]
]

QBIO_CLASSES = [
    'q-bio.' + cat for cat in [
        'BM', 'CB', 'GN', 'MN', 'NC', 'OT', 'PE', 'QM', 'SC', 'TO'
    ]
]

# Which categories do we search
CLASSES = CS_CLASSES + MATH_CLASSES + QBIO_CLASSES

ABSTR_EMBED_DIR = "/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/openai-emb-database/Embed_arxiv_abstr"

def prepare_arxiv_embedding_database(database_name, search_query, abstr_embed_dir, 
                max_paper_num=20000, embed_batch_size=100, print_details=False,
                no_embed=False):
    """
    Prepares the ArXiv embedding database by fetching papers based on a search query, 
    embedding the abstracts, and saving the results.

    Parameters:
    - database_name: Name of the database for saving.
    - search_query: ArXiv search query for fetching papers.
    - abstr_embed_dir: Directory for saving the database and embeddings.
    - max_paper_num: Maximum number of papers to fetch.
    - embed_batch_size: Batch size for generating embeddings.
    - print_details: If True, print detailed information during processing.
    """
    # Ensure the directory for saving the database exists
    if not os.path.exists(abstr_embed_dir):
        os.makedirs(abstr_embed_dir)

    # Initialize or load the database and embedding array
    paper_collection, embedding_arr = initialize_or_load_database(abstr_embed_dir, database_name)
    arxiv_indices = [paper.entry_id.strip("http://arxiv.org/abs/") for paper in paper_collection]
    
    # Fetch papers based on the search query
    update_paper_collection = fetch_papers(search_query, max_paper_num, arxiv_indices, print_details=print_details)

    paper_collection = paper_collection + update_paper_collection
    pkl.dump(paper_collection, open(join(abstr_embed_dir, f"arxiv_collection_{database_name}.pkl"), "wb"))
    df = pd.DataFrame(paper_collection)
    df.to_csv(join(abstr_embed_dir, f"arxiv_collection_{database_name}.csv"))
    #%%
    paper_time_histogram(paper_collection, database_name, search_query, abstr_embed_dir,
                        save_suffix="", time_limit=(None, None), bins=50)
    paper_time_histogram(paper_collection, database_name, search_query, abstr_embed_dir,
                        time_limit=(datetime.datetime(2017, 1, 1), datetime.datetime.today()), 
                        save_suffix="_recent2017", bins=200)
    paper_time_histogram(paper_collection, database_name, search_query, abstr_embed_dir,
                        time_limit=(datetime.datetime(2020, 1, 1), datetime.datetime.today()), 
                        save_suffix="_recent2020", bins=200)
    paper_time_histogram(paper_collection, database_name, search_query, abstr_embed_dir,
                        time_limit=(datetime.datetime(2022, 1, 1), datetime.datetime.today()), 
                        save_suffix="_recent2022", bins=200)
    if no_embed:
        print(f"Database {database_name} updated with {len(update_paper_collection)} papers and saved.")
        return paper_collection, embedding_arr
    # Input continue
    input("Press Enter to continue embedding the updated files...")
    # # Generate embeddings for the updated collection
    update_embedding_arr = generate_embeddings(update_paper_collection, embed_batch_size)
    # if there is any new entries, then update the embedding_arr
    if len(update_paper_collection) > 0:
        if embedding_arr is not None:
            embedding_arr = np.concatenate([embedding_arr, update_embedding_arr], axis=0)
        else:
            # if the embedding_arr is None, then write the update_embedding_arr to the embedding_arr
            embedding_arr = update_embedding_arr
        
    if not len(paper_collection) == len(embedding_arr):
        print("Warning: The number of papers and embeddings do not match!!!")
    # Save the updated embeddings and paper collection
    pkl.dump([embedding_arr, paper_collection],
            open(join(abstr_embed_dir, f"arxiv_embedding_arr_{database_name}.pkl"), "wb"))
    print(f"Database {database_name} updated with {len(update_paper_collection)} papers and saved.")
    print(f"Total {len(paper_collection)} papers with embedding shape {embedding_arr.shape} saved.")
    return paper_collection, embedding_arr
    
    # # Initialize or load the database and embedding array
    # paper_collection, embedding_arr = initialize_or_load_database(abstr_embed_dir, database_name)

    # # Fetch papers based on the search query
    # update_paper_collection = fetch_papers(search_query, max_paper_num, paper_collection)

    # # Update the paper collection
    # paper_collection.extend(update_paper_collection)

    # # Save the updated paper collection
    # save_paper_collection(abstr_embed_dir, database_name, paper_collection)

    # # Generate embeddings for the updated collection
    # update_embedding_arr = generate_embeddings(update_paper_collection, embed_batch_size)

    # # Concatenate the new embeddings with the existing ones
    # if embedding_arr is not None and len(update_embedding_arr) > 0:
    #     embedding_arr = np.concatenate([embedding_arr, update_embedding_arr], axis=0)

    # # Save the updated embeddings and paper collection
    # save_embeddings_and_collection(abstr_embed_dir, database_name, embedding_arr, paper_collection)

    # # Optionally, generate and save histograms of publication times
    # generate_publication_time_histograms(paper_collection, database_name, search_query, abstr_embed_dir)

    # return paper_collection, embedding_arr


def entry2string(paper):
    id_pure = paper.entry_id.strip("http://arxiv.org/abs/")
    return f"[{id_pure}] Title: {paper.title}\nAbstract: {paper.summary}\nDate: {paper.published}"


def paper_time_histogram(paper_collection, database_name, search_query, abstr_embed_dir,
                            save_suffix="", time_limit=(None, None), bins=50):
    time_col = [paper.published for paper in paper_collection]
    # filter the time based on the time limit
    if time_limit[0] is not None:
        time_col = [time for time in time_col if time.replace(tzinfo=None) > time_limit[0]]
    if time_limit[1] is not None:
        time_col = [time for time in time_col if time.replace(tzinfo=None) < time_limit[1]]
    figh = plt.figure()
    plt.hist(time_col, bins=bins)
    plt.title("Distribution of publication time")
    plt.title(f"Publication time distribution for {database_name}\n{search_query}")
    plt.ylabel("count")
    plt.xlabel("time")
    plt.xlim(time_limit)
    plt.savefig(join(abstr_embed_dir, f"arxiv_time_dist_{database_name}{save_suffix}.png"))
    plt.show()
    return figh


def initialize_or_load_database(abstr_embed_dir, database_name):
    # Function to initialize or load the database and embedding array
    # Returns: Initialized or loaded paper_collection and embedding_arr
    # Initialize or load the database and embedding array
    if os.path.exists(join(abstr_embed_dir, f"arxiv_embedding_arr_{database_name}.pkl")):
        print(f"Database {database_name} already exists, loading the paper and embedding array")
        embedding_arr, paper_collection = pkl.load(open(join(abstr_embed_dir, f"arxiv_embedding_arr_{database_name}.pkl"), "rb"))  
        print(f"{len(paper_collection)} papers with embedding shape {embedding_arr.shape} loaded.")
    elif os.path.exists(join(abstr_embed_dir, f"arxiv_collection_{database_name}.pkl")):
        print(f"Database {database_name} already exists, loading the database")
        print("Embedding array does not exist, start fetching papers")
        paper_collection = pkl.load(open(join(abstr_embed_dir, f"arxiv_collection_{database_name}.pkl"), "rb"))
        embedding_arr = None
        print(f"{len(paper_collection)} papers with loaded. No embedding found")
    else:
        print(f"Database {database_name} does not exist, start fetching papers")
        paper_collection = []
        embedding_arr = None
    return paper_collection, embedding_arr


def fetch_papers(search_query, max_paper_num, existing_indices=(), print_details=False):
    # Function to fetch papers based on the search_query
    # Returns: List of fetched papers not already in existing_papers
    # Fetch papers based on the search query
    search = arxiv.Search(
        query=search_query,
        max_results=max_paper_num,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    # Print titles and abstracts of the latest papers
    idx = 0
    update_paper_collection = []
    for paper in arxiv.Client(page_size=100, delay_seconds=5.0, num_retries=50).results(search):
        id_pure = paper.entry_id.strip("http://arxiv.org/abs/")
        if id_pure in existing_indices:
            print(f"Skip [{id_pure}] ({paper.published.date()}) already in database, stop upgrade.",)
            print(f"{idx} number of papers added to the database.")
            break
        update_paper_collection.append(paper)
        print(f"{idx} [{id_pure}] ({paper.published.date()})",
            paper.title)
        idx += 1
        if print_details:
            print("Abstract:", paper.summary)
            print("Categories:", paper.categories, end=" ")
            print("ID:", paper.entry_id, end=" ")
            print("-" * 80)
    return update_paper_collection


def save_paper_collection(abstr_embed_dir, database_name, paper_collection):
    # Function to save the paper collection to disk
    pkl.dump(paper_collection, open(join(abstr_embed_dir, f"arxiv_collection_{database_name}.pkl"), "wb"))
    df = pd.DataFrame(paper_collection)
    df.to_csv(join(abstr_embed_dir, f"arxiv_collection_{database_name}.csv"))


def generate_embeddings(paper_collection, batch_size):
    # Function to generate embeddings for the given paper_collection
    # Returns: Numpy array of embeddings
    update_embedstr_col = [entry2string(paper) for paper in paper_collection]
    update_embedding_col = []
    client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    for i in tqdm(range(0, len(update_embedstr_col), batch_size)):
        embedstr_batch = update_embedstr_col[i:i + batch_size]
        response = client.embeddings.create(
            input=embedstr_batch,
            model="text-embedding-ada-002"
        )
        update_embedding_col.extend(response.data)

    if len(update_embedding_col) > 0:
        # format as array
        update_embedding_arr = np.stack([embed.embedding for embed in update_embedding_col])
    else:
        update_embedding_arr = np.empty((0,0),) # TODO: add the 2nd dimension. 
    return update_embedding_arr
    

def save_embeddings_and_collection(abstr_embed_dir, database_name, embedding_arr, paper_collection):
    # Function to save embeddings and paper collection to disk
    pkl.dump([embedding_arr, paper_collection],
            open(join(abstr_embed_dir, f"arxiv_embedding_arr_{database_name}.pkl"), "wb"))
    print(f"Total {len(paper_collection)} papers with embedding shape {embedding_arr.shape} saved.")
    

def generate_publication_time_histograms(paper_collection, database_name, search_query, abstr_embed_dir):
    # Function to generate and save histograms of publication times
    paper_time_histogram(paper_collection, database_name, search_query, 
                        save_suffix="", time_limit=(None, None), bins=50)
    paper_time_histogram(paper_collection, database_name, search_query, 
                        time_limit=(datetime.datetime(2017, 1, 1), datetime.datetime.today()), 
                        save_suffix="_recent2017", bins=200)
    paper_time_histogram(paper_collection, database_name, search_query, 
                        time_limit=(datetime.datetime(2020, 1, 1), datetime.datetime.today()), 
                        save_suffix="_recent2020", bins=200)
    paper_time_histogram(paper_collection, database_name, search_query, 
                        time_limit=(datetime.datetime(2022, 1, 1), datetime.datetime.today()), 
                        save_suffix="_recent2022", bins=200)


def main():
    import argparse
    query_base = {"diffusion_10k": 'cat:cs.* AND all:diffusion OR all:score-based',
                  "GAN_6k": 'cat:cs.* AND all:"generative adversarial network" OR all:GAN',
                  "VAE_2k": 'cat:cs.* AND all:"variational autoencoder" OR all:VAE',
                  "flow_100": 'cat:cs.* AND all:"flow matching"',
                  "normflow_800": 'cat:cs.* AND all:"normalizing flow"',
                  "LLM_18k": 'cat:cs.* AND all:"language model" OR all:LLM'}
    
    parser = argparse.ArgumentParser(description="Prepare the ArXiv Embedding Database", 
                        epilog=f'Existing databases names {[*query_base.keys()]}')
    parser.add_argument("--database_name", type=str, required=True, help="Name of the database")
    parser.add_argument("--search_query", type=str, default="", help="Search query for fetching papers")
    parser.add_argument("--max_paper_num", type=int, default=20000, help="Maximum number of papers to fetch")
    parser.add_argument("--no_embed", action='store_true', help="Do not embed the papers")
    parser.add_argument("--embed_batch_size", type=int, default=100, help="Batch size for embedding papers")
    parser.add_argument("--print_details", action='store_true', help="Print detailed information during processing")
    args = parser.parse_args()
    plt.switch_backend('agg')
    # set the directory for saving the database
    # Name for saving the database
    if args.database_name == "all":
        print("Upgrading all the arxiv databases... ")
        print("The databases are: ",)
        for name, query in query_base.items():
            print(name, ":", query)
        print("\nStart Fetching all the databases... ")
        for name, query in query_base.items():
            prepare_arxiv_embedding_database(database_name=name,
                                     search_query=query,
                                     abstr_embed_dir=ABSTR_EMBED_DIR,
                                     max_paper_num=args.max_paper_num,
                                     embed_batch_size=args.embed_batch_size,
                                     print_details=args.print_details)
        print("All the databases are fetched and saved.")
        return
    elif args.search_query == "" and args.database_name not in query_base:
        print("Error: The database_name is not in the default list!!! and search_query is not provided.")
        return
    elif args.search_query == "" and args.database_name in query_base:
        print(f"Using the default search query for {args.database_name}: {query_base[args.database_name]}")
        args.search_query = query_base[args.database_name]
    
    # TODO: update the query_base when new database is created
    # assume we are using CLI, then use AGG backend
    prepare_arxiv_embedding_database(database_name=args.database_name,
                                     search_query=args.search_query,
                                     abstr_embed_dir=ABSTR_EMBED_DIR,
                                     max_paper_num=args.max_paper_num,
                                     embed_batch_size=args.embed_batch_size,
                                     print_details=args.print_details,
                                     no_embed=args.no_embed)

if __name__ == "__main__":
    main()




