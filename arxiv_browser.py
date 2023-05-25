import os
import arxiv
import questionary
import textwrap
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory

history = FileHistory("notion_arxiv_history.txt")
session = PromptSession(history=history)
MAX_RESULTS = 35


def print_arxiv_entry(paper: arxiv.arxiv.Result):
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


# query = "2106.05963"
# query = "au:Yann LeCun"
# Logic:
# Ctrl-C in the navigation loop to exit and start a new query
# Ctrl-C in the query prompt to exit the program
# Up/Down to navigate through prompts and query history
while True:
    try:
        cnt = 0
        query = session.prompt("Enter arXiv ID or query str: ", multiline=False)
        search_obj = arxiv.Search(query, )
        results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS, offset=cnt)
        if len(results_arxiv) == 0:
            print("No results found.")
            continue
        elif len(results_arxiv) == 1:
            paper = results_arxiv[0]
            print_arxiv_entry(paper)
            # Add the entry if confirmed
            if questionary.confirm("Return?").ask():
                pass
        elif len(results_arxiv) > 1:
            # multiple results found, complex logic to navigate through results
            last_selection = None  # last selected result to highlight
            while True:
                # looping of results and pages, navigating through search results
                print("Multiple results found. Please select one:")
                choices = [f"{i + 1}: [{paper.entry_id.split('/')[-1]}] {paper.title} " for i, paper in enumerate(results_arxiv)]
                if len(results_arxiv) == MAX_RESULTS:
                    choices.append("0: Next page")
                if cnt > 0:
                    choices.append("-1: Prev page")
                selection = questionary.select("Select paper:", choices=choices, default=None if last_selection is None
                                               else choices[last_selection]).ask()
                selection = int(selection.split(":")[0])
                if selection == 0:
                    cnt += MAX_RESULTS
                    results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS, offset=cnt)
                    continue
                if selection == -1:
                    cnt -= MAX_RESULTS
                    results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS, offset=cnt)
                    continue
                else:
                    paper = results_arxiv[int(selection) - 1]
                    last_selection = int(selection) - 1
                    print_arxiv_entry(paper)
                    if questionary.confirm("Return?").ask():
                        # Add the entry if confirmed
                        pass
                    # if questionary.confirm("Back to the list").ask():
                    #     continue
                    # else:
                    #     break

    except Exception as e:
        continue
