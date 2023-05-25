import os
import pickle as pkl
from os.path import join
import textwrap
from notion_client import Client

def QA_notion_blocks(Q, A, refs=()):
    """
    notion.blocks.children.append(page_id, children=QA_notion_blocks("Q1", "A1"))
    notion.blocks.children.append(page_id, children=QA_notion_blocks("Q1", "A1", ("ref1", "ref2")))

    :param Q: str question
    :param A: str answer
    :param refs: list or tuple of str references
    :return:
    """
    ref_blocks = []
    for ref in refs:
        ref_blocks.append({'quote': {"rich_text": [{"text": {"content": ref}}]}})
    return [
        {'divider': {}},
        {'paragraph': {"rich_text": [{"text": {"content": f"Question:"}, 'annotations': {'bold': True}}, ]}},
        {'paragraph': {"rich_text": [{"text": {"content": Q}}]}},
        {'paragraph': {"rich_text": [{"text": {"content": f"Answer:"}, 'annotations': {'bold': True}}, ]}},
        {'paragraph': {"rich_text": [{"text": {"content": A}}]}},
        {'toggle': {"rich_text": [{"text": {"content": f"Reference:"}, 'annotations': {'bold': True}}, ],
                    "children": ref_blocks, }},
    ]


def append_chathistory_to_notion_page(notion: Client, page_id: str, chat_history: list, ref_maxlen=250):
    """
    Append chat history to notion page

    :param notion: notion client
    :param page_id: str
    :param chat_history: list of tuple (query, answer_struct)
    :return:
    """
    for query, ans_struct in chat_history:
        answer = ans_struct["answer"]
        refdocs = ans_struct['source_documents']
        refstrs = [refdoc.page_content[:ref_maxlen] for refdoc in refdocs]
        notion.blocks.children.append(page_id, children=QA_notion_blocks(query, answer, refstrs))


def print_entries(entries_return, print_prop=()):
    # formating the output, so Name starts at the same column
    # pad the string to be 36 character
    if type(entries_return) == dict:
        entries_return = entries_return["results"]

    print("id".ljust(36), "\t", "Name",)
    for entry in entries_return:
        print(entry["id"], "\t", entry["properties"]["Name"]["title"][0]["plain_text"], entry["url"] if "url" in print_prop else "")


def clean_metadata(metadata):
    metadata_new = {}
    for k, v in metadata.items():
        if v is None or v == []:
            continue
        metadata_new[k] = metadata[k]
    return metadata_new


def save_qa_history(query, result, qa_path,):
    uid = 0
    while os.path.exists(join(qa_path, f"QA{uid:05d}.pkl")):
        uid += 1
    pkl.dump((query, result), open(join(qa_path, f"QA{uid:05d}.pkl"), "wb"))

    pkl_path = join(qa_path, "chat_history.pkl")
    if os.path.exists(pkl_path):
        chat_history = pkl.load(open(pkl_path, "rb"))
    else:
        chat_history = []
    chat_history.append((query, result))
    pkl.dump(chat_history, open(pkl_path, "wb"))

    with open(os.path.join(qa_path, "QA.md"), "a", encoding="utf-8") as f:
        f.write("\n**Question:**\n\n")
        f.write(query)
        f.write("\n\n**Answer:**\n\n")
        f.write(result["answer"])
        f.write("\n\nReferences:\n\n")
        for doc in result["source_documents"]:
            f.write("> ")
            f.write(doc.page_content[:250])
            f.write("\n\n")
        f.write("-------------------------\n\n")


def load_qa_history(qa_path):
    pkl_path = join(qa_path, "chat_history.pkl")
    if os.path.exists(pkl_path):
        chat_history = pkl.load(open(pkl_path, "rb"))
    else:
        chat_history = []
    queries = [q for q, _ in chat_history]
    results = [r for _, r in chat_history]
    return chat_history, queries, results


def print_qa_result(result, ref_maxlen=200, line_width=80):
    print("\nAnswer:")
    print(textwrap.fill(result["answer"], line_width))
    print("\nReference:")
    for refdoc in result['source_documents']:
        print("Ref doc:\n", refdoc.metadata)
        print(textwrap.fill(refdoc.page_content[:ref_maxlen], line_width))
    print("\n")


def update_title(notion: Client, page_id, title):
    update_struct = {
        "properties": {
            "title": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            }
        }
    }
    notion.pages.update(page_id, **update_struct)