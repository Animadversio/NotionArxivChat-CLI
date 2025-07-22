import datetime
import arxiv
from pyzotero import zotero

def convert_arxiv_result_to_zotero_item(result: arxiv.arxiv.Result, item_type='preprint'):
    """
    Convert an arxiv.Result object into a Zotero item dict for PyZotero's create_items().
    """
    # Build creators list
    creators = []
    for author in result.authors:
        name = author.name
        parts = name.split()
        if len(parts) > 1:
            creators.append({
                'creatorType': 'author',
                'firstName': parts[0],
                'lastName': ' '.join(parts[1:])
            })
        else:
            creators.append({
                'creatorType': 'author',
                'firstName': '',
                'lastName': name
            })
    
    # Extract arXiv ID (with version)
    arxiv_id = result.entry_id.rsplit('/', 1)[-1]
    
    if item_type == 'journalArticle':
        # Construct Zotero item payload
        item = {
            'itemType': 'journalArticle',
            'title': result.title,
            'creators': creators,
            'date': result.published.date().isoformat(),
            'abstractNote': result.summary,
            'url': result.entry_id,
            'publicationTitle': 'arXiv',
            'extra': f'arXiv ID: {arxiv_id}'
        }
    elif item_type == 'preprint':
        # item = {
        #     'itemType': 'preprint',
        #     'title': result.title,
        #     'creators': creators,
        #     'date': result.published.date().isoformat(),
        #     'abstractNote': result.summary,
        #     'url': result.entry_id,
        # }
        # Construct Zotero preprint item payload
        item = {
            'itemType': 'preprint',
            'title': result.title,
            'creators': creators,
            'abstractNote': result.summary,
            'genre': 'Preprint',
            'repository': 'arXiv',
            'archive': 'arXiv',
            'archiveID': arxiv_id,
            'archiveLocation': result.primary_category,
            'date': result.published.date().isoformat(),
            'DOI': result.doi or '',
            'citationKey': arxiv_id.replace('.', '_'),
            'url': result.entry_id,
            'accessDate': datetime.date.today().isoformat(),
            'libraryCatalog': 'arXiv',
            # 'tags': [],
            # 'collections': [],
            # 'relations': {}
        }
    return item


def check_existing_title_in_zotero(title: str, zot: zotero.Zotero):
    """
    Check whether an item with exactly this title already exists in the Zotero library.
    
    Parameters:
    - title (str): the title to check
    - zot (pyzotero.Zotero): an authenticated Zotero client
    
    Returns:
    - bool: True if no existing item has exactly this title (case-insensitive), False otherwise
    """
    # Run a full-text search for the title
    # (you can also scope to itemType='journalArticle' or whatever you like)
    matches = zot.items(q=title, ) # itemType='journalArticle'
    # Compare exact title strings (case-insensitive)
    lower = title.strip().lower()
    for item in matches:
        existing = item['data'].get('title', '').strip().lower()
        if existing == lower:
            # print the item key
            print("Entry already exists in Zotero:")
            print(item['key'], item['data']['title'])
            return item
    return None


def add_to_zotero(zot: zotero.Zotero, paper: arxiv.arxiv.Result, item_type='preprint', ):
    """
    Add an arxiv.Result object to the Zotero library.
    If the entry already exists, it will skip adding the entry and return the item.
    If the entry does not exist, it will create a new item and return the item.
    """
    existing_item = check_existing_title_in_zotero(paper.title, zot)
    if existing_item is not None:
        print("Entry already exists in Zotero, skipping...returning existing item")
        zot_item = existing_item
        # print(existing_item['key'], existing_item['data']['title'])
    else:
        print("Entry does not exist in Zotero. Adding to Zotero.")
        zotero_item = convert_arxiv_result_to_zotero_item(
                            paper, item_type=item_type)
        # zotero_item['collections'].extend(collection_keys)
        output = zot.create_items([zotero_item])
        # print(output)
        if len(output["successful"]) > 0:
            # print(output["successful"][0]["key"])
            print("Successfully added to Zotero!")
            zot_item = None
            # slightly hacky but usually works. 
            # this is not working for some reason. !!! TODO: fix this. 
            # zot_item = zot.top(limit=1)
            # if len(zot_item) == 0:
            #     print("No item found at top of Zotero!! Returning None")
            #     zot_item = None
            # else:
            #     zot_item = zot_item[0]
            #     if zot_item['key'] != output["successful"][0]["key"]:
            #         print("Zotero item key does not match the created item key!!! Returning None")
            #         zot_item = None
        else:
            print("Failed to add to Zotero!!")
            print(output["failed"][0]["error"])
            zot_item = None
    return zot_item