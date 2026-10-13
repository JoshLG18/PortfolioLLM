import os

from .chunking import Chunk, file_to_chunks
from .config import GITHUB_CACHE_DIR, GITHUB_REPOS
from .github_client import fetch_attachment, list_attachments, fetch_all_readmes

def load_attachment_chunks() -> list[Chunk]:
    chunks = []
    os.makedirs(GITHUB_CACHE_DIR, exist_ok=True) # make the cache directory if it doesn't exist

    for item in list_attachments(): # iterate through the list of attachments fetched from the GitHub repository
        cache_path = os.path.join(GITHUB_CACHE_DIR, item["name"]) # construct the cache path for the current attachment file
        content = fetch_attachment(item["download_url"]) # fetch the content of the attachment file as bytes from the provided download URL

        with open(cache_path, "wb") as f: # open the cache file in binary write mode
            f.write(content)

        try:
            chunks.extend(file_to_chunks(cache_path, source_type="github_attachment")) # convert the cached attachment file into chunks and extend the chunks list with the generated chunks
        except Exception as e:
            print(f"WARNING: failed to chunk {item['name']}: {e}")

    return chunks


def load_readme_chunks() -> list[Chunk]:
    chunks = []
    os.makedirs(GITHUB_CACHE_DIR, exist_ok=True)

    readmes = fetch_all_readmes(GITHUB_REPOS) # fetch the README files for all specified GitHub repositories

    for readme in readmes: # iterate through the list of fetched README files

        cache_path = os.path.join(GITHUB_CACHE_DIR, f"{readme['repo'].replace('/', '_')}_README.md") # construct the cache path for the current README file
        
        with open(cache_path, "w", encoding="utf-8") as f: # open the cache file in write mode with UTF-8 encoding
            f.write(readme["text"]) # write the content of the README file to the cache file

        try:
            chunks.extend(file_to_chunks(cache_path, source_type="github_readme")) # convert the cached README file into chunks and extend the chunks list with the generated chunks
        except Exception as e:
            print(f"WARNING: failed to chunk README for {readme['repo']}: {e}")

    return chunks