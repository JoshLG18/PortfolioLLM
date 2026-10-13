# ------------------------------------------------------------------------------------------------------
# Script Name: github_client.py
# Description: This script defines a GitHubClient class to interact with the GitHub API for fetching repository data, including README files and attachments.
# ------------------------------------------------------------------------------------------------------
# Configuration

import httpx

from .config import (
    GITHUB_TOKEN,
    PORTFOLIO_WEBSITE_REPO,
    PORTFOLIO_WEBSITE_BRANCH,
    PORTFOLIO_ATTACHMENTS_PATH,
)

GITHUB_API = "https://api.github.com"

# ------------------------------------------------------------------------------------------------------
# Functions

def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github.v3+json"} # set the headers to accept the GitHub API v3 JSON format

    if GITHUB_TOKEN: # if the token is present in the environment variables, add it to the headers for authentication
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    return headers

def list_attachments() -> list[str]:
    url = f"{GITHUB_API}/repos/{PORTFOLIO_WEBSITE_REPO}/contents/{PORTFOLIO_ATTACHMENTS_PATH}?ref={PORTFOLIO_WEBSITE_BRANCH}" # construct the URL to fetch the contents of the attachments directory in the specified branch of the repository

    response = httpx.get(url, headers=_headers()) # make a GET request to the GitHub API with the constructed URL and headers
    response.raise_for_status() # raise an exception if the request was unsuccessful

    items = response.json() # parse the JSON response from the API

    return_payload = [{
        "name": item["name"], # extract the name of the attachment file
        "path": item["path"], # extract the path of the attachment file
        "sha": item["sha"], # extract the SHA of the attachment file
        "size": item["size"], # extract the size of the attachment file
        "download_url": item["download_url"], # extract the download URL of the attachment file
    } for item in items if item["type"] == "file"] # filter the items to include only files (excluding directories)
    
    return return_payload # return the list of attachment files with their details

def fetch_attachment(download_url: str) -> bytes:
    response = httpx.get(download_url, headers=_headers()) # make a GET request to the provided download URL with the appropriate headers
    response.raise_for_status() # raise an exception if the request was unsuccessful

    return response.content # return the content of the fetched attachment file as bytes

def fetch_readme(repo: str) -> str:
    url = f"{GITHUB_API}/repos/{repo}/readme" # construct the URL to fetch the README file of the specified repository and branch

    response = httpx.get(url, headers=_headers()) # make a GET request to the GitHub API with the constructed URL and headers
    response.raise_for_status() # raise an exception if the request was unsuccessful

    readme_data = response.json() # parse the JSON response from the API

    download_url = readme_data.get("download_url") # extract the download URL of the README file from the response data

    readme_response = httpx.get(download_url, headers=_headers()) # make a GET request to fetch the content of the README file using its download URL
    readme_response.raise_for_status() # raise an exception if the request was unsuccessful

    return readme_response.text # return the content of the fetched README file as a string

def fetch_all_readmes(repos: list[str]) -> list[dict[str, str]]:
    readmes = [] # initialize an empty dictionary to store the fetched README files

    for repo in repos: # iterate through the list of repositories
        try:
            readme_content = fetch_readme(repo) # fetch the README content for the current repository
            readmes.append({"repo": repo, "text": readme_content}) # store the fetched README content in the dictionary with the repository name as the key

        except Exception as e:
            print(f"WARNING: failed to fetch README for {repo}: {e}") # print a warning message if fetching the README fails for any repository

    return readmes # return the dictionary containing all fetched README files