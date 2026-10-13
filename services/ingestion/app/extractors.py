# ------------------------------------------------------------------------------------------------------
# Script Name: extractors.py
# Description: This script defines functions to extract text from various file formats (PDF, HTML, Markdown) 
# and a mapping of file extensions to their corresponding extractor functions.
# ------------------------------------------------------------------------------------------------------

# Configuration
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

# ------------------------------------------------------------------------------------------------------
# Functions

def extract_text_pdf(path: Path) -> str:
    output = [] # initialize an empty list to store the extracted text from each page of the PDF

    with open(path, "rb") as f: # open the PDF file in binary read mode

        reader = PdfReader(f) # read the file using PdfReader from the pypdf library
        for page in reader.pages: # iterate through each page in the PDF
            output.append(page.extract_text() or "")  # extract the text from the current page and append it to the output list

    return "\n\n".join(output) # join the extracted text from all pages with double newlines and return it as a single string

def extract_text_html(path: Path) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f: # open the file in read mode with UTF-8 encoding and ignore any errors

        return BeautifulSoup(f.read(), "html.parser").get_text( # parse the HTML content using BeautifulSoup and extract the text, joining it with newlines and stripping any leading/trailing whitespace
            separator="\n", strip=True
        )

def extract_text_md(path: Path) -> str:

    with open(path, encoding="utf-8", errors="ignore") as f: # open the file in read mode with UTF-8 encoding and ignore any errors
        return f.read() # read the entire content of the Markdown file and return it as a string


EXTRACTORS = {
    ".pdf": extract_text_pdf,
    ".html": extract_text_html,
    ".htm": extract_text_html,
    ".md": extract_text_md,
    ".markdown": extract_text_md,
}
