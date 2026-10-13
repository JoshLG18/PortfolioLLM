# ------------------------------------------------------------------------------------------------------
# Script Name: chunking.py
# Description: This script defines functions to chunk text into smaller segments based on word count and overlap
# ------------------------------------------------------------------------------------------------------

# Configuration
from dataclasses import dataclass
from pathlib import Path
import os
from .extractors import EXTRACTORS
from .config import CHUNK_WORDS, CHUNK_OVERLAP_WORDS, MIN_CHARS

@dataclass
class Chunk:
    chunk_id: str
    source_type: str  # "local_doc" | "github_readme" | "portfolio_header"
    source_id: str
    title: str
    text: str
    section_path: str | None
    token_count: int
    href: str | None = None

# ------------------------------------------------------------------------------------------------------
# Functions

def chunk_text_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:

    words = text.split() # split the input text into a list of words using whitespace as the delimiter

    if len(words) <= chunk_words: # if the number of words is less than or equal to the specified chunk size
        return [text] if len(text) >= MIN_CHARS else [] # return the entire text as a single chunk
    
    chunks, step = [], chunk_words - overlap_words # initialize an empty list to store the chunks and calculate the step size for overlapping chunks

    for start in range(0, len(words), step): # iterate through the words list in steps of the calculated step size
        chunk = " ".join(words[start:start + chunk_words])

        if len(chunk) >= MIN_CHARS: # if the length of the current chunk is greater than or equal to the specified minimum character count
            chunks.append(chunk) # join the words in the current chunk and append it to the chunks list

        if start + chunk_words >= len(words): # if the end index of the current chunk exceeds the length of the words list
            break # exit the loop to avoid creating an incomplete chunk

    return chunks # return the list of generated chunks


def chunk_structured(text: str, source_type: str) -> list[str]:
    """Heading/section-boundary chunking for HTML/Markdown (READMEs, Blog.html,
    Website.html) and sentence-aware paragraph-merge chunking for PDFs.
    TODO: new structure-aware chunker, see plan section "Search pipeline improvements".
    """
    raise NotImplementedError


def file_to_chunks(path: Path, source_type: str) -> list[Chunk]:
    ext = os.path.splitext(path)[1].lower() # get the file extension of the input path and convert it to lowercase
    text = EXTRACTORS[ext](path) # use the appropriate extractor function from the EXTRACTORS dictionary to extract text from the file based on its extension
    base = os.path.basename(path) # get the base name of the file (i.e., the file name without the directory path)

    return [
        Chunk(
            chunk_id=f"{base}__{i}",
            source_type=source_type,
            source_id=base,
            title=f"{base} — chunk{i+1}",
            text=chunk,
            section_path=None,
            token_count=len(chunk.split()),
        )
        for i, chunk in enumerate(chunk_text_words(text, chunk_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP_WORDS))
    ]