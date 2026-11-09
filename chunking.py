import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    book_name: str
    chunk_id:  int
    text:      str


def chunk_book(book_name, text, chunk_size=200, overlap=50):
    words  = text.split()
    stride = chunk_size - overlap
    chunks = []

    for i, start in enumerate(range(0, max(1, len(words) - overlap), stride)):
        end  = start + chunk_size
        chunk_text = " ".join(words[start:end])

        if not chunk_text.strip():
            continue

        chunks.append(Passage(book_name=book_name, chunk_id=i, text=chunk_text))

        if end >= len(words):
            break

    return chunks


def chunk_all_books(books, chunk_size=200, overlap=50):
    all_passages = []

    for name, text in books.items():
        chunks = chunk_book(name, text, chunk_size, overlap)
        all_passages.extend(chunks)
        logger.info(f"{name}: {len(chunks)} chunks")

    logger.info(f"total passages: {len(all_passages)}")
    return all_passages
