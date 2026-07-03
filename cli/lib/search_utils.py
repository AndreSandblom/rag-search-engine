import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
WORDS_PATH = os.path.join(PROJECT_ROOT,"data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
BM_25K1 = 1.5
BM_25B = 0.75
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(WORDS_PATH,"r") as f:
        words = f.read().splitlines()
    return words

def format_search_result(
    doc_id: int,
    title: str,
    document: str, 
    score: float,
    **metadata: Any
):
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
