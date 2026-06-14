import sys

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from .inverted_index import InvertedIndex
from .text_process import tokenize

indexer = InvertedIndex()

def has_matching_token(query_tokens, title_tokens):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    try:
        indexer.load()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    
    query_tokens = tokenize(query)
    
    results = []
    seen = set()

    for q in query_tokens:
            doc_ids = indexer.get_documents(q)
            for doc_id in doc_ids:
                if doc_id in seen:
                    continue
                movie = indexer.docmap[doc_id]
                results.append(movie)
                seen.add(doc_id)

                if len(results) >= limit:
                    return results
    return results

def build_command():
    indexer = InvertedIndex()

    indexer.build()

    indexer.save()
