import sys

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, BM_25K1, BM_25B
from .inverted_index import InvertedIndex
from .text_process import tokenize, tokenize_term

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

def tf_command(doc_id: int, term: str):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    try:
        token = tokenize_term(term)
    except Exception as e:
        print(f"Error tokenizing term: {e}")
        sys.exit(1)
    tf = indexer.get_tf(doc_id, token)
    print(f"Term frequency of '{term}' in document '{doc_id}': {tf}")

def idf_command(term: str):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    try:
        token = tokenize_term(term)
    except Exception as e:
        print(f"Error tokenizing term: {e}")
        sys.exit(1)
    idf = indexer.get_idf(token)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tf_idf_command(doc_id: int, term: str):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    try:
        token = tokenize_term(term)
    except Exception as e:
        print(f"Error tokenizing term: {e}")
        sys.exit(1)
    tf_idf = indexer.get_tfidf(doc_id, token)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def bm25_tf_command(doc_id: int, term: str, k1: float = BM_25K1, b: float = BM_25B):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    try:
        token = tokenize_term(term)
    except Exception as e:
        print(f"Error tokenizing term: {e}")
        sys.exit(1)
    bm25_tf = indexer.get_bm_tf(doc_id, token, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")

def bm25_idf_command(term: str):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    try:
        token = tokenize_term(term)
    except Exception as e:
        print(f"Error tokenizing term: {e}")
        sys.exit(1)
    bm25_idf = indexer.get_bm25_idf(token)
    print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    
    results = indexer.bm25_search(query, limit)
    return results