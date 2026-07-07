import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies

def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize a list of scores to the range [0, 1].

    Args:
        scores: List of scores to normalize.

    Returns:
        List of normalized scores.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def weighted_search_command(query: str, alpha: float, limit: int = 5) -> None:
    """Perform a weighted hybrid search and print the results.

    Args:
        query: Search query.
        alpha: Weighting factor for semantic search.
        limit: Number of search results to return.
    """
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.weighted_search(query, alpha=alpha, limit=limit)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"  BM25: {result.get('bm25_score', 0.0):.3f}, Semantic: {result.get('semantic_score', 0.0):.3f}")
        print(f"  {result['description']}...")

def rrf_search_command(query: str, k: int, limit: int = 5) -> None:
    """Perform an RRF hybrid search and print the results.

    Args:
        query: Search query.
        k: RRF parameter k.
        limit: Number of search results to return.
    """
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, k=k, limit=limit)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"  RRF Score: {result['rrf_score']:.3f}")
        print(f"  BM25 Rank: {result.get('bm25_rank', 0.0)}, Semantic Rank: {result.get('semantic_rank', 0.0)}")
        print(f"  {result['description']}...")

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit=limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit=limit*500)

        # Extract scores
        bm25_scores = [result.get("score", 0.0) for result in bm25_results]
        semantic_scores = [result.get("score", 0.0) for result in semantic_results]
        
        # Normalize scores
        normalized_bm25_scores = normalize_scores(bm25_scores)
        normalized_semantic_scores = normalize_scores(semantic_scores)
        
        # Create a dictionary mapping document IDs to their information and scores
        doc_scores = {}
        
        # Add BM25 scores to the dictionary
        for i, result in enumerate(bm25_results):
            doc_id = result.get("id")
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": result,
                    "bm25_score": normalized_bm25_scores[i],
                    "semantic_score": 0.0,
                }
            else:
                doc_scores[doc_id]["bm25_score"] = normalized_bm25_scores[i]
        
        # Add semantic scores to the dictionary
        for i, result in enumerate(semantic_results):
            doc_id = result.get("id")
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": result,
                    "bm25_score": 0.0,
                    "semantic_score": normalized_semantic_scores[i],
                }
            else:
                doc_scores[doc_id]["semantic_score"] = normalized_semantic_scores[i]
        
        # Calculate hybrid scores and prepare results
        results = []
        for doc_id, data in doc_scores.items():
            hybrid = hybrid_score(
                data["bm25_score"],
                data["semantic_score"],
                alpha=alpha
            )
            result_dict = data["document"].copy()
            result_dict["hybrid_score"] = hybrid
            results.append(result_dict)
        
        # Sort by hybrid score in descending order
        results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        
        return results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit=limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit=limit*500)

        results = {}
        for i, result in enumerate(bm25_results):
            doc_id = result.get("id")
            if doc_id not in results:
                results[doc_id] = {
                    "document": result,
                    "bm25_rank": i + 1,
                    "semantic_rank": float('inf'),
                }
            else:
                results[doc_id]["bm25_rank"] = i + 1

        for i, result in enumerate(semantic_results):
            doc_id = result.get("id")
            if doc_id not in results:
                results[doc_id] = {
                    "document": result,
                    "bm25_rank": float('inf'),
                    "semantic_rank": i + 1,
                }
            else:
                results[doc_id]["semantic_rank"] = i + 1

        # Calculate RRF scores
        for doc_id, data in results.items():
            rrf_value = rrf_score(data["bm25_rank"], k) + rrf_score(data["semantic_rank"], k)
            data["rrf_score"] = rrf_value

        # Sort by RRF score in descending order
        sorted_results = sorted(results.values(), key=lambda x: x["rrf_score"], reverse=True)

        final_results = []
        for data in sorted_results[:limit]:
            result_dict = data["document"].copy()
            result_dict["rrf_score"] = data["rrf_score"]
            result_dict["bm25_rank"] = data["bm25_rank"]
            result_dict["semantic_rank"] = data["semantic_rank"]
            final_results.append(result_dict)

        return final_results