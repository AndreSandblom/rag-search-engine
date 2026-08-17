import argparse

from lib.hybrid_search import normalize_scores, rrf_search_command, weighted_search_command
from lib.query_enhancement import rewrite_query, spell_correction, expand_query

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", help="List of scores to normalize")

    weighted_parser = subparser.add_parser("weighted-search", help="Perform weighted hybrid search (not implemented)")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for semantic search")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")
    
    rrf_parser = subparser.add_parser("rrf-search", help="Perform RRF hybrid search (not implemented)")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("--k", type=int, default=60, help="RRF parameter k")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")
    rrf_parser.add_argument("--enhance",type=str,choices=["spell","rewrite", "expand"],help="Query enhancement method (e.g., spell correction, rewrite, expansion)")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method to use (e.g., individual, batch, cross_encoder)")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores([float(x) for x in args.scores])
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, alpha=args.alpha, limit=args.limit)
        case "rrf-search":
            results = rrf_search_command(args.query, k=args.k, limit=args.limit, enhance=args.enhance, rerank_method=args.rerank_method)
            if results["enhanced_query"]:
                print(f"Enhanced query ({results['enhance_method']}): '{results['original_query']}' -> '{results['enhanced_query']}'\n")
            if results["rerank_method"] == "individual":
                print(f"Re-ranking top {args.limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
            elif results["rerank_method"] == "batch":
                print(f"Re-ranking top {args.limit} results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
            elif results["rerank_method"] == "cross_encoder":
                print(f"Re-ranking top {args.limit} results using cross-encoder method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
            for i, result in enumerate(results["results"], 1):
                print(f"\n{i}. {result['title']}")
                if "reranked_score" in result:
                    print(f"  Re-rank Score: {result['reranked_score']:.3f}/10")
                elif "rerank_rank" in result:
                    print(f"  Re-rank Rank: {result['rerank_rank']}")
                elif "cross_encoder_score" in result:
                    print(f"  Cross-Encoder Score: {result['cross_encoder_score']:.3f}/10")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(f"  BM25 Rank: {result.get('bm25_rank', 0.0)}, Semantic Rank: {result.get('semantic_rank', 0.0)}")
                print(f"  {result['description'][0:100]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()