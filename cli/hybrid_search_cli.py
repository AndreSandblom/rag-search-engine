import argparse

from lib.hybrid_search import normalize_scores, rrf_search_command, weighted_search_command

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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores([float(x) for x in args.scores])
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, alpha=args.alpha, limit=args.limit)
        case "rrf-search":
            rrf_search_command(args.query, k=args.k, limit=args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()