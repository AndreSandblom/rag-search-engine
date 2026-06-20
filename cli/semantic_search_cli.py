#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from lib.semantic_search import embed_query_text, embed_text, verify_model,verify_embeddings, SemanticSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the semantic search model can be loaded")
    embed_parser = subparsers.add_parser("embed_text", help="Generate and display embedding for a sample text")
    embed_parser.add_argument("text", help="Text to embed")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings for the loaded movies")
    query_parser = subparsers.add_parser("embed_query", help="Generate and display embedding for a query text")
    query_parser.add_argument("text", help="Query text to embed")
    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of search results to return")
    

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.text)
        case "search":
            semantic_search = SemanticSearch()
            movies = load_movies()
            semantic_search.load_or_create_embeddings(movies)
            results = semantic_search.search(args.query, limit=DEFAULT_SEARCH_LIMIT)
            for i,result in enumerate(results,1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})\n  {result['description']}\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()