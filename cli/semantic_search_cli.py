#!/usr/bin/env python3

import argparse
import re

from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from lib.semantic_search import embed_chunks, embed_query_text, embed_text, verify_model,verify_embeddings, SemanticSearch,semantic_chunking

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
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a sample text and display the chunks")
    chunk_parser.add_argument("text", help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk in tokens")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of tokens to overlap between chunks")

    chunk_semantic_parser = subparsers.add_parser("semantic_chunk", help="Chunk a sample text using semantic chunking and display the chunks")
    chunk_semantic_parser.add_argument("text", help="Text to chunk")
    chunk_semantic_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of chunks to create")
    chunk_semantic_parser.add_argument("--overlap", type=int, default=0, help="Number of tokens to overlap between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate and display embeddings for chunked text")
    

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
        case "chunk":
            words = args.text.split()
            grouped_chunks = []
            step = args.chunk_size - args.overlap
            start = 0
            
            while start < len(words):
                chunk = " ".join(words[start:start + args.chunk_size])
                grouped_chunks.append(chunk)
                start += step

            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(grouped_chunks, 1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            semantic_chunks = semantic_chunking(args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)

            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(semantic_chunks, 1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()