#!/usr/bin/env python3

import argparse
import math
import sys

from lib.keyword_search import search_command,build_command
from lib.inverted_index import InvertedIndex
from lib.text_process import tokenize_term


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to check")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "tf":
            indexer = InvertedIndex()
            try:
                indexer.load()
            except Exception as e:
                print(f"Error loading index: {e}")
                sys.exit(1)
            try:
                token = tokenize_term(args.term)
            except Exception as e:
                print(f"Error tokenizing term: {e}")
            tf = indexer.get_tf(args.doc_id, token)
            print(f"Term frequency for '{token}' in doc {args.doc_id}: {tf}")
        case "build":
            print("Building...")
            build_command()
        case "idf":
            indexer = InvertedIndex()
            try:
                indexer.load()
            except Exception as e:
                print(f"Error loading index: {e}")
                sys.exit(1)
            try:
                token = tokenize_term(args.term)
            except Exception as e:
                print(f"Error tokenizing term: {e}")
                sys.exit(1)
            idf = indexer.get_idf(token)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            indexer = InvertedIndex()
            try:
                indexer.load()
            except Exception as e:
                print(f"Error loading index: {e}")
                sys.exit(1)
            try:
                token = tokenize_term(args.term)
            except Exception as e:
                print(f"Error tokenizing term: {e}")
                sys.exit(1)
            tf_idf = indexer.get_tfidf(args.doc_id, token)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

