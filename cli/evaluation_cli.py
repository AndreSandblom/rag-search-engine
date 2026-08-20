import argparse
import json
from lib.hybrid_search import rrf_search_command
import os

PROJECT_ROOT = os.path.dirname((os.path.dirname(__file__)))
GOLD_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(GOLD_PATH, "r") as f:
        gold_set = json.load(f)


    test_cases = gold_set["test_cases"]

    print(f"k={limit}")
    for case in test_cases:
        query = case['query']
        relevant_results = case['relevant_docs']

        # Perform search using the query and get the results
        search_results = rrf_search_command(query, k=60, limit=limit)

        # Calculate precision@k and recall@k
        retrieved_results = [result['title'] for result in search_results['results']]
        relevant_retrieved = len(set(retrieved_results) & set(relevant_results))
        total_retrieved = len(retrieved_results)
        total_relevant = len(relevant_results)

        precision_at_k = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_at_k:.4f}")
        print(f"  - Recall@{limit}: {recall_at_k:.4f}")
        print(f"  - Retrieved: {', '.join([str(x) for x in retrieved_results])}")
        print(f"  - Relevant: {', '.join([str(x) for x in relevant_results])}\n")

if __name__ == "__main__":
    main()