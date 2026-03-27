"""Batch search: run all configured queries, store, tag, and deduplicate results."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import yaml
from tqdm import tqdm

from src.search.unified_search import unified_search
from src.storage.paper_store import PaperStore
from src.analysis.tagger import tag_all
from src.analysis.dedup import deduplicate_papers


def load_queries(config_path: Path = None) -> dict[str, list[str]]:
    """Load search queries from the YAML config file."""
    if config_path is None:
        config_path = project_root / "config" / "search_queries.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    queries = {}
    for cat_key, cat_data in config.get("categories", {}).items():
        queries[cat_key] = cat_data.get("queries", [])

    return queries


def main():
    print("=== SPAR Literature Search ===\n")

    # Load queries
    queries = load_queries()
    print(f"Loaded {sum(len(q) for q in queries.values())} queries across {len(queries)} categories\n")

    # Initialize store
    store = PaperStore()
    initial_count = store.count()
    print(f"Current database: {initial_count} papers\n")

    # Run searches by category
    all_results = []

    for category, query_list in queries.items():
        print(f"\n--- Category: {category} ---")
        category_results = []

        for query in query_list:
            print(f"  Searching: {query!r}")
            try:
                papers = unified_search(
                    query,
                    max_results_per_source=20,
                    verbose=False,
                )
                category_results.extend(papers)
                print(f"    Found {len(papers)} results")
            except Exception as e:
                print(f"    Error: {e}")

        # Deduplicate within category
        category_results = deduplicate_papers(category_results)
        print(f"  Category total (after dedup): {len(category_results)}")

        # Tag with category name
        for result in category_results:
            cats = result.get("categories", [])
            if category not in cats:
                cats.append(category)
            result["categories"] = cats

        all_results.extend(category_results)

    # Final dedup across all categories
    all_results = deduplicate_papers(all_results)
    print(f"\n=== Total unique results: {len(all_results)} ===")

    # Store results
    added, skipped = store.merge_search_results(all_results)
    print(f"Added: {added}, Skipped (already in DB): {skipped}")

    # Auto-tag all papers
    print("\nAuto-tagging papers...")
    papers = store.get_all()
    tag_all(papers)
    for paper in papers:
        store.update(paper)

    final_count = store.count()
    print(f"\nDatabase now contains {final_count} papers")
    print(f"Database saved to: {store.db_path}")


if __name__ == "__main__":
    main()
