"""Unified search that aggregates and deduplicates results from multiple backends."""

from typing import Optional

from tqdm import tqdm

from src.search.arxiv_search import search_arxiv
from src.search.semantic_scholar_search import search_semantic_scholar
from src.analysis.dedup import deduplicate_papers


def unified_search(
    query: str,
    max_results_per_source: int = 50,
    sources: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """Search multiple backends and return deduplicated results.

    Args:
        query: Search query string.
        max_results_per_source: Max results from each source.
        sources: List of sources to search. Defaults to ["arxiv", "semantic_scholar"].
        verbose: Whether to print progress.

    Returns:
        Deduplicated list of normalized paper dicts.
    """
    if sources is None:
        sources = ["arxiv", "semantic_scholar"]

    all_papers = []

    for source in sources:
        if verbose:
            print(f"  Searching {source} for: {query!r}")

        try:
            if source == "arxiv":
                papers = search_arxiv(query, max_results=max_results_per_source)
            elif source == "semantic_scholar":
                papers = search_semantic_scholar(query, max_results=max_results_per_source)
            else:
                print(f"  Unknown source: {source}, skipping")
                continue

            if verbose:
                print(f"    Found {len(papers)} results from {source}")
            all_papers.extend(papers)

        except Exception as e:
            print(f"    Error searching {source}: {e}")

    # Deduplicate across sources
    deduped = deduplicate_papers(all_papers)
    if verbose:
        print(f"  Total after dedup: {len(deduped)} (from {len(all_papers)} raw)")

    return deduped


def batch_search(
    queries: dict[str, list[str]],
    max_results_per_source: int = 20,
    sources: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """Run multiple queries organized by category.

    Args:
        queries: Dict mapping category name to list of query strings.
        max_results_per_source: Max results per source per query.
        sources: Sources to search.

    Returns:
        Dict mapping category name to list of deduplicated papers.
    """
    results = {}

    for category, query_list in queries.items():
        print(f"\n=== Category: {category} ===")
        category_papers = []

        for query in tqdm(query_list, desc=category):
            papers = unified_search(
                query,
                max_results_per_source=max_results_per_source,
                sources=sources,
                verbose=False,
            )
            category_papers.extend(papers)

        # Deduplicate within category
        category_papers = deduplicate_papers(category_papers)
        results[category] = category_papers
        print(f"  {category}: {len(category_papers)} unique papers")

    return results
