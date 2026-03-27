"""arXiv API wrapper for literature search."""

import time
from typing import Optional

import arxiv


def search_arxiv(
    query: str,
    max_results: int = 50,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    delay_seconds: float = 3.0,
) -> list[dict]:
    """Search arXiv and return normalized paper dicts.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        sort_by: Sort criterion (Relevance or SubmittedDate).
        delay_seconds: Delay between API requests for rate limiting.

    Returns:
        List of normalized paper dicts.
    """
    client = arxiv.Client(
        page_size=100,
        delay_seconds=delay_seconds,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )

    papers = []
    for result in client.results(search):
        paper = _normalize_result(result)
        papers.append(paper)

    return papers


def fetch_paper_by_id(arxiv_id: str, delay_seconds: float = 3.0) -> Optional[dict]:
    """Fetch a single paper by its arXiv ID.

    Args:
        arxiv_id: The arXiv ID (e.g., "2509.00300").
        delay_seconds: Delay for rate limiting.

    Returns:
        Normalized paper dict, or None if not found.
    """
    client = arxiv.Client(delay_seconds=delay_seconds, num_retries=3)
    search = arxiv.Search(id_list=[arxiv_id])

    results = list(client.results(search))
    if not results:
        return None

    return _normalize_result(results[0])


def _normalize_result(result: arxiv.Result) -> dict:
    """Convert an arxiv.Result to a normalized paper dict."""
    # Extract arXiv ID from the entry_id URL
    arxiv_id = result.entry_id.split("/abs/")[-1]
    # Remove version suffix for the canonical ID
    arxiv_id_base = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

    return {
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "abstract": result.summary,
        "published": result.published.isoformat() if result.published else None,
        "year": result.published.year if result.published else None,
        "pdf_url": result.pdf_url,
        "source": "arxiv",
        "source_ids": {"arxiv": arxiv_id_base},
        "categories": [cat for cat in result.categories],
        "doi": result.doi,
    }
