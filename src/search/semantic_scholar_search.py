"""Semantic Scholar API wrapper for literature search."""

import time
from pathlib import Path
from typing import Optional

import yaml
from semanticscholar import SemanticScholar


# Fields to request from the Semantic Scholar API
_PAPER_FIELDS = [
    "title",
    "abstract",
    "authors",
    "year",
    "externalIds",
    "citationCount",
    "tldr",
    "fieldsOfStudy",
    "url",
    "publicationDate",
]


def _load_api_key() -> Optional[str]:
    """Load Semantic Scholar API key from config if available."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "api_keys.yaml"
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    key = config.get("semantic_scholar_api_key", "")
    return key if key else None


def search_semantic_scholar(
    query: str,
    max_results: int = 50,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Search Semantic Scholar and return normalized paper dicts.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        api_key: Optional API key. If None, tries to load from config.

    Returns:
        List of normalized paper dicts.
    """
    if api_key is None:
        api_key = _load_api_key()

    sch = SemanticScholar(api_key=api_key) if api_key else SemanticScholar()

    results = sch.search_paper(
        query,
        limit=min(max_results, 100),
        fields=_PAPER_FIELDS,
    )

    papers = []
    if results is None:
        return papers

    for result in results[:max_results]:
        paper = _normalize_result(result)
        if paper is not None:
            papers.append(paper)

    return papers


def fetch_paper_by_s2_id(
    paper_id: str,
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """Fetch a single paper by Semantic Scholar ID, DOI, or arXiv ID.

    Args:
        paper_id: A Semantic Scholar paper ID, DOI (e.g., "10.1145/..."),
                  or arXiv ID prefixed with "ARXIV:" (e.g., "ARXIV:2509.00300").
        api_key: Optional API key.

    Returns:
        Normalized paper dict, or None if not found.
    """
    if api_key is None:
        api_key = _load_api_key()

    sch = SemanticScholar(api_key=api_key) if api_key else SemanticScholar()

    try:
        result = sch.get_paper(paper_id, fields=_PAPER_FIELDS)
    except Exception:
        return None

    return _normalize_result(result)


def _normalize_result(result) -> Optional[dict]:
    """Convert a Semantic Scholar result to a normalized paper dict."""
    if result is None:
        return None

    title = getattr(result, "title", None)
    if not title:
        return None

    # Extract external IDs
    external_ids = getattr(result, "externalIds", {}) or {}
    source_ids = {}
    if external_ids.get("ArXiv"):
        source_ids["arxiv"] = external_ids["ArXiv"]
    if external_ids.get("DOI"):
        source_ids["doi"] = external_ids["DOI"]
    if external_ids.get("CorpusId"):
        source_ids["s2"] = str(external_ids["CorpusId"])

    # Extract authors
    authors_raw = getattr(result, "authors", []) or []
    authors = []
    for a in authors_raw:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        if name:
            authors.append(name)

    # Extract TLDR
    tldr_obj = getattr(result, "tldr", None)
    tldr = None
    if tldr_obj:
        tldr = tldr_obj.get("text") if isinstance(tldr_obj, dict) else getattr(tldr_obj, "text", None)

    # Build PDF URL from arXiv ID if available
    pdf_url = None
    if source_ids.get("arxiv"):
        pdf_url = f"https://arxiv.org/pdf/{source_ids['arxiv']}.pdf"

    publication_date = getattr(result, "publicationDate", None)
    published = publication_date.isoformat() if publication_date else None

    return {
        "title": title,
        "authors": authors,
        "abstract": getattr(result, "abstract", None) or "",
        "published": published,
        "year": getattr(result, "year", None),
        "pdf_url": pdf_url,
        "source": "semantic_scholar",
        "source_ids": source_ids,
        "categories": getattr(result, "fieldsOfStudy", []) or [],
        "doi": external_ids.get("DOI"),
        "citation_count": getattr(result, "citationCount", None),
        "tldr": tldr,
    }
