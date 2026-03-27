"""Deduplicate papers across sources by DOI, arXiv ID, or title similarity."""

import re
from typing import Optional


def normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, remove punctuation/whitespace."""
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_token_overlap(title_a: str, title_b: str) -> float:
    """Compute Jaccard similarity between title tokens.

    Returns:
        Float between 0 and 1. Higher means more similar.
    """
    tokens_a = set(normalize_title(title_a).split())
    tokens_b = set(normalize_title(title_b).split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def are_duplicates(paper_a: dict, paper_b: dict, title_threshold: float = 0.85) -> bool:
    """Check if two paper dicts refer to the same paper.

    Matches by (in order):
    1. Shared DOI
    2. Shared arXiv ID
    3. Title token overlap above threshold

    Args:
        paper_a: First paper dict.
        paper_b: Second paper dict.
        title_threshold: Minimum Jaccard similarity to consider titles a match.

    Returns:
        True if the papers are likely duplicates.
    """
    ids_a = paper_a.get("source_ids", {})
    ids_b = paper_b.get("source_ids", {})

    # Match by DOI
    doi_a = ids_a.get("doi") or paper_a.get("doi")
    doi_b = ids_b.get("doi") or paper_b.get("doi")
    if doi_a and doi_b and doi_a.lower() == doi_b.lower():
        return True

    # Match by arXiv ID
    arxiv_a = ids_a.get("arxiv")
    arxiv_b = ids_b.get("arxiv")
    if arxiv_a and arxiv_b and arxiv_a == arxiv_b:
        return True

    # Match by title similarity
    title_a = paper_a.get("title", "")
    title_b = paper_b.get("title", "")
    if title_a and title_b:
        similarity = title_token_overlap(title_a, title_b)
        if similarity >= title_threshold:
            return True

    return False


def merge_paper_dicts(primary: dict, secondary: dict) -> dict:
    """Merge two paper dicts, preferring data from the primary.

    Fills in missing fields from the secondary.
    """
    merged = dict(primary)

    # Merge source_ids
    merged_ids = dict(secondary.get("source_ids", {}))
    merged_ids.update(primary.get("source_ids", {}))
    merged["source_ids"] = merged_ids

    # Merge categories
    cats = set(primary.get("categories", []))
    cats.update(secondary.get("categories", []))
    merged["categories"] = sorted(cats)

    # Fill missing fields from secondary
    for field_name in ["abstract", "year", "published", "pdf_url", "doi", "citation_count", "tldr"]:
        if not merged.get(field_name) and secondary.get(field_name):
            merged[field_name] = secondary[field_name]

    # Use the longer author list
    if len(secondary.get("authors", [])) > len(merged.get("authors", [])):
        merged["authors"] = secondary["authors"]

    return merged


def deduplicate_papers(
    papers: list[dict],
    title_threshold: float = 0.85,
) -> list[dict]:
    """Deduplicate a list of paper dicts.

    When duplicates are found, their metadata is merged.

    Args:
        papers: List of paper dicts to deduplicate.
        title_threshold: Minimum title similarity for dedup.

    Returns:
        Deduplicated list of paper dicts.
    """
    if not papers:
        return []

    unique: list[dict] = []

    for paper in papers:
        found_match = False
        for i, existing in enumerate(unique):
            if are_duplicates(paper, existing, title_threshold):
                # Merge, keeping the existing one as primary
                unique[i] = merge_paper_dicts(existing, paper)
                found_match = True
                break

        if not found_match:
            unique.append(dict(paper))

    return unique
