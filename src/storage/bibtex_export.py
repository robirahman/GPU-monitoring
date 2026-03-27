"""Export papers to BibTeX format."""

import re
from pathlib import Path
from typing import Optional

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

from src.storage.paper_store import Paper, PaperStore


def paper_to_bibtex_entry(paper: Paper) -> dict:
    """Convert a Paper to a BibTeX entry dict.

    Args:
        paper: The paper to convert.

    Returns:
        Dict suitable for bibtexparser.
    """
    # Generate a cite key: first_author_last_name + year + first_title_word
    cite_key = _generate_cite_key(paper)

    entry = {
        "ENTRYTYPE": "article",
        "ID": cite_key,
        "title": paper.title,
        "author": " and ".join(paper.authors),
        "year": str(paper.year) if paper.year else "",
    }

    if paper.abstract:
        entry["abstract"] = paper.abstract

    if paper.doi:
        entry["doi"] = paper.doi

    if paper.pdf_url:
        entry["url"] = paper.pdf_url

    if paper.source_ids.get("arxiv"):
        entry["eprint"] = paper.source_ids["arxiv"]
        entry["archiveprefix"] = "arXiv"

    if paper.categories:
        entry["keywords"] = ", ".join(paper.categories)

    return entry


def export_to_bibtex(
    store: PaperStore,
    output_path: Optional[Path] = None,
    papers: Optional[list[Paper]] = None,
) -> str:
    """Export papers to a BibTeX file.

    Args:
        store: The paper store (used if papers is None).
        output_path: Path to write the .bib file. If None, only returns the string.
        papers: Specific papers to export. If None, exports all from store.

    Returns:
        The BibTeX string.
    """
    if papers is None:
        papers = store.get_all()

    db = BibDatabase()
    db.entries = [paper_to_bibtex_entry(p) for p in papers]

    writer = BibTexWriter()
    writer.indent = "  "
    bibtex_str = writer.write(db)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(bibtex_str)

    return bibtex_str


def _generate_cite_key(paper: Paper) -> str:
    """Generate a BibTeX cite key from paper metadata."""
    # First author's last name
    if paper.authors:
        last_name = paper.authors[0].split()[-1].lower()
        last_name = re.sub(r"[^a-z]", "", last_name)
    else:
        last_name = "unknown"

    # Year
    year = str(paper.year) if paper.year else "nodate"

    # First significant word from title
    title_words = re.sub(r"[^a-zA-Z\s]", "", paper.title).split()
    stop_words = {"a", "an", "the", "of", "in", "on", "for", "and", "to", "with", "by"}
    title_word = "untitled"
    for word in title_words:
        if word.lower() not in stop_words:
            title_word = word.lower()
            break

    return f"{last_name}{year}{title_word}"
