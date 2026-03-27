"""Generate markdown summary reports from the paper database."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.storage.paper_store import Paper, PaperStore


def generate_summary(
    store: PaperStore,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a markdown summary report of all papers, grouped by tag.

    Args:
        store: The paper store to summarize.
        output_path: If provided, write the report to this file.

    Returns:
        The markdown report as a string.
    """
    papers = store.get_all()
    lines = []

    lines.append("# Literature Review Summary")
    lines.append("")
    lines.append(f"**Total papers:** {len(papers)}")
    lines.append("")

    # Stats by read status
    status_counts = defaultdict(int)
    for p in papers:
        status_counts[p.read_status] += 1
    lines.append("## Reading Progress")
    lines.append("")
    for status in ["unread", "skimmed", "read", "annotated"]:
        count = status_counts.get(status, 0)
        lines.append(f"- {status}: {count}")
    lines.append("")

    # Group by tags
    tag_groups: dict[str, list[Paper]] = defaultdict(list)
    untagged = []
    for paper in papers:
        if paper.tags:
            for tag in paper.tags:
                tag_groups[tag].append(paper)
        else:
            untagged.append(paper)

    # Sort tags alphabetically
    lines.append("## Papers by Tag")
    lines.append("")

    for tag in sorted(tag_groups.keys()):
        tag_papers = tag_groups[tag]
        # Sort by citation count (descending), then title
        tag_papers.sort(
            key=lambda p: (-(p.citation_count or 0), p.title)
        )

        lines.append(f"### {tag} ({len(tag_papers)} papers)")
        lines.append("")

        for paper in tag_papers:
            lines.append(_format_paper_entry(paper))
            lines.append("")

    if untagged:
        lines.append(f"### Untagged ({len(untagged)} papers)")
        lines.append("")
        for paper in untagged:
            lines.append(_format_paper_entry(paper))
            lines.append("")

    # Year distribution
    lines.append("## Year Distribution")
    lines.append("")
    year_counts: dict[int, int] = defaultdict(int)
    for paper in papers:
        if paper.year:
            year_counts[paper.year] += 1
    for year in sorted(year_counts.keys()):
        lines.append(f"- {year}: {year_counts[year]} papers")
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def _format_paper_entry(paper: Paper) -> str:
    """Format a single paper as a markdown list entry."""
    # Authors (truncate to first 3)
    if len(paper.authors) > 3:
        authors_str = ", ".join(paper.authors[:3]) + " et al."
    else:
        authors_str = ", ".join(paper.authors) if paper.authors else "Unknown"

    year_str = f" ({paper.year})" if paper.year else ""
    cite_str = f" [cited: {paper.citation_count}]" if paper.citation_count else ""

    entry = f"- **{paper.title}**{year_str}{cite_str}\n"
    entry += f"  {authors_str}\n"

    if paper.tldr:
        entry += f"  > {paper.tldr}\n"
    elif paper.abstract:
        # Show first 200 chars of abstract
        snippet = paper.abstract[:200].replace("\n", " ")
        if len(paper.abstract) > 200:
            snippet += "..."
        entry += f"  > {snippet}\n"

    if paper.tags:
        entry += f"  Tags: {', '.join(paper.tags)}\n"

    if paper.notes:
        entry += f"  Notes: {paper.notes}\n"

    status_marker = {"unread": " ", "skimmed": "~", "read": "v", "annotated": "*"}
    entry += f"  Status: [{status_marker.get(paper.read_status, ' ')}] {paper.read_status}"

    return entry
