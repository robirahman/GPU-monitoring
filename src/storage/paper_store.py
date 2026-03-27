"""JSON-based paper database with CRUD operations."""

import json
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Paper:
    """A paper in the literature database."""

    id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    published: Optional[str] = None
    pdf_url: Optional[str] = None
    source_ids: dict[str, str] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    citation_count: Optional[int] = None
    tldr: Optional[str] = None
    doi: Optional[str] = None
    notes: str = ""
    relevance_score: Optional[float] = None
    read_status: str = "unread"  # unread, skimmed, read, annotated
    added_date: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.added_date:
            self.added_date = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_search_result(cls, result: dict) -> "Paper":
        """Create a Paper from a normalized search result dict."""
        return cls(
            title=result.get("title", ""),
            authors=result.get("authors", []),
            abstract=result.get("abstract", ""),
            year=result.get("year"),
            published=result.get("published"),
            pdf_url=result.get("pdf_url"),
            source_ids=result.get("source_ids", {}),
            categories=result.get("categories", []),
            citation_count=result.get("citation_count"),
            tldr=result.get("tldr"),
            doi=result.get("doi"),
        )


class PaperStore:
    """JSON file-backed paper database.

    Provides CRUD operations with atomic writes for safety.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the store.

        Args:
            db_path: Path to the JSON database file.
                     Defaults to data/papers.json relative to project root.
        """
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent.parent / "data" / "papers.json"
        self.db_path = Path(db_path)
        self._papers: dict[str, Paper] = {}
        self._load()

    def _load(self):
        """Load papers from the JSON file."""
        if not self.db_path.exists():
            self._papers = {}
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._papers = {}
        for paper_data in data.get("papers", []):
            paper = Paper(**paper_data)
            self._papers[paper.id] = paper

    def _save(self):
        """Save papers to JSON file with atomic write."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "updated": datetime.now(timezone.utc).isoformat(),
            "count": len(self._papers),
            "papers": [asdict(p) for p in self._papers.values()],
        }

        # Atomic write: write to temp file, then rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.db_path.parent, suffix=".tmp"
        )
        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # On Windows, we need to remove the target first
            if self.db_path.exists():
                self.db_path.unlink()
            Path(temp_path).rename(self.db_path)
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def add(self, paper: Paper) -> Paper:
        """Add a paper to the store. Returns the paper with its assigned ID."""
        self._papers[paper.id] = paper
        self._save()
        return paper

    def add_from_search_result(self, result: dict) -> Paper:
        """Create and add a paper from a search result dict."""
        paper = Paper.from_search_result(result)
        return self.add(paper)

    def get(self, paper_id: str) -> Optional[Paper]:
        """Get a paper by ID."""
        return self._papers.get(paper_id)

    def get_all(self) -> list[Paper]:
        """Get all papers."""
        return list(self._papers.values())

    def update(self, paper: Paper) -> Paper:
        """Update an existing paper."""
        if paper.id not in self._papers:
            raise KeyError(f"Paper {paper.id} not found")
        self._papers[paper.id] = paper
        self._save()
        return paper

    def delete(self, paper_id: str) -> bool:
        """Delete a paper by ID. Returns True if deleted."""
        if paper_id in self._papers:
            del self._papers[paper_id]
            self._save()
            return True
        return False

    def find_by_tag(self, tag: str) -> list[Paper]:
        """Find papers that have a specific tag."""
        return [p for p in self._papers.values() if tag in p.tags]

    def find_by_category(self, category: str) -> list[Paper]:
        """Find papers that have a specific category."""
        return [p for p in self._papers.values() if category in p.categories]

    def find_by_title(self, title_substring: str) -> list[Paper]:
        """Find papers whose title contains the given substring (case-insensitive)."""
        needle = title_substring.lower()
        return [p for p in self._papers.values() if needle in p.title.lower()]

    def find_by_read_status(self, status: str) -> list[Paper]:
        """Find papers by read status."""
        return [p for p in self._papers.values() if p.read_status == status]

    def has_paper_with_source_id(self, source: str, source_id: str) -> bool:
        """Check if a paper with the given source ID already exists."""
        for paper in self._papers.values():
            if paper.source_ids.get(source) == source_id:
                return True
        return False

    def count(self) -> int:
        """Return the number of papers in the store."""
        return len(self._papers)

    def merge_search_results(self, results: list[dict]) -> tuple[int, int]:
        """Add multiple search results, skipping duplicates.

        Returns:
            Tuple of (added_count, skipped_count).
        """
        added = 0
        skipped = 0

        for result in results:
            # Check for existing paper by source IDs
            is_duplicate = False
            source_ids = result.get("source_ids", {})
            for source, sid in source_ids.items():
                if self.has_paper_with_source_id(source, sid):
                    is_duplicate = True
                    break

            if is_duplicate:
                skipped += 1
            else:
                self.add_from_search_result(result)
                added += 1

        return added, skipped
