"""Tests for the paper store module."""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.storage.paper_store import Paper, PaperStore


@pytest.fixture
def temp_db(tmp_path):
    """Create a PaperStore with a temporary database file."""
    db_path = tmp_path / "test_papers.json"
    return PaperStore(db_path=db_path)


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return Paper(
        title="GPU Side-Channel Attacks: A Survey",
        authors=["Alice Smith", "Bob Jones"],
        abstract="This paper surveys GPU side-channel attacks.",
        year=2024,
        published="2024-01-15T00:00:00",
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        source_ids={"arxiv": "2401.12345", "doi": "10.1145/test"},
        categories=["cs.CR"],
        tags=["side-channel", "gpu-profiling"],
        citation_count=15,
    )


@pytest.fixture
def sample_search_result():
    """Create a sample search result dict."""
    return {
        "title": "ML Training Fingerprinting via Performance Counters",
        "authors": ["Carol White"],
        "abstract": "We show how to detect ML training using hardware performance counters.",
        "year": 2023,
        "published": "2023-06-01T00:00:00",
        "pdf_url": "https://arxiv.org/pdf/2306.00001.pdf",
        "source_ids": {"arxiv": "2306.00001"},
        "categories": ["cs.LG"],
        "citation_count": 8,
        "tldr": "Performance counters can detect ML training.",
        "doi": None,
    }


class TestPaper:
    def test_creates_id_automatically(self):
        paper = Paper(title="Test")
        assert paper.id
        assert len(paper.id) == 8

    def test_creates_added_date_automatically(self):
        paper = Paper(title="Test")
        assert paper.added_date

    def test_from_search_result(self, sample_search_result):
        paper = Paper.from_search_result(sample_search_result)
        assert paper.title == "ML Training Fingerprinting via Performance Counters"
        assert paper.authors == ["Carol White"]
        assert paper.year == 2023
        assert paper.source_ids["arxiv"] == "2306.00001"
        assert paper.tldr == "Performance counters can detect ML training."

    def test_default_read_status(self):
        paper = Paper(title="Test")
        assert paper.read_status == "unread"


class TestPaperStore:
    def test_add_and_get(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        retrieved = temp_db.get(sample_paper.id)
        assert retrieved is not None
        assert retrieved.title == sample_paper.title

    def test_count(self, temp_db, sample_paper):
        assert temp_db.count() == 0
        temp_db.add(sample_paper)
        assert temp_db.count() == 1

    def test_get_all(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        all_papers = temp_db.get_all()
        assert len(all_papers) == 1

    def test_update(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        sample_paper.notes = "Very relevant"
        sample_paper.read_status = "read"
        temp_db.update(sample_paper)

        retrieved = temp_db.get(sample_paper.id)
        assert retrieved.notes == "Very relevant"
        assert retrieved.read_status == "read"

    def test_update_nonexistent_raises(self, temp_db):
        paper = Paper(id="nonexistent", title="Ghost")
        with pytest.raises(KeyError):
            temp_db.update(paper)

    def test_delete(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        assert temp_db.delete(sample_paper.id) is True
        assert temp_db.count() == 0

    def test_delete_nonexistent(self, temp_db):
        assert temp_db.delete("nonexistent") is False

    def test_find_by_tag(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        results = temp_db.find_by_tag("side-channel")
        assert len(results) == 1
        assert results[0].title == sample_paper.title

        results = temp_db.find_by_tag("nonexistent-tag")
        assert len(results) == 0

    def test_find_by_category(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        results = temp_db.find_by_category("cs.CR")
        assert len(results) == 1

    def test_find_by_title(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        results = temp_db.find_by_title("side-channel")
        assert len(results) == 1

        results = temp_db.find_by_title("SIDE-CHANNEL")  # case insensitive
        assert len(results) == 1

    def test_find_by_read_status(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        results = temp_db.find_by_read_status("unread")
        assert len(results) == 1

    def test_has_paper_with_source_id(self, temp_db, sample_paper):
        temp_db.add(sample_paper)
        assert temp_db.has_paper_with_source_id("arxiv", "2401.12345") is True
        assert temp_db.has_paper_with_source_id("arxiv", "0000.00000") is False

    def test_persistence(self, tmp_path, sample_paper):
        db_path = tmp_path / "persist_test.json"

        # Write
        store1 = PaperStore(db_path=db_path)
        store1.add(sample_paper)

        # Read back in a new instance
        store2 = PaperStore(db_path=db_path)
        assert store2.count() == 1
        retrieved = store2.get(sample_paper.id)
        assert retrieved.title == sample_paper.title
        assert retrieved.authors == sample_paper.authors

    def test_add_from_search_result(self, temp_db, sample_search_result):
        paper = temp_db.add_from_search_result(sample_search_result)
        assert temp_db.count() == 1
        assert paper.title == "ML Training Fingerprinting via Performance Counters"

    def test_merge_search_results_skips_duplicates(self, temp_db, sample_search_result):
        added, skipped = temp_db.merge_search_results([sample_search_result])
        assert added == 1
        assert skipped == 0

        # Adding the same result again should be skipped
        added, skipped = temp_db.merge_search_results([sample_search_result])
        assert added == 0
        assert skipped == 1
        assert temp_db.count() == 1

    def test_json_file_format(self, tmp_path, sample_paper):
        db_path = tmp_path / "format_test.json"
        store = PaperStore(db_path=db_path)
        store.add(sample_paper)

        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"] == 1
        assert data["count"] == 1
        assert len(data["papers"]) == 1
        assert data["papers"][0]["title"] == sample_paper.title
