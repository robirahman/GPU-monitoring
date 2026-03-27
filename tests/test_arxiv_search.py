"""Tests for the arXiv search module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.search.arxiv_search import search_arxiv, fetch_paper_by_id, _normalize_result


def _make_mock_result(
    entry_id="http://arxiv.org/abs/2509.00300v1",
    title="Test Paper Title",
    authors=None,
    summary="This is a test abstract.",
    published=None,
    pdf_url="http://arxiv.org/pdf/2509.00300v1",
    categories=None,
    doi=None,
):
    """Create a mock arxiv.Result object."""
    result = MagicMock()
    result.entry_id = entry_id
    result.title = title
    result.summary = summary
    result.pdf_url = pdf_url
    result.doi = doi
    result.categories = categories or ["cs.CR", "cs.LG"]

    if published is None:
        published = datetime(2025, 9, 1)
    result.published = published

    if authors is None:
        author1 = MagicMock()
        author1.name = "Alice Smith"
        author2 = MagicMock()
        author2.name = "Bob Jones"
        authors = [author1, author2]
    result.authors = authors

    return result


class TestNormalizeResult:
    def test_basic_normalization(self):
        mock = _make_mock_result()
        paper = _normalize_result(mock)

        assert paper["title"] == "Test Paper Title"
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert paper["abstract"] == "This is a test abstract."
        assert paper["source"] == "arxiv"
        assert paper["source_ids"]["arxiv"] == "2509.00300"
        assert paper["year"] == 2025

    def test_extracts_arxiv_id_without_version(self):
        mock = _make_mock_result(entry_id="http://arxiv.org/abs/2301.12345v3")
        paper = _normalize_result(mock)
        assert paper["source_ids"]["arxiv"] == "2301.12345"

    def test_preserves_categories(self):
        mock = _make_mock_result(categories=["cs.CR", "cs.LG", "cs.AI"])
        paper = _normalize_result(mock)
        assert paper["categories"] == ["cs.CR", "cs.LG", "cs.AI"]

    def test_handles_doi(self):
        mock = _make_mock_result(doi="10.1145/1234567.1234568")
        paper = _normalize_result(mock)
        assert paper["doi"] == "10.1145/1234567.1234568"


class TestSearchArxiv:
    @patch("src.search.arxiv_search.arxiv.Client")
    @patch("src.search.arxiv_search.arxiv.Search")
    def test_search_returns_normalized_papers(self, mock_search_cls, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.return_value = [
            _make_mock_result(title="Paper 1"),
            _make_mock_result(title="Paper 2"),
        ]

        results = search_arxiv("test query", max_results=10)

        assert len(results) == 2
        assert results[0]["title"] == "Paper 1"
        assert results[1]["title"] == "Paper 2"
        assert all(r["source"] == "arxiv" for r in results)


class TestFetchPaperById:
    @patch("src.search.arxiv_search.arxiv.Client")
    @patch("src.search.arxiv_search.arxiv.Search")
    def test_fetch_existing_paper(self, mock_search_cls, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.return_value = [_make_mock_result()]

        paper = fetch_paper_by_id("2509.00300")

        assert paper is not None
        assert paper["title"] == "Test Paper Title"

    @patch("src.search.arxiv_search.arxiv.Client")
    @patch("src.search.arxiv_search.arxiv.Search")
    def test_fetch_nonexistent_returns_none(self, mock_search_cls, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.return_value = []

        paper = fetch_paper_by_id("0000.00000")

        assert paper is None
