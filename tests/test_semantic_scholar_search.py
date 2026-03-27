"""Tests for the Semantic Scholar search module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.search.semantic_scholar_search import _normalize_result, search_semantic_scholar


def _make_mock_s2_result(
    title="Test Paper",
    abstract="A test abstract.",
    authors=None,
    year=2024,
    external_ids=None,
    citation_count=42,
    tldr=None,
    fields_of_study=None,
    publication_date=None,
):
    """Create a mock Semantic Scholar result object."""
    result = MagicMock()
    result.title = title
    result.abstract = abstract
    result.year = year
    result.citationCount = citation_count
    result.publicationDate = publication_date

    if authors is None:
        a1 = MagicMock()
        a1.name = "Alice Smith"
        a2 = MagicMock()
        a2.name = "Bob Jones"
        authors = [a1, a2]
    result.authors = authors

    if external_ids is None:
        external_ids = {"ArXiv": "2509.00300", "DOI": "10.1145/test", "CorpusId": 12345}
    result.externalIds = external_ids

    if tldr is None:
        tldr_obj = MagicMock()
        tldr_obj.text = "This paper does something interesting."
        tldr = tldr_obj
    result.tldr = tldr

    result.fieldsOfStudy = fields_of_study or ["Computer Science"]

    return result


class TestNormalizeResult:
    def test_basic_normalization(self):
        mock = _make_mock_s2_result()
        paper = _normalize_result(mock)

        assert paper["title"] == "Test Paper"
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert paper["abstract"] == "A test abstract."
        assert paper["year"] == 2024
        assert paper["source"] == "semantic_scholar"
        assert paper["citation_count"] == 42
        assert paper["source_ids"]["arxiv"] == "2509.00300"
        assert paper["source_ids"]["doi"] == "10.1145/test"

    def test_extracts_tldr(self):
        mock = _make_mock_s2_result()
        paper = _normalize_result(mock)
        assert paper["tldr"] == "This paper does something interesting."

    def test_handles_none_result(self):
        paper = _normalize_result(None)
        assert paper is None

    def test_handles_missing_title(self):
        mock = _make_mock_s2_result(title=None)
        paper = _normalize_result(mock)
        assert paper is None

    def test_builds_pdf_url_from_arxiv_id(self):
        mock = _make_mock_s2_result()
        paper = _normalize_result(mock)
        assert paper["pdf_url"] == "https://arxiv.org/pdf/2509.00300.pdf"

    def test_handles_no_external_ids(self):
        mock = _make_mock_s2_result(external_ids={})
        paper = _normalize_result(mock)
        assert paper["source_ids"] == {}
        assert paper["pdf_url"] is None


class TestSearchSemanticScholar:
    @patch("src.search.semantic_scholar_search.SemanticScholar")
    @patch("src.search.semantic_scholar_search._load_api_key", return_value=None)
    def test_search_returns_normalized_results(self, mock_load_key, mock_s2_cls):
        mock_s2 = MagicMock()
        mock_s2_cls.return_value = mock_s2

        mock_results = [
            _make_mock_s2_result(title="Paper A"),
            _make_mock_s2_result(title="Paper B"),
        ]
        mock_s2.search_paper.return_value = mock_results

        results = search_semantic_scholar("test query", max_results=10)

        assert len(results) == 2
        assert results[0]["title"] == "Paper A"
        assert results[1]["title"] == "Paper B"

    @patch("src.search.semantic_scholar_search.SemanticScholar")
    @patch("src.search.semantic_scholar_search._load_api_key", return_value=None)
    def test_search_handles_none_results(self, mock_load_key, mock_s2_cls):
        mock_s2 = MagicMock()
        mock_s2_cls.return_value = mock_s2
        mock_s2.search_paper.return_value = None

        results = search_semantic_scholar("nonexistent topic", max_results=10)
        assert results == []
