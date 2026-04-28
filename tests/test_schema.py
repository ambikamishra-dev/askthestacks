"""Tests for the database entry and corpus schema."""

import pytest
from pydantic import ValidationError

from askthestacks.schema import Corpus, DatabaseEntry


VALID_URL = "https://www.wiu.edu/library/direct/?ASC"


def _make_entry(**overrides) -> DatabaseEntry:
    defaults = {
        "code": "ASC",
        "name": "Academic Search Complete",
        "url": VALID_URL,
    }
    return DatabaseEntry(**(defaults | overrides))


class TestDatabaseEntry:
    def test_minimal_entry_constructs(self):
        entry = _make_entry()
        assert entry.code == "ASC"
        assert entry.name == "Academic Search Complete"

    def test_name_is_stripped(self):
        entry = _make_entry(name="  Spaced Out  ")
        assert entry.name == "Spaced Out"

    def test_subjects_are_deduped_and_sorted(self):
        entry = _make_entry(
            subjects=["Psychology", "Education", "Psychology", "  "])
        assert entry.subjects == ["Education", "Psychology"]

    def test_embedding_text_includes_name(self):
        entry = _make_entry()
        assert "Academic Search Complete" in entry.embedding_text

    def test_embedding_text_includes_subject_hint(self):
        entry = _make_entry(subject_hint="Multi-Disciplinary")
        assert "Multi-Disciplinary" in entry.embedding_text

    def test_embedding_text_includes_coverage(self):
        entry = _make_entry(coverage="indexes 5800+ journals")
        assert "5800+ journals" in entry.embedding_text

    def test_embedding_text_excludes_no_full_text(self):
        entry = _make_entry(full_text="No")
        assert "Full text" not in entry.embedding_text

    def test_rejects_non_wiu_url(self):
        with pytest.raises(ValidationError, match="WIU database URL"):
            _make_entry(url="https://example.com/oops")

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            _make_entry(name="")

    def test_rejects_missing_url(self):
        with pytest.raises(ValidationError):
            DatabaseEntry(code="X", name="No URL DB")  # type: ignore[call-arg]


class TestCorpus:
    def test_corpus_constructs_from_entries(self):
        entries = [_make_entry(code="A"), _make_entry(code="B")]
        corpus = Corpus(entries=entries)
        assert corpus.entry_count == 2

    def test_corpus_rejects_duplicate_codes(self):
        entries = [_make_entry(code="DUP"), _make_entry(code="DUP")]
        with pytest.raises(ValidationError, match="duplicate codes"):
            Corpus(entries=entries)

    def test_corpus_round_trips_through_json(self):
        entries = [_make_entry(code="A"), _make_entry(
            code="B", subject_hint="Education")]
        corpus = Corpus(entries=entries)
        serialized = corpus.model_dump_json()
        rebuilt = Corpus.model_validate_json(serialized)
        assert rebuilt.entry_count == 2
        assert rebuilt.entries[1].subject_hint == "Education"
        assert rebuilt.entries[1].embedding_text == corpus.entries[1].embedding_text
