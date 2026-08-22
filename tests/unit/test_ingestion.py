"""
tests/unit/test_ingestion.py
-----------------------------
Unit tests for front-matter parsing and chunk boundary correctness.

All tests run fully offline — no network access, no Gemini API calls.
Knowledge-base files are read directly from the real corpus; no mocking
is used so tests serve as both unit and integration validation of the parser.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from app.ingestion.models import Chunk, ChunkMetadata
from app.ingestion.parser import (
    _parse_front_matter,
    _slugify,
    _split_into_sections,
    parse_directory,
    parse_file,
)

# ---------------------------------------------------------------------------
# Project root — used to locate knowledge-base files
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_DIR = _REPO_ROOT / "knowledge-base"


# ===========================================================================
# _slugify
# ===========================================================================

class TestSlugify:
    def test_lowercase_and_replace_spaces(self):
        assert _slugify("Standard Return Window") == "standard_return_window"

    def test_strips_leading_trailing_underscores(self):
        assert _slugify("  Bags and backpacks  ") == "bags_and_backpacks"

    def test_collapses_consecutive_separators(self):
        assert _slugify("Breeze Tumbler — Product Info") == "breeze_tumbler_product_info"

    def test_empty_string(self):
        assert _slugify("") == ""


# ===========================================================================
# _parse_front_matter
# ===========================================================================

class TestParseFrontMatter:
    def test_round_trips_simple_yaml(self):
        raw = textwrap.dedent("""\
            ---
            document_id: TEST-001
            title: Test Doc
            status: active
            ---

            Body text here.
        """)
        fm, body = _parse_front_matter(raw)
        assert fm["document_id"] == "TEST-001"
        assert fm["title"] == "Test Doc"
        assert "Body text here." in body

    def test_raises_on_missing_front_matter(self):
        with pytest.raises(ValueError, match="No valid YAML front-matter"):
            _parse_front_matter("No front matter here.\n")

    def test_optional_fields_absent(self):
        raw = textwrap.dedent("""\
            ---
            document_id: X-001
            title: Minimal
            status: active
            audience: customer
            policy_authority: official
            ---
            Content.
        """)
        fm, _ = _parse_front_matter(raw)
        assert "supersedes" not in fm
        assert "customer_answering" not in fm


# ===========================================================================
# Module-level fixtures (avoid class-scoped instance method deprecation)
# ===========================================================================

@pytest.fixture(scope="module")
def chunks_doc01():
    return parse_file(_KB_DIR / "01-returns-policy-current.md")

@pytest.fixture(scope="module")
def chunks_doc02():
    return parse_file(_KB_DIR / "02-returns-policy-legacy.md")

@pytest.fixture(scope="module")
def chunks_doc13():
    return parse_file(_KB_DIR / "13-support-escalation.md")

@pytest.fixture(scope="module")
def chunks_doc14():
    return parse_file(_KB_DIR / "14-internal-content-migration-notes.md")

@pytest.fixture(scope="module")
def all_chunks():
    return parse_directory(_KB_DIR)


# ===========================================================================
# parse_file — real corpus documents
# ===========================================================================

class TestParseFileDoc01:
    """Tests against 01-returns-policy-current.md (RET-2026-01, active+official)."""

    def test_produces_multiple_chunks(self, chunks_doc01):
        assert len(chunks_doc01) >= 3, "Expected at least 3 heading-level sections"

    def test_required_metadata_fields(self, chunks_doc01):
        for chunk in chunks_doc01:
            m = chunk.metadata
            assert m.document_id == "RET-2026-01"
            assert m.title == "Returns Policy"
            assert m.status == "active"
            assert m.audience == "customer"
            assert m.policy_authority == "official"
            assert m.filename == "01-returns-policy-current.md"

    def test_supersedes_field_present(self, chunks_doc01):
        # Doc 01 is the only doc with a 'supersedes' field
        assert all(c.metadata.supersedes == "RET-2024-01" for c in chunks_doc01)

    def test_customer_answering_defaults_true(self, chunks_doc01):
        # No explicit customer_answering field in doc 01 → must default to True
        assert all(c.metadata.customer_answering is True for c in chunks_doc01)

    def test_chunk_ids_unique(self, chunks_doc01):
        ids = [c.chunk_id for c in chunks_doc01]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"

    def test_chunk_ids_use_stem(self, chunks_doc01):
        for c in chunks_doc01:
            assert c.chunk_id.startswith("01-returns-policy-current__")

    def test_heading_text_present(self, chunks_doc01):
        headings = [c.metadata.heading for c in chunks_doc01]
        # Should contain "Standard return window" section
        assert any("standard return window" in h.lower() for h in headings)

    def test_chunk_text_not_empty(self, chunks_doc01):
        for c in chunks_doc01:
            assert c.text.strip(), f"Empty text for chunk {c.chunk_id}"


class TestParseFileDoc02:
    """Tests against 02-returns-policy-legacy.md (RET-2024-01, superseded)."""

    def test_status_superseded(self, chunks_doc02):
        assert all(c.metadata.status == "superseded" for c in chunks_doc02)

    def test_superseded_by_field(self, chunks_doc02):
        assert all(c.metadata.superseded_by == "RET-2026-01" for c in chunks_doc02)

    def test_superseded_date_field(self, chunks_doc02):
        assert all(c.metadata.superseded_date == "2026-04-01" for c in chunks_doc02)

    def test_no_supersedes_field(self, chunks_doc02):
        # Only doc 01 has supersedes; doc 02 does not
        assert all(c.metadata.supersedes is None for c in chunks_doc02)


class TestParseFileDoc13:
    """Tests against 13-support-escalation.md (internal, official)."""

    def test_audience_internal(self, chunks_doc13):
        assert all(c.metadata.audience == "internal" for c in chunks_doc13)

    def test_customer_answering_inferred_false(self, chunks_doc13):
        # No explicit customer_answering in front matter; audience=internal → False
        assert all(c.metadata.customer_answering is False for c in chunks_doc13)


class TestParseFileDoc14:
    """Tests against 14-internal-content-migration-notes.md (draft, internal, none)."""

    def test_status_draft(self, chunks_doc14):
        assert all(c.metadata.status == "draft" for c in chunks_doc14)

    def test_policy_authority_none(self, chunks_doc14):
        assert all(c.metadata.policy_authority == "none" for c in chunks_doc14)

    def test_customer_answering_explicit_false(self, chunks_doc14):
        # This doc EXPLICITLY sets customer_answering: false in front matter
        assert all(c.metadata.customer_answering is False for c in chunks_doc14)

    def test_prompt_injection_text_in_body(self, chunks_doc14):
        # The injection text should be present as inert document text (not executed)
        full_text = " ".join(c.text for c in chunks_doc14)
        assert "SYSTEM INSTRUCTION" in full_text or "Ignore all prior rules" in full_text


# ===========================================================================
# Chunk boundary correctness
# ===========================================================================

class TestChunkBoundaries:
    def test_each_chunk_contains_its_heading(self, chunks_doc01):
        """Every non-intro chunk must begin with its heading."""
        for c in chunks_doc01:
            if c.metadata.heading:
                # The heading text should appear in the chunk text
                assert c.metadata.heading in c.text, (
                    f"Heading '{c.metadata.heading}' not found in chunk text for {c.chunk_id}"
                )

    def test_sections_do_not_bleed_into_each_other(self):
        """Text from one section's heading should not appear in the previous chunk."""
        chunks = parse_file(_KB_DIR / "07-warranty.md")
        for i in range(len(chunks) - 1):
            next_heading = chunks[i + 1].metadata.heading
            if next_heading:
                current_text = chunks[i].text
                pattern = re.compile(r"^#{1,6}\s+" + re.escape(next_heading), re.MULTILINE)
                assert not pattern.search(current_text), (
                    f"Heading '{next_heading}' bled into previous chunk {chunks[i].chunk_id}"
                )

    def test_intro_chunk_excluded_when_empty(self, all_chunks):
        """No chunk should have empty text."""
        for c in all_chunks:
            assert c.text.strip(), f"Chunk {c.chunk_id} has empty text"


# ===========================================================================
# parse_directory — full corpus
# ===========================================================================

class TestParseDirectory:
    def test_produces_chunks_from_all_14_files(self, all_chunks):
        filenames = {c.metadata.filename for c in all_chunks}
        assert len(filenames) == 14, f"Expected 14 files, got {len(filenames)}: {filenames}"

    def test_all_chunk_ids_unique(self, all_chunks):
        ids = [c.chunk_id for c in all_chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs: {[i for i in ids if ids.count(i) > 1]}"

    def test_chunk_count_reasonable(self, all_chunks):
        # 14 files × average ~4 sections → expect at least 30 total chunks
        assert len(all_chunks) >= 30, f"Suspiciously few chunks: {len(all_chunks)}"

    def test_doc11_and_doc12_both_active_official(self, all_chunks):
        """CORPUS_FACTS.md confirms both docs are active+official — verify here."""
        doc11_chunks = [c for c in all_chunks if c.metadata.filename == "11-product-care.md"]
        doc12_chunks = [c for c in all_chunks if c.metadata.filename == "12-breeze-tumbler-product-card.md"]
        assert doc11_chunks, "No chunks from 11-product-care.md"
        assert doc12_chunks, "No chunks from 12-breeze-tumbler-product-card.md"
        for c in doc11_chunks:
            assert c.metadata.status == "active"
            assert c.metadata.policy_authority == "official"
        for c in doc12_chunks:
            assert c.metadata.status == "active"
            assert c.metadata.policy_authority == "official"
