"""
tests/regression/test_citation_artifact_cleanup.py
--------------------------------------------------
Regression test: ensure stripping hallucinated citations leaves no display
artifacts (such as '[ ]' or '[ Word]'), while valid citations from the evidence
pack are preserved.
"""

from app.ingestion.models import Chunk, ChunkMetadata
from app.policy.scoring import ScoredEvidence
from app.safety.trust import validate_response


def _make_evidence(filename: str, heading: str) -> ScoredEvidence:
    chunk = Chunk(
        chunk_id=f"{filename}#{heading}",
        text="Sample policy text.",
        metadata=ChunkMetadata(
            filename=filename,
            heading=heading,
            document_id="DOC-1",
            title="Sample Title",
            status="active",
            policy_authority="official",
            audience="customer",
        ),
    )
    return ScoredEvidence(
        chunk=chunk,
        dense_score=0.9,
        bm25_score=0.9,
        metadata_bonus=0.0,
        metadata_penalty=0.0,
        final_score=0.9,
    )


def test_hallucinated_citation_leaves_no_artifacts():
    evidence = [_make_evidence("01-returns-policy-current.md", "Standard return window")]

    # 1. Output with a hallucinated citation token
    raw_output = "Your order is shipped [99-fake-policy.md#FakeHeading]."
    result = validate_response(raw_output, evidence_pack=evidence)
    assert "[99-fake-policy.md#FakeHeading]" not in result.cleaned_response
    assert "[" not in result.cleaned_response
    assert "]" not in result.cleaned_response
    assert result.cleaned_response == "Your order is shipped."

    # 2. Output with a hallucinated label/citation bracket artifact
    raw_output2 = "in transit with UPS [ Lookups#Order]"
    result2 = validate_response(raw_output2, evidence_pack=evidence)
    assert "[ Lookups#Order]" not in result2.cleaned_response
    assert "[" not in result2.cleaned_response
    assert "]" not in result2.cleaned_response


def test_valid_real_citation_not_stripped():
    evidence = [_make_evidence("01-returns-policy-current.md", "Standard return window")]

    raw_output = "Returns are accepted within 30 days [01-returns-policy-current.md#Standard return window]."
    result = validate_response(raw_output, evidence_pack=evidence)
    assert "[01-returns-policy-current.md#Standard return window]" in result.cleaned_response
    assert result.is_valid is True
    assert len(result.flags) == 0
