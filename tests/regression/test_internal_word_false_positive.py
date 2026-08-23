"""
tests/regression/test_internal_word_false_positive.py
------------------------------------------------------
Regression test: ensure the forbidden-field validator distinguishes between
legitimate descriptive English prose (e.g. "internal document", "internal policy")
and actual data field disclosures (e.g. "internal notes: ...", "risk_score: ...").
"""

from app.safety.trust import validate_response


def test_descriptive_internal_word_not_flagged():
    raw_output = "The migration note is an internal document and not authoritative policy."
    result = validate_response(raw_output, evidence_pack=[])
    # Must NOT raise forbidden field flags for descriptive prose
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) == 0
    assert "internal document" in result.cleaned_response


def test_internal_notes_field_disclosure_flagged():
    raw_output = "Your order status is shipped. internal notes: fraud review cleared."
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) == 1
    assert "internal notes" not in result.cleaned_response
    assert "fraud review cleared" not in result.cleaned_response


def test_risk_score_field_disclosure_flagged():
    raw_output = "Account verified. risk_score: 82."
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) == 1
    assert "risk_score" not in result.cleaned_response
    assert "82" not in result.cleaned_response
