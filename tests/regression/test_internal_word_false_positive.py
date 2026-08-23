"""
tests/regression/test_internal_word_false_positive.py
------------------------------------------------------
Regression tests: ensure the forbidden-field validator distinguishes between
legitimate descriptive English prose and actual data field disclosures.
Covers "internal", "address" (verb/noun vs field label), and "email"
(word vs actual email address).
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


# ---------------------------------------------------------------------------
# "address" false-positive regression tests
# ---------------------------------------------------------------------------

def test_address_verb_not_flagged():
    """'address' used as an English verb must NOT trigger redaction."""
    raw_output = (
        "Please address this issue within 7 calendar days of delivery "
        "with photos and your order details."
    )
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) == 0, f"False positive on 'address' verb: {forbidden_flags}"
    assert "7 calendar days" in result.cleaned_response


def test_address_manufacturing_defects_not_flagged():
    """'address' adjacent to 'manufacturing defects' must NOT trigger redaction."""
    raw_output = (
        "Manufacturing defects are addressed under the 2-year warranty policy "
        "for bags and backpacks."
    )
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) == 0, f"False positive on 'address' + 'manufacturing defects': {forbidden_flags}"
    assert "manufacturing defects" in result.cleaned_response.lower()


def test_address_customs_duties_not_flagged():
    """'address' in an international-shipping context must NOT trigger redaction."""
    raw_output = (
        "Import duties and taxes are not prepaid — the recipient must "
        "address any customs charges on delivery."
    )
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) == 0, f"False positive on 'address' in customs context: {forbidden_flags}"
    assert "duties" in result.cleaned_response


def test_shipping_address_field_label_flagged():
    """'shipping address:' as a field label MUST still be flagged."""
    raw_output = "Your shipping address: 220 King Street, Toronto, ON M5H 1K1."
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) >= 1, "shipping address field label should be flagged"


# ---------------------------------------------------------------------------
# "email" false-positive regression tests
# ---------------------------------------------------------------------------

def test_email_word_not_flagged():
    """The word 'email' used as a verb/noun must NOT trigger redaction."""
    raw_output = "Please email our support team at the address on our contact page."
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) == 0, f"False positive on 'email' word: {forbidden_flags}"


def test_actual_email_address_flagged():
    """An actual email address (containing @) MUST be flagged."""
    raw_output = "Customer email is ava.morgan@example.test — please do not share."
    result = validate_response(raw_output, evidence_pack=[])
    forbidden_flags = [f for f in result.flags if "Forbidden field" in f]
    assert len(forbidden_flags) >= 1, "Actual email address should be flagged"
    assert "ava.morgan@example.test" not in result.cleaned_response
