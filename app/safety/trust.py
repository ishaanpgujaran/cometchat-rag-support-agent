"""
app/safety/trust.py
-------------------
Evidence formatting and response validation for the support agent.

Responsibilities
~~~~~~~~~~~~~~~~
1. ``format_evidence_pack`` — wraps ScoredEvidence chunks in clearly delimited
   ``<untrusted_evidence>`` blocks and SafeOrderResult in a ``<tool_result>``
   block.  These labels explicitly signal to the model that the content is
   DATA, not instructions.

2. ``validate_response`` — deterministically checks the raw model output for:
   (a) Invalid/hallucinated citations
   (b) Forbidden internal field names in the output text
   (c) Unsupported action claims (refund/cancel/replacement/address-change
       never actually executed — any such claim is flagged and replaced)
   (d) Silent conflict handling — if ConflictDetector flagged a conflict,
       the response MUST acknowledge it; if it silently picks one source,
       the conflict disclosure is prepended.

Escalation triggers (from Doc 13 — encoded as app logic, NOT from KB)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
human_handoff=True is set when:
  • Authoritative sources conflict (ConflictGroup list non-empty)
  • No authoritative evidence was found (evidence_pack empty)
  • Order lookup failed (found=False) or returned status="exception"
  • Customer requested cancel/refund/replacement/price-adj/warranty/address-change
    (detected by router; caller passes human_handoff_reason to signal this)
  • Safety/legal/fraud/internal-probe detected (same router flag)
  • Model output claims a completed action that no tool confirmed

Public API
~~~~~~~~~~
  format_evidence_pack(evidence, tool_result) -> str
  validate_response(raw, evidence, tool_result, conflicts,
                    force_handoff, handoff_reason) -> ValidationResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.policy.scoring import ScoredEvidence, is_authoritative
from app.policy.conflict import ConflictGroup
from app.orders.models import SafeOrderResult


# Forbidden field patterns (must never appear in customer-facing output)
# Note: "internal" is specifically targeted to data-disclosure contexts (e.g. "internal notes")
# so that descriptive prose like "internal document" or "internal policy" is not falsely flagged.
ALLOWED_INTERNAL_PHRASES: list[str] = [
    "internal document",
    "internal content",
    "internal policy",
    "internal migration",
    "internal material",
]

_FORBIDDEN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bemail\b", re.IGNORECASE),
    re.compile(r"\baddress\b", re.IGNORECASE),
    re.compile(r"\binternal\s+(?:note|notes|field|fields|data|record|score|tag|tags|flag|flags)\b", re.IGNORECASE),
    re.compile(r"\bnote[-_]?raw\b", re.IGNORECASE),
    re.compile(r"\bwarehouse_note\b|\bwarehousenote\b", re.IGNORECASE),
    re.compile(r"\brisk_score\b|\briskscore\b", re.IGNORECASE),
    re.compile(r"\bsupport_tags\b|\bsupporttags\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Unsupported completed-action claim patterns
# These signal that the model *claimed* it performed an action that no tool
# actually executed.  All such claims must be flagged and replaced.
# ---------------------------------------------------------------------------

_COMPLETED_ACTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b((?:i\s+have|i've|we\s+have|we've)\s+(?:refunded|cancell?ed|processed|approved|initiated|sent|created|opened|updated)|"
        r"(?:i\s+have|i've|we\s+have|we've)\s+(?:issued|processed|approved|initiated)\s+(?:your|the|a)?\s*(?:refund|replacement|cancellation|ticket|claim)|"
        r"cancell?ation\s+(?:has\s+been\s+confirmed|is\s+complete|is\s+confirmed)|"
        r"replacement\s+(?:is\s+on\s+its\s+way|has\s+been\s+sent|has\s+been\s+dispatched)|"
        r"(?:i|we)\s+(?:created|opened|filed|submitted|raised)\s+(?:a|the|your)\s+ticket|"
        r"warranty\s+claim\s+has\s+been\s+(?:submitted|approved|processed))\b",
        re.IGNORECASE,
    ),
]

_UNSUPPORTED_ACTION_REPLACEMENT = (
    "I'm not able to process that action — please contact our support team "
    "directly for assistance with cancellations, refunds, replacements, "
    "address changes, or warranty approvals."
)

# ---------------------------------------------------------------------------
# Citation pattern: filename#heading  (e.g. 01-returns-policy-current.md#Return Window)
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"(\d{2}-[\w-]+\.md#[^\s\]\)\"',;]+)")

CITATION_ARTIFACT_PATTERN: re.Pattern = re.compile(
    r'\s*\[(?:[^\]]*(?:\.md|#)[^\]]*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\]'
)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Result of validate_response().

    Attributes
    ----------
    is_valid : bool
        True when no flags were raised.
    human_handoff : bool
        True when the response requires escalation to a human agent.
    flags : list[str]
        Human-readable descriptions of each issue found.
    cleaned_response : str
        The (possibly modified) response text safe to return to the customer.
    """
    is_valid: bool
    human_handoff: bool
    flags: list[str] = field(default_factory=list)
    cleaned_response: str = ""


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

def format_evidence_pack(
    evidence: list[ScoredEvidence],
    tool_result: Optional[SafeOrderResult] = None,
) -> str:
    """
    Format retrieved evidence and sanitized tool results into labelled blocks.

    Each knowledge-base chunk is wrapped in::

        <untrusted_evidence source="filename#heading" score="0.83">
        ...chunk text...
        </untrusted_evidence>

    The ``untrusted_evidence`` label explicitly tells the model that this
    content is DATA supplied as context, not system instructions.

    A SafeOrderResult (if provided) is wrapped in::

        <tool_result source="order_lookup" order_id="ORD-XXXX">
        ...sanitized fields...
        </tool_result>

    Only customer-safe fields are included via model_dump(); internal fields
    can never appear here because SafeOrderResult's whitelist excludes them.

    Parameters
    ----------
    evidence:
        Scored evidence chunks (audience=internal chunks should already be
        filtered out by the orchestrator before calling this function).
    tool_result:
        Optional SafeOrderResult from lookup_order().

    Returns
    -------
    str
        A single string containing all formatted blocks, ready to embed in
        a prompt.
    """
    blocks: list[str] = []

    for ev in evidence:
        m = ev.chunk.metadata
        citation_key = f"{m.filename}#{m.heading}" if m.heading else m.filename
        score_str = f"{ev.final_score:.3f}"
        block = (
            f'<untrusted_evidence source="{citation_key}" score="{score_str}">\n'
            f"{ev.chunk.text}\n"
            f"</untrusted_evidence>"
        )
        blocks.append(block)

    if tool_result is not None:
        safe_dict = tool_result.model_dump(exclude_none=True)
        lines = "\n".join(f"  {k}: {v}" for k, v in safe_dict.items())
        block = (
            f'<tool_result source="order_lookup" order_id="{tool_result.order_id}">\n'
            f"{lines}\n"
            f"</tool_result>"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

def validate_response(
    raw_model_output: str,
    evidence_pack: list[ScoredEvidence],
    tool_result: Optional[SafeOrderResult] = None,
    conflict_groups: Optional[list[ConflictGroup]] = None,
    force_handoff: bool = False,
    handoff_reason: Optional[str] = None,
) -> ValidationResult:
    """
    Deterministically validate and clean a raw model response.

    Checks performed (in order)
    ----------------------------
    (a) Citation validation — any ``filename#heading`` citation must exist in
        the evidence pack.  Unknown citations are stripped from the text.
    (b) Forbidden field check — if any forbidden field name (email, address,
        internal, risk_score, note-raw, etc.) appears in the output, the
        offending sentence is replaced with a redaction notice.
    (c) Completed-action claim check — if the model claims it performed a
        refund, cancellation, replacement, ticket creation, or address change
        (none of which any tool actually executed), the claim is replaced with
        a refusal.
    (d) Conflict acknowledgement — if conflict_groups is non-empty, the
        response MUST mention both conflicting source filenames.  If it does
        not, a conflict disclosure header is prepended.

    Escalation triggers (Doc 13 rules)
    -----------------------------------
    human_handoff=True is set when:
    • force_handoff=True (caller detected an unsafe/unsupported route)
    • conflict_groups is non-empty (authoritative sources conflict)
    • evidence_pack is empty (insufficient KB evidence)
    • tool_result.found is False (order not found)
    • tool_result.status == "exception" (order in exception state)
    • A completed-action claim was found and replaced (flag (c) fired)

    Parameters
    ----------
    raw_model_output : str
        The raw text response from the Gemini model.
    evidence_pack : list[ScoredEvidence]
        The evidence used to ground the model's response (may be empty).
    tool_result : SafeOrderResult | None
        The sanitized order result, if an order lookup was performed.
    conflict_groups : list[ConflictGroup] | None
        Detected conflicts from ConflictDetector.detect() (may be empty/None).
    force_handoff : bool
        True when the router or orchestrator has already determined that
        human escalation is required (e.g., unsupported action route).
    handoff_reason : str | None
        Human-readable reason for the forced handoff (shown to agent, not customer).

    Returns
    -------
    ValidationResult
    """
    conflict_groups = conflict_groups or []
    flags: list[str] = []
    human_handoff: bool = force_handoff
    text = raw_model_output

    # Build valid citation set from evidence
    valid_citations: set[str] = set()
    for ev in evidence_pack:
        m = ev.chunk.metadata
        # Accept both "filename#heading" and "filename" forms
        valid_citations.add(m.filename)
        if m.heading:
            valid_citations.add(f"{m.filename}#{m.heading}")

    # ------------------------------------------------------------------
    # (a) Citation validation & artifact stripping
    # ------------------------------------------------------------------
    # Find all bracketed references [ ... ]
    bracketed_refs = re.findall(r"\[([^\]]+)\]", text)
    for raw_ref in bracketed_refs:
        cleaned_ref = raw_ref.strip()
        # If it looks like a citation reference (contains .md or #)
        if ".md" in cleaned_ref or "#" in cleaned_ref:
            filename_part = cleaned_ref.split("#")[0].strip()
            # Check if this citation matches any authoritative evidence chunk
            is_valid_cit = (
                cleaned_ref in valid_citations
                or filename_part in valid_citations
                or any(cleaned_ref == v or filename_part == v.split("#")[0] for v in valid_citations)
            )
            if not is_valid_cit:
                flags.append(f"Hallucinated citation stripped: '{cleaned_ref}'")
                # Remove the entire bracketed token including surrounding space
                text = re.sub(r"\s*\[\s*" + re.escape(raw_ref) + r"\s*\]", "", text)

    # Bare citation fallback (unbracketed)
    for citation in _CITATION_RE.findall(text):
        filename_part = citation.split("#")[0]
        if citation not in valid_citations and filename_part not in valid_citations:
            flags.append(f"Hallucinated citation stripped: '{citation}'")
            text = text.replace(citation, "")

    # Cleanup pass: remove empty brackets, artifact brackets, and collapse whitespace
    text = re.sub(r"\[\s{0,20}\]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # ------------------------------------------------------------------
    # (b) Forbidden field names
    # ------------------------------------------------------------------
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(text):
            field_name = pat.pattern.strip(r"\b")
            flags.append(f"Forbidden field name '{field_name}' found in output — sentence redacted.")
            # Remove the entire sentence containing the forbidden word
            text = _redact_sentence_containing(text, pat)

    # ------------------------------------------------------------------
    # (c) Unsupported completed-action claims
    # ------------------------------------------------------------------
    action_claim_found = False
    for pat in _COMPLETED_ACTION_PATTERNS:
        if pat.search(text):
            action_claim_found = True
            flags.append(
                "Model claimed a completed action that no tool confirmed "
                f"(pattern: {pat.pattern[:60]}…) — replaced with refusal."
            )
            # Replace the whole response with the refusal message to avoid
            # partial information leaking alongside the claim
            text = _UNSUPPORTED_ACTION_REPLACEMENT
            human_handoff = True
            break  # one replacement is enough

    # ------------------------------------------------------------------
    # (d) Conflict acknowledgement
    # ------------------------------------------------------------------
    if conflict_groups and not action_claim_found:
        for cg in conflict_groups:
            # Check whether both filenames are mentioned in the response
            doc_a_mentioned = cg.doc_a_filename in text
            doc_b_mentioned = cg.doc_b_filename in text
            if not (doc_a_mentioned and doc_b_mentioned):
                disclosure = (
                    f"⚠️ I found conflicting information between "
                    f"**{cg.doc_a_filename}** and **{cg.doc_b_filename}**: "
                    f"{cg.note} "
                    f"I recommend speaking with a support agent for a definitive answer.\n\n"
                )
                text = disclosure + text
                flags.append(
                    f"Conflict between {cg.doc_a_filename} and {cg.doc_b_filename} "
                    "was not acknowledged — disclosure prepended."
                )
            human_handoff = True  # Conflicts always require human handoff

    # ------------------------------------------------------------------
    # Escalation triggers (Doc 13 rules)
    # ------------------------------------------------------------------
    if (not evidence_pack and tool_result is None) or re.search(
        r"\b(?:do\s+not\s+have|don't\s+have|insufficient)\b.{0,30}\b(?:information|evidence|details)\b",
        text,
        re.IGNORECASE,
    ):
        human_handoff = True
        flags.append("Insufficient knowledge base information — human review required.")

    if tool_result is not None:
        if not tool_result.found:
            human_handoff = True
            flags.append(f"Order '{tool_result.order_id}' not found — human review required.")
        elif tool_result.status == "exception":
            human_handoff = True
            flags.append(
                f"Order '{tool_result.order_id}' is in exception status — human review required."
            )

    is_valid = len(flags) == 0

    return ValidationResult(
        is_valid=is_valid,
        human_handoff=human_handoff,
        flags=flags,
        cleaned_response=text.strip(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _redact_sentence_containing(text: str, pattern: re.Pattern) -> str:
    """
    Replace any sentence in *text* that contains a match for *pattern*
    with a generic redaction notice.

    Uses a simple heuristic: split on sentence-ending punctuation followed
    by whitespace.  Works well for English support responses.
    """
    # Split into sentences (rough heuristic)
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_re.split(text)
    cleaned: list[str] = []
    for sent in sentences:
        if pattern.search(sent):
            cleaned.append("[Redacted: internal field reference removed.]")
        else:
            cleaned.append(sent)
    return " ".join(cleaned)
