"""
app/ingestion/parser.py
-----------------------
Parser for the knowledge-base markdown files.

Responsibilities:
1. Parse YAML front matter (between leading '---' delimiters).
2. Split the body into per-heading chunks using ATX-style headings (# ## ###).
3. Produce a list of Chunk objects with correct ChunkMetadata.

Heading-splitting rules:
- The text before the first heading is the "intro" block.
  It is dropped if empty (most docs start immediately with a heading).
- Each heading (any level) starts a new chunk.
- chunk_id = "<filename_stem>__<heading_slug>"
  where heading_slug = heading lowercased, non-alphanumeric chars → '_',
  multiple underscores collapsed, leading/trailing underscores stripped.
- For the intro block: heading = "", heading_slug = "intro".

customer_answering inference rule (from CORPUS_FACTS.md):
- Use the value from front matter when present.
- When absent, default to True — UNLESS the parsed audience is "internal"
  OR policy_authority is "none", in which case we default to False as an
  additional safety guard even when the field is not explicitly set.
  (Doc 13 and doc 14 are internal/none respectively and must never be
  served as customer-facing answers.)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml  # PyYAML

from app.ingestion.models import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def _slugify(text: str) -> str:
    """Convert a heading string to a URL/identifier-safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Return (front_matter_dict, body_text).

    Strips the leading YAML block (between '---') from *raw* and returns
    the parsed front-matter as a dict and the remaining body as a string.
    Raises ValueError if no valid front-matter block is found.
    """
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError("No valid YAML front-matter block found")
    fm = yaml.safe_load(m.group(1)) or {}
    body = raw[m.end():]
    return fm, body


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split body text into (heading, section_text) pairs.

    The heading for the intro block (text before first heading) is ''.
    Each subsequent section includes the heading line itself in section_text.
    Returns list of (heading_text, section_text).
    """
    sections: list[tuple[str, str]] = []
    positions = [(m.start(), m.group(2), m.group(0)) for m in _HEADING_RE.finditer(body)]

    if not positions:
        # No headings at all — entire body is one intro block
        if body.strip():
            sections.append(("", body.strip()))
        return sections

    # Text before first heading (intro block)
    intro = body[: positions[0][0]].strip()
    if intro:
        sections.append(("", intro))

    for i, (start, heading, full_heading_line) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        section_text = body[start:end].strip()
        sections.append((heading, section_text))

    return sections


def _infer_customer_answering(
    fm: dict,
    audience: str,
    policy_authority: str,
) -> bool:
    """Determine the effective customer_answering value.

    Priority:
    1. Explicit front-matter value (if present).
    2. Safety default: False when audience='internal' OR policy_authority='none'.
    3. Default: True (per CORPUS_FACTS.md design rule).
    """
    if "customer_answering" in fm:
        return bool(fm["customer_answering"])
    if audience == "internal" or policy_authority == "none":
        return False
    return True


def parse_file(path: Path) -> list[Chunk]:
    """Parse a single knowledge-base markdown file into a list of Chunk objects.

    Parameters
    ----------
    path:
        Absolute or relative path to the .md file.

    Returns
    -------
    list[Chunk]
        One Chunk per heading section (plus an intro chunk if present).
        The list is never empty for a well-formed file; a file with no
        headings produces a single intro chunk.
    """
    raw = path.read_text(encoding="utf-8")
    fm, body = _parse_front_matter(raw)

    filename = path.name
    stem = path.stem  # e.g. "01-returns-policy-current"

    # Extract required fields (KeyError propagates as a clear parse error)
    document_id: str = str(fm["document_id"])
    title: str = str(fm["title"])
    status: str = str(fm["status"])
    audience: str = str(fm["audience"])
    policy_authority: str = str(fm["policy_authority"])

    # Optional fields
    effective_date: Optional[str] = str(fm["effective_date"]) if "effective_date" in fm else None
    last_reviewed: Optional[str] = str(fm["last_reviewed"]) if "last_reviewed" in fm else None
    supersedes: Optional[str] = str(fm["supersedes"]) if "supersedes" in fm else None
    superseded_by: Optional[str] = str(fm["superseded_by"]) if "superseded_by" in fm else None
    superseded_date: Optional[str] = str(fm["superseded_date"]) if "superseded_date" in fm else None

    customer_answering = _infer_customer_answering(fm, audience, policy_authority)

    sections = _split_into_sections(body)
    chunks: list[Chunk] = []

    for heading, text in sections:
        slug = _slugify(heading) if heading else "intro"
        chunk_id = f"{stem}__{slug}"

        metadata = ChunkMetadata(
            filename=filename,
            document_id=document_id,
            title=title,
            heading=heading,
            status=status,
            effective_date=effective_date,
            last_reviewed=last_reviewed,
            audience=audience,
            policy_authority=policy_authority,
            customer_answering=customer_answering,
            supersedes=supersedes,
            superseded_by=superseded_by,
            superseded_date=superseded_date,
        )
        chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))

    return chunks


def parse_directory(directory: Path) -> list[Chunk]:
    """Parse all .md files in *directory* and return combined list of Chunks.

    Files are processed in sorted filename order so results are deterministic.
    """
    all_chunks: list[Chunk] = []
    for md_file in sorted(directory.glob("*.md")):
        all_chunks.extend(parse_file(md_file))
    return all_chunks
