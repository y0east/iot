"""Helpers for matching user target text to detector class names."""

from __future__ import annotations

import re


_WORD_RE = re.compile(r"[a-z0-9]+")

_FILLER_WORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "please",
    "see",
    "find",
    "detect",
    "track",
    "target",
    "object",
}

_ALIASES = {
    "person": "person",
    "persons": "person",
    "people": "person",
    "human": "person",
    "humans": "person",
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "boy": "person",
    "boys": "person",
    "girl": "person",
    "girls": "person",
}


def canonical_detection_query(query: str) -> str:
    """Return a model-friendly query for generic class-only targets."""

    stripped = query.strip()
    terms = _query_terms(stripped)
    if terms and terms == {"person"}:
        return "person"
    return stripped


def query_matches_class(query: str, class_name: str) -> bool:
    """Return whether a natural-language query can map to a detector class."""

    query_terms = _query_terms(query)
    class_terms = _query_terms(class_name)
    if not query_terms or not class_terms:
        return False
    return bool(query_terms & class_terms)


def _query_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _WORD_RE.findall(text.lower()):
        if token in _FILLER_WORDS:
            continue
        terms.add(_ALIASES.get(token, token))
    return terms
