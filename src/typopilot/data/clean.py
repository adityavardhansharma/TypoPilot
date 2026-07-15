"""Cleaning and filtering for clean-target sentences (idea.md section 9).

``clean_sentence`` returns either a cleaned sentence or a rejection reason,
so the pipeline can report exactly why every dropped example was dropped.
"""

from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_FILE_PATH = re.compile(r"[A-Za-z]:\\|/(?:usr|home|var|etc)/|\.(?:py|js|exe|dll|json|xml)\b")
_HTML_MARKDOWN = re.compile(r"<[a-zA-Z/][^>]*>|```|\[.+\]\(.+\)|^#{1,6}\s", re.MULTILINE)
_CODE_CHARS = re.compile(r"[{}<>=_|\\^~`]")
_PHONE_LONG_NUMBER = re.compile(r"\d{7,}")
_MULTI_SPACE = re.compile(r"  +")

# Pilot scope: everyday-message length sentences.
MIN_WORDS = 3
MAX_WORDS = 30


def normalise(text: str) -> str:
    """Unicode + whitespace + quote normalisation (not error correction)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return _MULTI_SPACE.sub(" ", text).strip()


def clean_sentence(text: str) -> tuple[str | None, str]:
    """Return (cleaned_text, "ok") or (None, rejection_reason)."""
    text = normalise(text)
    if not text:
        return None, "empty"

    words = text.split()
    if len(words) < MIN_WORDS:
        return None, "too_short"
    if len(words) > MAX_WORDS:
        return None, "too_long"
    if _URL.search(text):
        return None, "url"
    if _EMAIL.search(text):
        return None, "email"
    if _FILE_PATH.search(text):
        return None, "file_path"
    if _HTML_MARKDOWN.search(text):
        return None, "markup"
    if len(_CODE_CHARS.findall(text)) >= 2:
        return None, "code_like"
    if _PHONE_LONG_NUMBER.search(text):
        return None, "long_number"

    alpha = sum(ch.isalpha() for ch in text)
    if alpha / len(text) < 0.6:
        return None, "low_alpha_ratio"
    if not all(ord(ch) < 128 for ch in text):
        return None, "non_ascii"

    return text, "ok"


def dedup_key(text: str) -> str:
    """Key for exact-duplicate detection: casefold + strip punctuation/space.

    Deduplication runs on the CLEAN side before any corruption or splitting,
    so the same source sentence can never appear in both train and test.
    """
    return re.sub(r"[^a-z0-9]", "", text.casefold())
