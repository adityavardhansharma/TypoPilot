"""Rule-based spacing/capitalisation baseline (idea.md section 19).

Deliberately simple: fixes only mechanical whitespace, sentence-start
capitalisation, and standalone-i issues. No dictionary, no reordering.
"""

from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"  +")
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,?!;:])")
_NO_SPACE_AFTER_PUNCT = re.compile(r"([.,?!;:])([A-Za-z])")
_SENTENCE_START = re.compile(r"(^|[.?!]\s+)([a-z])")
_LONE_I = re.compile(r"\b(i)\b")


def rule_correct(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text.strip())
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _NO_SPACE_AFTER_PUNCT.sub(r"\1 \2", text)
    text = _LONE_I.sub("I", text)
    text = _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    return text
