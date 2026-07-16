"""Real-world misspelling dictionaries from commercially usable sources.

Two sources, merged into one ``{correct_word: [misspelling, ...]}`` map:

* client9/misspell ``words.go`` — MIT licence. A Go source file containing
  a flat string array of alternating ``"wrong", "right"`` literals.
* Wikipedia "Lists of common misspellings/For machines" — CC BY-SA.
  Lines of the form ``misspelling->correction`` (occasionally multiple
  corrections separated by commas; those are skipped as ambiguous).

Used by the noise generator to corrupt words the way real humans do
(``received -> recieved``), which pure keyboard noise never produces.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

_GO_STRING_PAIR = re.compile(r'"([^"]+)",\s*"([^"]+)",')
_WIKI_LINE = re.compile(r"^([A-Za-z']+)->([A-Za-z' ]+)$")


def load_misspell_go(path: str | Path) -> dict[str, list[str]]:
    """Parse client9/misspell words.go into {correct: [wrong, ...]}."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    mapping: dict[str, list[str]] = defaultdict(list)
    for wrong, right in _GO_STRING_PAIR.findall(text):
        if wrong.isalpha() and right.isalpha() and wrong.lower() != right.lower():
            mapping[right.lower()].append(wrong.lower())
    return dict(mapping)


def load_wikipedia_list(path: str | Path) -> dict[str, list[str]]:
    """Parse the Wikipedia machine-readable misspelling list (HTML page).

    The list lives inside <pre> blocks as ``misspelling->correction`` lines.
    Entries with multiple comma-separated corrections are ambiguous
    (context-dependent) and are skipped.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    mapping: dict[str, list[str]] = defaultdict(list)
    for raw_line in text.splitlines():
        match = _WIKI_LINE.match(raw_line.strip())
        if not match:
            continue
        wrong, right = match.group(1).lower(), match.group(2).lower().strip()
        if "," in right or " " in wrong:
            continue
        if wrong != right:
            mapping[right].append(wrong)
    return dict(mapping)


def load_combined(misspell_go: str | Path, wikipedia_html: str | Path) -> dict[str, list[str]]:
    combined: dict[str, list[str]] = defaultdict(list)
    for source in (load_misspell_go(misspell_go), load_wikipedia_list(wikipedia_html)):
        for right, wrongs in source.items():
            for wrong in wrongs:
                if wrong not in combined[right]:
                    combined[right].append(wrong)
    return dict(combined)
