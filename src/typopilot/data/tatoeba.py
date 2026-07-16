"""Stream English sentences from the Tatoeba per-language export (CC-BY).

Format: bz2-compressed TSV with columns ``id<TAB>lang<TAB>text``.
Source: https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
Licence: CC-BY 2.0 FR (commercial use permitted with attribution).
"""

from __future__ import annotations

import bz2
from pathlib import Path
from typing import Iterator


def iter_sentences(path: str | Path) -> Iterator[str]:
    """Yield sentence text, streaming — the export is ~2M sentences."""
    with bz2.open(path, mode="rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) == 3 and parts[1] == "eng" and parts[2]:
                yield parts[2]
