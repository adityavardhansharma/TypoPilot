"""Parser for the M2 grammatical-error-correction annotation format (BEA-2019).

An M2 block looks like::

    S My town is a medium size city .
    A 4 6|||R:OTHER|||medium-sized|||REQUIRED|||-NONE-|||0

``S`` is the tokenised source sentence; each ``A`` line is one edit:
``start end|||type|||replacement|||...|||annotator_id``. Applying all edits
from one annotator to the source yields the corrected sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .detokenize import detokenize


@dataclass(frozen=True)
class M2Example:
    source: str          # detokenised original (possibly errorful) sentence
    target: str          # detokenised corrected sentence
    edit_count: int      # number of edits applied
    edit_types: tuple[str, ...]


def _apply_edits(tokens: list[str], edits: list[tuple[int, int, str]]) -> list[str]:
    # Apply right-to-left so earlier offsets stay valid.
    corrected = list(tokens)
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        replacement_tokens = replacement.split() if replacement not in ("", "-NONE-") else []
        corrected[start:end] = replacement_tokens
    return corrected


def parse_m2(path: str | Path, annotator: int = 0) -> Iterator[M2Example]:
    """Yield one example per M2 block, using edits from *annotator* only."""
    source_tokens: list[str] | None = None
    edits: list[tuple[int, int, str]] = []
    edit_types: list[str] = []

    def flush() -> M2Example | None:
        if source_tokens is None:
            return None
        corrected = _apply_edits(source_tokens, edits)
        return M2Example(
            source=detokenize(source_tokens),
            target=detokenize(corrected),
            edit_count=len(edits),
            edit_types=tuple(edit_types),
        )

    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("S "):
                source_tokens = line[2:].split()
                edits, edit_types = [], []
            elif line.startswith("A "):
                fields = line[2:].split("|||")
                span, edit_type, replacement = fields[0], fields[1], fields[2]
                if int(fields[-1]) != annotator or edit_type == "noop":
                    continue
                start, end = (int(x) for x in span.split())
                if start == -1:
                    continue
                edits.append((start, end, replacement))
                edit_types.append(edit_type)
            elif not line.strip():
                example = flush()
                if example is not None:
                    yield example
                source_tokens = None
    example = flush()
    if example is not None:
        yield example
