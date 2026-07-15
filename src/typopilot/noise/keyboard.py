"""QWERTY keyboard geometry and adjacency map.

Nearby-key substitution errors ("given" -> "giben") must come from keys that
are physically adjacent on a real keyboard, not from random characters.
The adjacency map is derived from key coordinates on a standard staggered
QWERTY layout so it stays correct and symmetric by construction.
"""

from __future__ import annotations

# (row_string, x_offset_in_key_widths). Row spacing is 1.0 key width.
# Offsets reflect the physical stagger of a standard ANSI QWERTY board.
_ROWS: list[tuple[str, float]] = [
    ("1234567890", 1.0),
    ("qwertyuiop", 1.5),
    ("asdfghjkl", 1.75),
    ("zxcvbnm", 2.25),
]

# Keys within this euclidean distance (in key widths) count as neighbours.
_ADJACENCY_THRESHOLD = 1.3


def _build_adjacency() -> dict[str, str]:
    coords: dict[str, tuple[float, float]] = {}
    for row_index, (row, offset) in enumerate(_ROWS):
        for col_index, key in enumerate(row):
            coords[key] = (float(row_index), offset + col_index)

    adjacency: dict[str, str] = {}
    for key, (ky, kx) in coords.items():
        neighbours = [
            other
            for other, (oy, ox) in coords.items()
            if other != key
            and ((ky - oy) ** 2 + (kx - ox) ** 2) ** 0.5 <= _ADJACENCY_THRESHOLD
        ]
        adjacency[key] = "".join(sorted(neighbours))
    return adjacency


#: Lowercase key -> string of physically adjacent lowercase keys.
QWERTY_ADJACENCY: dict[str, str] = _build_adjacency()


def nearby_keys(char: str) -> str:
    """Return the neighbours of *char*, preserving its case.

    Returns an empty string for characters not on the letter/digit grid
    (punctuation, whitespace, non-ASCII), which callers treat as
    "this character cannot take a nearby-key error".
    """
    lower = char.lower()
    neighbours = QWERTY_ADJACENCY.get(lower, "")
    if char.isupper():
        return neighbours.upper()
    return neighbours
