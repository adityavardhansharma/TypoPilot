"""Detokenizer for pre-tokenized corpora (BEA M2, DailyDialog).

Both corpora store text as space-separated tokens ("a few beers , please ?").
Training targets must be natural text, otherwise the model learns to emit
spaced-out punctuation.
"""

from __future__ import annotations

import re

# Punctuation that attaches to the preceding token.
_ATTACH_LEFT = {".", ",", "?", "!", ";", ":", "%", ")", "]", "}", "...", "'"}
# Tokens that attach to the following token.
_ATTACH_RIGHT = {"(", "[", "{", "$", "#"}
# Clitics that attach to the preceding word: do n't -> don't, it 's -> it's.
_CLITICS = {"n't", "'s", "'re", "'m", "'ve", "'ll", "'d", "'em"}


def detokenize(tokens: list[str] | str) -> str:
    if isinstance(tokens, str):
        tokens = tokens.split()

    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        if out and (token in _ATTACH_LEFT or token.lower() in _CLITICS):
            out[-1] += token
        elif out and out[-1] and out[-1][-1] in _ATTACH_RIGHT:
            out[-1] += token
        else:
            out.append(token)
    text = " ".join(out)

    # Tokenised hyphens and slashes: "well - known" -> "well-known".
    text = re.sub(r"(\w) ([-/]) (\w)", r"\1\2\3", text)
    # Collapse any leftover double spaces.
    text = re.sub(r"  +", " ", text)
    return text.strip()
