"""SymSpell baseline (idea.md section 19).

Uses ``lookup_compound``, which handles misspellings plus some split/merged
words in one pass against the bundled English frequency dictionary.
SymSpell lowercases and strips most punctuation, so we restore sentence-start
capitalisation and trailing punctuation afterwards to make the comparison
against ByT5 fair rather than trivially losing on formatting.
"""

from __future__ import annotations

import importlib.resources

from symspellpy import SymSpell, Verbosity


def _load_symspell() -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
    with importlib.resources.as_file(dictionary) as path:
        sym.load_dictionary(str(path), term_index=0, count_index=1)
    return sym


class SymSpellCorrector:
    def __init__(self) -> None:
        self._sym = _load_symspell()

    def __call__(self, text: str) -> str:
        if not text.strip():
            return text
        suggestions = self._sym.lookup_compound(
            text, max_edit_distance=2, transfer_casing=True
        )
        if not suggestions:
            return text
        corrected = suggestions[0].term

        # lookup_compound drops terminal punctuation; restore it.
        trailing = text.rstrip()
        if trailing and trailing[-1] in ".?!" and corrected and corrected[-1] not in ".?!":
            corrected += trailing[-1]
        # Restore sentence-start capitalisation lost by lowercased dictionary hits.
        if corrected and corrected[0].islower() and text and text[0].isalpha():
            corrected = corrected[0].upper() + corrected[1:]
        return corrected
