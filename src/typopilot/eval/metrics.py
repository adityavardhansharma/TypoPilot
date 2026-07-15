"""Evaluation metrics (idea.md section 13).

All metrics are computed on raw text — capitalisation and punctuation are
part of the correction task, so no normalisation is applied before scoring.

Multi-reference support: JFLEG provides four human corrections per sentence;
`*_multi` variants score against the closest reference.
"""

from __future__ import annotations


def levenshtein(a: list | str, b: list | str) -> int:
    """Edit distance over characters (str inputs) or tokens (list inputs)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (item_a != item_b),
            ))
        previous = current
    return previous[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate: char edit distance / reference length."""
    if not reference:
        return float(bool(hypothesis))
    return levenshtein(hypothesis, reference) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    """Word error rate: token edit distance / reference token count."""
    ref_tokens = reference.split()
    if not ref_tokens:
        return float(bool(hypothesis.split()))
    return levenshtein(hypothesis.split(), ref_tokens) / len(ref_tokens)


def exact_match(hypothesis: str, reference: str) -> bool:
    return hypothesis == reference


def cer_multi(hypothesis: str, references: list[str]) -> float:
    return min(cer(hypothesis, ref) for ref in references)


def wer_multi(hypothesis: str, references: list[str]) -> float:
    return min(wer(hypothesis, ref) for ref in references)


def exact_match_multi(hypothesis: str, references: list[str]) -> bool:
    return any(hypothesis == ref for ref in references)


def correction_rates(source: str, hypothesis: str, reference: str) -> dict:
    """Did the system move the text toward the reference, and how far?

    Returns:
      input_cer      — how broken the input was (baseline damage)
      output_cer     — how broken the output still is
      improvement    — input_cer - output_cer (positive = helped)
      overcorrected  — output changed things but ended up *further* from the
                       reference than the input was (worse than doing nothing)
      undercorrected — output is closer than input but still not exact
    """
    input_cer = cer(source, reference)
    output_cer = cer(hypothesis, reference)
    return {
        "input_cer": input_cer,
        "output_cer": output_cer,
        "improvement": input_cer - output_cer,
        "overcorrected": output_cer > input_cer,
        "undercorrected": 0 < output_cer <= input_cer,
    }


def aggregate(rows: list[dict]) -> dict:
    """Average a list of per-example metric dicts (numeric fields only)."""
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if isinstance(values[0], bool):
            result[key] = round(sum(values) / len(values), 4)
        elif isinstance(values[0], (int, float)):
            result[key] = round(sum(values) / len(values), 4)
    result["n"] = len(rows)
    return result
