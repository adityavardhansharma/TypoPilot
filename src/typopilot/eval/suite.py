"""Run any correction system over the fixed self-test suite (idea.md s14, s16).

A "system" is any callable ``correct(text: str) -> str``. The runner applies
it to every test file, computes per-category and overall metrics, measures
per-example latency, and writes all predictions for later regression
comparison between runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .metrics import aggregate, cer, correction_rates, exact_match, wer


def load_suite(suite_dir: str | Path) -> dict[str, list[dict]]:
    """Load every category file in the test-suite directory."""
    suite = {}
    for path in sorted(Path(suite_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            suite[path.stem] = [json.loads(line) for line in fh if line.strip()]
    if not suite:
        raise FileNotFoundError(f"no *.jsonl test files in {suite_dir}")
    return suite


def run_suite(
    correct: Callable[[str], str],
    suite_dir: str | Path,
    predictions_path: str | Path | None = None,
    system_name: str = "system",
) -> dict:
    """Evaluate *correct* over the suite; return per-category + overall metrics."""
    suite = load_suite(suite_dir)
    per_category: dict[str, dict] = {}
    all_rows: list[dict] = []
    predictions: list[dict] = []

    for category, examples in suite.items():
        rows = []
        for example in examples:
            started = time.perf_counter()
            hypothesis = correct(example["input"])
            latency_ms = (time.perf_counter() - started) * 1000

            row = {
                "cer": cer(hypothesis, example["reference"]),
                "wer": wer(hypothesis, example["reference"]),
                "exact_match": exact_match(hypothesis, example["reference"]),
                "latency_ms": latency_ms,
                **correction_rates(example["input"], hypothesis, example["reference"]),
            }
            rows.append(row)
            predictions.append({
                "id": example["id"],
                "category": category,
                "input": example["input"],
                "reference": example["reference"],
                "hypothesis": hypothesis,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
            })
        per_category[category] = aggregate(rows)
        all_rows.extend(rows)

    report = {
        "system": system_name,
        "overall": aggregate(all_rows),
        "categories": per_category,
    }

    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as fh:
            for prediction in predictions:
                fh.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return report
