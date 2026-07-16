"""Evaluate the native TypoPilot Llama harness on a JSONL test suite.

The model is loaded once. Inputs are sent through the harness in one TSV batch,
and predictions/report files use the same metrics as ``typopilot.eval.suite``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from typopilot.eval.metrics import aggregate, cer, correction_rates, exact_match, wer
from typopilot.eval.suite import load_suite


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    examples: list[tuple[str, dict]] = [
        (category, example)
        for category, category_examples in suite.items()
        for example in category_examples
    ]
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", prefix="native-eval-", dir=args.out,
        encoding="utf-8", newline="", delete=False,
    ) as handle:
        input_path = Path(handle.name)
        for index, (_, example) in enumerate(examples):
            text = example["input"]
            if "\t" in text or "\n" in text or "\r" in text:
                raise ValueError(f"example {example['id']} contains TSV-breaking whitespace")
            handle.write(f"{index}\t{text}\n")

    command = [
        str(args.harness.resolve()),
        "--model", str(args.model.resolve()),
        "--threads", str(args.threads),
        "--max-tokens", str(args.max_tokens),
        "--input-tsv", str(input_path.resolve()),
        "--warmup",
    ]
    print(f"Evaluating {len(examples)} examples with {args.model.name}", flush=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    outputs: dict[int, dict] = {}
    assert process.stdout is not None
    for line in process.stdout:
        if not line.strip():
            continue
        result = json.loads(line)
        outputs[int(result["id"])] = result
        count = len(outputs)
        if count % 25 == 0 or count == len(examples):
            elapsed = time.perf_counter() - started
            rate = count / elapsed
            remaining = (len(examples) - count) / rate if rate else 0
            print(
                f"progress {count}/{len(examples)} "
                f"({100 * count / len(examples):.1f}%) "
                f"elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                flush=True,
            )

    assert process.stderr is not None
    stderr = process.stderr.read().strip()
    return_code = process.wait()
    input_path.unlink(missing_ok=True)
    if return_code:
        raise RuntimeError(f"harness failed ({return_code}): {stderr}")
    if len(outputs) != len(examples):
        raise RuntimeError(f"expected {len(examples)} predictions, received {len(outputs)}")

    all_metric_rows: list[dict] = []
    predictions: list[dict] = []
    category_rows: dict[str, list[dict]] = {category: [] for category in suite}
    no_change_total = 0
    no_change_changed = 0

    for index, (category, example) in enumerate(examples):
        native = outputs[index]
        hypothesis = native["output"]
        rates = correction_rates(example["input"], hypothesis, example["reference"])
        row = {
            "cer": cer(hypothesis, example["reference"]),
            "wer": wer(hypothesis, example["reference"]),
            "exact_match": exact_match(hypothesis, example["reference"]),
            "latency_ms": float(native["latency_ms"]),
            **rates,
        }
        all_metric_rows.append(row)
        category_rows[category].append(row)
        if example["input"] == example["reference"]:
            no_change_total += 1
            no_change_changed += hypothesis != example["input"]
        predictions.append({
            "id": example["id"],
            "category": category,
            "input": example["input"],
            "reference": example["reference"],
            "hypothesis": hypothesis,
            "prompt_tokens": native["prompt_tokens"],
            "output_tokens": native["output_tokens"],
            **{key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()},
        })

    latencies = [row["latency_ms"] for row in all_metric_rows]
    overall = aggregate(all_metric_rows)
    overall.update({
        "exact_match_count": sum(row["exact_match"] for row in all_metric_rows),
        "improved_count": sum(row["improvement"] > 0 for row in all_metric_rows),
        "overcorrected_count": sum(row["overcorrected"] for row in all_metric_rows),
        "undercorrected_count": sum(row["undercorrected"] for row in all_metric_rows),
        "unchanged_distance_count": sum(row["improvement"] == 0 for row in all_metric_rows),
        "no_change_examples": no_change_total,
        "no_change_changed_count": no_change_changed,
        "no_change_preservation_rate": round(1 - no_change_changed / no_change_total, 4) if no_change_total else None,
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(percentile(latencies, 0.95), 2),
        "latency_max_ms": round(max(latencies), 2),
    })

    categories = {}
    for category, rows in category_rows.items():
        metrics = aggregate(rows)
        metrics.update({
            "exact_match_count": sum(row["exact_match"] for row in rows),
            "overcorrected_count": sum(row["overcorrected"] for row in rows),
            "improved_count": sum(row["improvement"] > 0 for row in rows),
        })
        categories[category] = metrics

    report = {
        "system": "typopilot-llama32-1b-lora-001-q4-k-m-native-cpu",
        "model": str(args.model),
        "harness": str(args.harness),
        "threads": args.threads,
        "max_tokens": args.max_tokens,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "harness_stderr": stderr,
        "overall": overall,
        "categories": categories,
    }
    predictions_path = args.out / "predictions.jsonl"
    report_path = args.out / "report.json"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {predictions_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
