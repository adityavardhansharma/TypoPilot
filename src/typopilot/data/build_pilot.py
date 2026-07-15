"""Build the English pilot dataset (idea.md sections 9, 11 Stage A/B, 14).

Composition (targets, before rounding):
  ~75% synthetic  — clean sentences corrupted by the noise generator
  ~20% real       — genuine BEA-2019 (source -> corrected) learner edits
  ~5%  identity   — clean sentence as both input and target, teaching the
                    model to leave already-correct text alone

Leakage rules:
  * Deduplication runs on the clean side (casefolded, punctuation-stripped)
    across ALL sources before any splitting or corruption.
  * Train/val pools come from BEA *train* + DailyDialog *train* only.
  * The fixed self-test suite comes from BEA *dev* + DailyDialog *test*,
    and is additionally deduplicated against every train/val key.
  * JFLEG is never touched here; it is a pure held-out benchmark.

Usage:
    python -m typopilot.data.build_pilot --raw data/raw --out data/pilot --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from typopilot.noise.generator import NoiseGenerator

from .clean import clean_sentence, dedup_key
from .dailydialog import iter_utterances
from .m2 import parse_m2

SEVERITY_MIX = {"light": 1.0, "medium": 2.0, "heavy": 2.0, "extreme": 1.0}

# Pilot sizes (idea.md section 11 Stage A: 5k-20k examples).
TRAIN_SYNTHETIC = 13500
TRAIN_REAL = 3600
TRAIN_IDENTITY = 900
VAL_FRACTION = 0.05

# Fixed self-test suite sizes per category (idea.md section 14).
TEST_SUITE = {
    "english_light": 150,
    "english_medium": 150,
    "english_heavy": 150,
    "english_extreme": 100,
    "no_change_needed": 100,
    "real_errors_bea_dev": 150,
}


def _collect_clean(sentences, seen: set[str], rejections: Counter) -> list[str]:
    """Clean + filter + dedup a stream of sentences against *seen* keys."""
    kept = []
    for sentence in sentences:
        cleaned, reason = clean_sentence(sentence)
        rejections[reason] += 1
        if cleaned is None:
            continue
        key = dedup_key(cleaned)
        if key in seen:
            rejections["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(cleaned)
    return kept


def _collect_real_pairs(m2_path: Path, seen: set[str], rejections: Counter) -> list[dict]:
    """Real (errorful -> corrected) BEA pairs whose target passes cleaning."""
    pairs = []
    for example in parse_m2(m2_path):
        if example.edit_count == 0:
            continue
        target, reason = clean_sentence(example.target)
        rejections[reason] += 1
        if target is None:
            continue
        source, source_reason = clean_sentence(example.source)
        if source is None:
            rejections[f"source_{source_reason}"] += 1
            continue
        if source == target:
            rejections["source_equals_target"] += 1
            continue
        key = dedup_key(target)
        if key in seen:
            rejections["duplicate"] += 1
            continue
        seen.add(key)
        pairs.append({
            "input": source,
            "target": target,
            "kind": "real",
            "source": "bea",
            "edit_types": list(example.edit_types),
        })
    return pairs


def build(raw_dir: Path, out_dir: Path, seed: int) -> dict:
    rng = random.Random(seed)
    generator = NoiseGenerator(seed=seed)
    rejections: Counter = Counter()
    seen: set[str] = set()

    # ---- Train/val pools (BEA train + DailyDialog train only) ----
    bea_train_m2 = raw_dir / "wi+locness" / "m2" / "ABC.train.gold.bea19.m2"
    real_pairs = _collect_real_pairs(bea_train_m2, seen, rejections)

    bea_clean = _collect_clean(
        (ex.target for ex in parse_m2(bea_train_m2) if ex.edit_count == 0),
        seen, rejections,
    )
    dd_clean = _collect_clean(
        iter_utterances(raw_dir / "dailydialog" / "train.parquet"),
        seen, rejections,
    )

    # ---- Test-suite pool (BEA dev + DailyDialog test; disjoint by `seen`) ----
    bea_dev_m2 = raw_dir / "wi+locness" / "m2" / "ABCN.dev.gold.bea19.m2"
    dev_real_pairs = _collect_real_pairs(bea_dev_m2, seen, rejections)
    test_clean = _collect_clean(
        iter_utterances(raw_dir / "dailydialog" / "test.parquet"),
        seen, rejections,
    )

    # ---- Assemble training pairs ----
    clean_pool = bea_clean + dd_clean
    rng.shuffle(clean_pool)
    needed = TRAIN_SYNTHETIC + TRAIN_IDENTITY
    if len(clean_pool) < needed:
        raise RuntimeError(f"clean pool too small: {len(clean_pool)} < {needed}")

    severities = list(SEVERITY_MIX)
    weights = [SEVERITY_MIX[s] for s in severities]

    bea_clean_set = set(bea_clean)
    records: list[dict] = []
    for i, sentence in enumerate(clean_pool[:TRAIN_SYNTHETIC]):
        severity = rng.choices(severities, weights=weights, k=1)[0]
        result = generator.corrupt(sentence, severity, example_id=f"pilot-{i}")
        record = result.to_record(f"syn-{i}")
        record.update({"kind": "synthetic",
                       "source": "bea" if sentence in bea_clean_set else "dailydialog"})
        records.append(record)

    for i, sentence in enumerate(clean_pool[TRAIN_SYNTHETIC:TRAIN_SYNTHETIC + TRAIN_IDENTITY]):
        records.append({
            "id": f"ident-{i}", "input": sentence, "target": sentence,
            "kind": "identity", "source": "mixed", "severity": "none",
        })

    rng.shuffle(real_pairs)
    for i, pair in enumerate(real_pairs[:TRAIN_REAL]):
        pair = dict(pair, id=f"real-{i}", severity="real")
        records.append(pair)

    rng.shuffle(records)
    val_size = round(len(records) * VAL_FRACTION)
    val_records, train_records = records[:val_size], records[val_size:]

    # ---- Assemble fixed self-test suite ----
    rng.shuffle(test_clean)
    suite_dir = out_dir.parent / "test_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    suite_counts = {}
    for name, count in TEST_SUITE.items():
        rows = []
        if name.startswith("english_"):
            severity = name.removeprefix("english_")
            chunk = test_clean[cursor:cursor + count]
            cursor += count
            for i, sentence in enumerate(chunk):
                result = generator.corrupt(sentence, severity, example_id=f"suite-{name}-{i}")
                rows.append({
                    "id": f"{name}-{i:03}", "input": result.noisy, "reference": sentence,
                    "language": "english", "severity": severity,
                    "error_types": sorted({e["op"] for e in result.ops_log}),
                })
        elif name == "no_change_needed":
            chunk = test_clean[cursor:cursor + count]
            cursor += count
            rows = [{
                "id": f"{name}-{i:03}", "input": s, "reference": s,
                "language": "english", "severity": "none", "error_types": [],
            } for i, s in enumerate(chunk)]
        elif name == "real_errors_bea_dev":
            rng.shuffle(dev_real_pairs)
            rows = [{
                "id": f"{name}-{i:03}", "input": p["input"], "reference": p["target"],
                "language": "english", "severity": "real",
                "error_types": p["edit_types"],
            } for i, p in enumerate(dev_real_pairs[:count])]
        with (suite_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        suite_counts[name] = len(rows)

    # ---- Write train/val + summary ----
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in (("train", train_records), ("val", val_records)):
        with (out_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as fh:
            for record in split_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _lengths(rows, key):
        lengths = [len(r[key]) for r in rows]
        return round(sum(lengths) / max(1, len(lengths)), 1)

    summary = {
        "seed": seed,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "composition": dict(Counter(r["kind"] for r in records)),
        "severity_distribution": dict(Counter(r.get("severity", "?") for r in records)),
        "source_distribution": dict(Counter(r.get("source", "?") for r in records)),
        "avg_input_chars": _lengths(records, "input"),
        "avg_target_chars": _lengths(records, "target"),
        "clean_pool": {"bea": len(bea_clean), "dailydialog": len(dd_clean)},
        "real_pairs_available": len(real_pairs),
        "test_suite": suite_counts,
        "rejections": dict(rejections.most_common()),
    }
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/pilot"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    summary = build(args.raw, args.out, args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
