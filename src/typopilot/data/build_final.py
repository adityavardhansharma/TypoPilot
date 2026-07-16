"""Build the commercial-clean final training dataset (run 002).

Data policy: every source permits commercial use —
  * Tatoeba (CC-BY)                — clean everyday sentences
  * C4_200M via HF parquet (ODC-BY) — grammar-flavoured error pairs
  * client9/misspell (MIT) + Wikipedia lists (CC BY-SA) — misspelling dict
  * our synthetic noise generator  — everything else
BEA-2019 / DailyDialog / JFLEG (non-commercial) are NOT used.

Composition (~116k pairs):
  55k  synthetic single-sentence   (Tatoeba + noise generator w/ dictionary)
  25k  synthetic multi-sentence    (2-4 Tatoeba sentences concatenated, then
                                    corrupted — trains long-input handling)
  30k  grammar pairs               (C4_200M)
  6k   identity                    (clean in = clean out)

Fixed test suite (new, commercial-clean — replaces the BEA/DD suite for all
future runs): light/medium/heavy/extreme singles, multi-sentence paragraphs,
no-change, and C4 grammar categories, all from held-out sentences.

Usage:
    python -m typopilot.data.build_final --raw data/raw --out data/final --seed 29
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from typopilot.noise.generator import NoiseGenerator

from .c4_200m import iter_pairs as iter_c4
from .clean import clean_sentence, dedup_key
from .misspellings import load_combined
from .tatoeba import iter_sentences

# Default pool / composition sizes (run 002). Overridable via CLI for larger
# runs; see DEFAULT_SIZES and the --* size flags in main().
TATOEBA_POOL = 300_000          # cleaned sentences to collect before sampling
SYNTH_SINGLE = 55_000
SYNTH_MULTI = 25_000
C4_PAIRS = 30_000
IDENTITY = 6_000
VAL_FRACTION = 0.02

DEFAULT_SIZES = {
    "tatoeba_pool": TATOEBA_POOL,
    "synth_single": SYNTH_SINGLE,
    "synth_multi": SYNTH_MULTI,
    "c4_pairs": C4_PAIRS,
    "identity": IDENTITY,
}

MULTI_MAX_BYTES = 280           # concatenated clean text cap (training cap 320)
# Hard cap on both sides of every pair. Anything longer would be truncated by
# the trainer, and truncated TARGETS teach the model to drop sentence endings.
MAX_PAIR_BYTES = 320

SEVERITY_MIX = {"light": 1.0, "medium": 2.0, "heavy": 2.0, "extreme": 1.0}

TEST_SUITE = {
    "english_light": 150,
    "english_medium": 150,
    "english_heavy": 150,
    "english_extreme": 100,
    "paragraphs_multi": 100,
    "no_change_needed": 100,
    "real_errors_c4": 150,
}


_SENTENCE_SPLIT = None  # lazily compiled below


def _load_suite_keys(suite_dir: Path) -> set[str]:
    """Dedup keys of every sentence already used in an existing test suite.

    Seeding the training-collection ``seen`` set with these guarantees no
    test sentence leaks into training, so a new (larger) run stays directly
    comparable on the SAME fixed suite. For multi-sentence paragraph
    references we also add the keys of the split sentences, since those are
    what would otherwise be collected individually from Tatoeba.
    """
    import re
    keys: set[str] = set()
    for path in Path(suite_dir).glob("*.jsonl"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in ("reference", "input"):
                    text = row.get(field, "")
                    keys.add(dedup_key(text))
                    for part in re.split(r"(?<=[.?!])\s+", text):
                        if part.strip():
                            keys.add(dedup_key(part))
    return keys


def _collect_tatoeba(path: Path, seen: set[str], rejections: Counter,
                     limit: int) -> list[str]:
    kept = []
    for sentence in iter_sentences(path):
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
        if len(kept) >= limit:
            break
    return kept


def _collect_c4(path: Path, seen: set[str], rejections: Counter, limit: int) -> list[dict]:
    pairs = []
    for errorful, clean in iter_c4(path):
        target, reason = clean_sentence(clean)
        rejections[f"c4_{reason}"] += 1
        if target is None:
            continue
        source, source_reason = clean_sentence(errorful)
        if source is None:
            rejections[f"c4_source_{source_reason}"] += 1
            continue
        if source == target:
            rejections["c4_source_equals_target"] += 1
            continue
        key = dedup_key(target)
        if key in seen:
            rejections["duplicate"] += 1
            continue
        seen.add(key)
        pairs.append({"input": source, "target": target,
                      "kind": "grammar", "source": "c4_200m"})
        if len(pairs) >= limit:
            break
    return pairs


def _concat_sentences(pool: list[str], rng: random.Random, count: int) -> list[str]:
    """Build multi-sentence 'paragraphs' from consecutive pool sentences."""
    paragraphs = []
    index = 0
    while len(paragraphs) < count and index + 4 < len(pool):
        n = rng.randint(2, 4)
        chunk = pool[index:index + n]
        index += n
        text = " ".join(chunk)
        while len(text.encode()) > MULTI_MAX_BYTES and len(chunk) > 2:
            chunk = chunk[:-1]
            text = " ".join(chunk)
        if len(text.encode()) <= MULTI_MAX_BYTES and len(chunk) >= 2:
            paragraphs.append(text)
    return paragraphs


def _write_suite(suite_dir: Path, holdout: list[str], c4_holdout: list[dict],
                 generator, rng, severities, weights) -> dict:
    """Generate and write the fixed commercial test suite from held-out data."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_singles_needed = sum(
        n for name, n in TEST_SUITE.items()
        if name.startswith("english_") or name == "no_change_needed"
    )
    cursor = 0
    para_cursor = suite_singles_needed
    suite_counts = {}
    for name, count in TEST_SUITE.items():
        rows = []
        if name.startswith("english_"):
            severity = name.removeprefix("english_")
            chunk = holdout[cursor:cursor + count]
            cursor += count
            for i, sentence in enumerate(chunk):
                result = generator.corrupt(sentence, severity, example_id=f"suiteC-{name}-{i}")
                rows.append({
                    "id": f"{name}-{i:03}", "input": result.noisy, "reference": sentence,
                    "language": "english", "severity": severity,
                    "error_types": sorted({e["op"] for e in result.ops_log}),
                })
        elif name == "no_change_needed":
            chunk = holdout[cursor:cursor + count]
            cursor += count
            rows = [{
                "id": f"{name}-{i:03}", "input": s, "reference": s,
                "language": "english", "severity": "none", "error_types": [],
            } for i, s in enumerate(chunk)]
        elif name == "paragraphs_multi":
            paragraphs = _concat_sentences(holdout[para_cursor:], rng, count)
            for i, paragraph in enumerate(paragraphs):
                severity = rng.choices(severities, weights=weights, k=1)[0]
                result = generator.corrupt(paragraph, severity, example_id=f"suiteC-para-{i}")
                rows.append({
                    "id": f"{name}-{i:03}", "input": result.noisy, "reference": paragraph,
                    "language": "english", "severity": severity,
                    "error_types": sorted({e["op"] for e in result.ops_log}),
                })
        elif name == "real_errors_c4":
            rows = [{
                "id": f"{name}-{i:03}", "input": p["input"], "reference": p["target"],
                "language": "english", "severity": "grammar", "error_types": [],
            } for i, p in enumerate(c4_holdout)]
        with (suite_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        suite_counts[name] = len(rows)
    return suite_counts


def build(raw_dir: Path, out_dir: Path, seed: int,
          sizes: dict | None = None, reuse_suite_dir: Path | None = None) -> dict:
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    synth_single = sizes["synth_single"]
    synth_multi = sizes["synth_multi"]
    c4_pairs_n = sizes["c4_pairs"]
    identity_n = sizes["identity"]
    tatoeba_pool = sizes["tatoeba_pool"]

    rng = random.Random(seed)
    rejections: Counter = Counter()
    seen: set[str] = set()

    # Reuse an existing fixed test suite: exclude its sentences from training
    # (no leakage) and skip regenerating the suite, so the new run is directly
    # comparable to the previous one on the SAME benchmark.
    reuse = reuse_suite_dir is not None
    if reuse:
        seen |= _load_suite_keys(reuse_suite_dir)
        rejections["excluded_suite_keys"] = len(seen)

    misspellings = load_combined(
        raw_dir / "misspellings" / "misspell_mit.go",
        raw_dir / "misspellings" / "wikipedia_misspellings.html",
    )
    generator = NoiseGenerator(seed=seed, misspellings=misspellings)

    tatoeba = _collect_tatoeba(
        raw_dir / "tatoeba" / "eng_sentences.tsv.bz2", seen, rejections,
        limit=tatoeba_pool,
    )
    rng.shuffle(tatoeba)
    c4_pairs = _collect_c4(
        raw_dir / "c4_200m" / "train-0.parquet", seen, rejections,
        limit=c4_pairs_n + TEST_SUITE["real_errors_c4"],
    )
    rng.shuffle(c4_pairs)

    if reuse:
        # No holdout carving needed; the suite already exists on disk.
        holdout: list[str] = []
        trainable = tatoeba
        c4_holdout: list[dict] = []
        c4_train = c4_pairs[:c4_pairs_n]
    else:
        # ---- Carve held-out test material FIRST (leakage rule) ----
        suite_singles_needed = sum(
            n for name, n in TEST_SUITE.items()
            if name.startswith("english_") or name == "no_change_needed"
        )
        suite_para_sentences = TEST_SUITE["paragraphs_multi"] * 4
        holdout = tatoeba[:suite_singles_needed + suite_para_sentences]
        trainable = tatoeba[suite_singles_needed + suite_para_sentences:]
        c4_holdout = c4_pairs[:TEST_SUITE["real_errors_c4"]]
        c4_train = c4_pairs[TEST_SUITE["real_errors_c4"]:c4_pairs_n + TEST_SUITE["real_errors_c4"]]

    # ---- Training pairs ----
    needed = synth_single + identity_n
    if len(trainable) < needed:
        raise RuntimeError(f"tatoeba pool too small: {len(trainable)} < {needed}")

    severities = list(SEVERITY_MIX)
    weights = [SEVERITY_MIX[s] for s in severities]
    records: list[dict] = []

    for i, sentence in enumerate(trainable[:synth_single]):
        severity = rng.choices(severities, weights=weights, k=1)[0]
        result = generator.corrupt(sentence, severity, example_id=f"final-s-{i}")
        record = result.to_record(f"syn-{i}")
        record.update({"kind": "synthetic", "source": "tatoeba"})
        records.append(record)

    multi_pool = trainable[synth_single + identity_n:]
    for i, paragraph in enumerate(_concat_sentences(multi_pool, rng, synth_multi)):
        severity = rng.choices(severities, weights=weights, k=1)[0]
        result = generator.corrupt(paragraph, severity, example_id=f"final-m-{i}")
        record = result.to_record(f"multi-{i}")
        record.update({"kind": "synthetic_multi", "source": "tatoeba"})
        records.append(record)

    for i, sentence in enumerate(trainable[synth_single:synth_single + identity_n]):
        records.append({
            "id": f"ident-{i}", "input": sentence, "target": sentence,
            "kind": "identity", "source": "tatoeba", "severity": "none",
        })

    for i, pair in enumerate(c4_train):
        records.append(dict(pair, id=f"c4-{i}", severity="grammar"))

    def _fits(record: dict) -> bool:
        return (len(record["input"].encode()) <= MAX_PAIR_BYTES
                and len(record["target"].encode()) <= MAX_PAIR_BYTES)

    before = len(records)
    records = [r for r in records if _fits(r)]
    rejections["pair_too_many_bytes"] = before - len(records)

    rng.shuffle(records)
    val_size = round(len(records) * VAL_FRACTION)
    val_records, train_records = records[:val_size], records[val_size:]

    # ---- Test suite ----
    if reuse:
        suite_counts = {"reused_from": str(reuse_suite_dir)}
    else:
        suite_counts = _write_suite(
            out_dir.parent / "test_suite_commercial",
            holdout, c4_holdout, generator, rng, severities, weights,
        )

    # ---- Write splits + summary ----
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in (("train", train_records), ("val", val_records)):
        with (out_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as fh:
            for record in split_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "seed": seed,
        "licence_policy": "commercial-clean only (Tatoeba CC-BY, C4_200M ODC-BY, "
                          "misspell MIT, Wikipedia CC BY-SA, own generator)",
        "train_size": len(train_records),
        "val_size": len(val_records),
        "composition": dict(Counter(r["kind"] for r in records)),
        "severity_distribution": dict(Counter(r.get("severity", "?") for r in records)),
        "avg_input_chars": round(sum(len(r["input"]) for r in records) / len(records), 1),
        "max_input_bytes": max(len(r["input"].encode()) for r in records),
        "tatoeba_pool": len(tatoeba),
        "c4_pairs_used": len(c4_train),
        "misspelling_dict_words": len(misspellings),
        "test_suite": suite_counts,
        "rejections": dict(rejections.most_common(20)),
    }
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/final"))
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--tatoeba-pool", type=int, default=DEFAULT_SIZES["tatoeba_pool"])
    parser.add_argument("--synth-single", type=int, default=DEFAULT_SIZES["synth_single"])
    parser.add_argument("--synth-multi", type=int, default=DEFAULT_SIZES["synth_multi"])
    parser.add_argument("--c4-pairs", type=int, default=DEFAULT_SIZES["c4_pairs"])
    parser.add_argument("--identity", type=int, default=DEFAULT_SIZES["identity"])
    parser.add_argument("--reuse-suite", type=Path, default=None,
                        help="reuse an existing test-suite dir; exclude its "
                             "sentences from training and skip regenerating it")
    args = parser.parse_args(argv)
    sizes = {
        "tatoeba_pool": args.tatoeba_pool,
        "synth_single": args.synth_single,
        "synth_multi": args.synth_multi,
        "c4_pairs": args.c4_pairs,
        "identity": args.identity,
    }
    summary = build(args.raw, args.out, args.seed,
                    sizes=sizes, reuse_suite_dir=args.reuse_suite)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
