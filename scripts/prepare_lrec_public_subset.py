"""Prepare a deterministic TypoPilot suite from the public NSEC benchmark.

The LREC 2020 SpellingBenchmark repository publishes the generator/evaluator,
but its paper test files are not bundled.  The author's companion
``nsec_benchmark`` repository contains a 9,060-sentence English Wikipedia
benchmark in ``demo/``.  That downloadable corpus exercises word splitting
and concatenation.  This script reconstructs sentence text from its token
JSON and emits the JSONL layout consumed by TypoPilot's native evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


ID_RE = re.compile(r"a(?P<article>\d+)\.s(?P<sentence>\d+)\.w(?P<word>\d+)")


def parse_id(identifier: str) -> tuple[int, int, int]:
    match = ID_RE.fullmatch(identifier)
    if match is None:
        raise ValueError(f"unexpected benchmark token id: {identifier}")
    return tuple(int(match.group(name)) for name in ("article", "sentence", "word"))


def affected_bounds(identifier: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    pieces = identifier.split("-")
    first = parse_id(pieces[0])
    last = parse_id(pieces[-1])
    if first[:2] != last[:2]:
        raise ValueError(f"correction crosses sentence boundary: {identifier}")
    return first, last


def render_source(tokens: list[dict]) -> str:
    return "".join(token["token"] + (" " if token["space"] else "") for token in tokens).strip()


def render_reference(corrections: list[dict], source_by_word: dict[int, dict]) -> str:
    pieces: list[str] = []
    for index, correction in enumerate(corrections):
        _, last = affected_bounds(correction["affected-id"])
        text = correction["correct"]
        if not text:
            continue

        add_space = bool(source_by_word[last[2]]["space"])
        if correction["type"] == "CONCATENATION" and index + 1 < len(corrections):
            next_correction = corrections[index + 1]
            next_first, _ = affected_bounds(next_correction["affected-id"])
            # A concatenation error is represented by consecutive ground-truth
            # tokens whose source tokens have no separating whitespace.
            if next_correction["type"] == "CONCATENATION" and next_first[2] == last[2] + 1:
                add_space = True

        pieces.append(text + (" " if add_space else ""))
    return "".join(pieces).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_path = args.benchmark_dir / "demo" / "source_backup.json"
    groundtruth_path = args.benchmark_dir / "demo" / "groundtruth.json"
    source_doc = json.loads(source_path.read_text(encoding="utf-8"))
    groundtruth_doc = json.loads(groundtruth_path.read_text(encoding="utf-8"))

    source_sentences: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for token in source_doc["tokens"]:
        article, sentence, _ = parse_id(token["id"])
        source_sentences[(article, sentence)].append(token)

    correction_sentences: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for correction in groundtruth_doc["corrections"]:
        first, _ = affected_bounds(correction["affected-id"])
        correction_sentences[first[:2]].append(correction)

    expected = sum(groundtruth_doc["information"]["sentences"])
    if len(source_sentences) != expected or len(correction_sentences) != expected:
        raise RuntimeError(
            f"expected {expected} sentences, found source={len(source_sentences)} "
            f"groundtruth={len(correction_sentences)}"
        )

    examples: list[dict] = []
    for article_sentence in sorted(source_sentences):
        tokens = source_sentences[article_sentence]
        corrections = correction_sentences[article_sentence]
        by_word = {parse_id(token["id"])[2]: token for token in tokens}
        error_types = sorted({item["type"] for item in corrections if item["type"] != "NONE"})
        if error_types == ["SPLIT"]:
            category = "split_only"
        elif error_types == ["CONCATENATION"]:
            category = "concatenation_only"
        elif error_types:
            category = "mixed_split_concatenation"
        else:
            category = "no_change"

        article, sentence = article_sentence
        examples.append({
            "id": f"nsec-a{article}-s{sentence}",
            "input": render_source(tokens),
            "reference": render_reference(corrections, by_word),
            "error_types": error_types,
            "category": category,
        })

    rng = random.Random(args.seed)
    selected = rng.sample(examples, args.limit)
    selected.sort(key=lambda item: item["id"])
    args.out.mkdir(parents=True, exist_ok=True)
    for old_path in args.out.glob("*.jsonl"):
        old_path.unlink()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for example in selected:
        grouped[example.pop("category")].append(example)
    for category, rows in sorted(grouped.items()):
        with (args.out / f"{category}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source_repository": "https://github.com/naetherm/nsec_benchmark",
        "source_files": [str(source_path), str(groundtruth_path)],
        "scope": "public 9,060-sentence NSEC English Wikipedia split/concatenation benchmark",
        "not_the_lrec_paper_test_set": True,
        "seed": args.seed,
        "selected": len(selected),
        "available": len(examples),
        "categories": {category: len(rows) for category, rows in sorted(grouped.items())},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
