"""Command-line entry point for the noise generator.

Corrupt a JSONL file of clean sentences into (noisy -> clean) training pairs:

    python -m typopilot.noise.cli corrupt data/clean.jsonl data/pairs.jsonl \
        --seed 13 --variants 2 --mix light=1,medium=2,heavy=2,extreme=1

Or corrupt ad-hoc text from stdin to eyeball the output:

    echo {"text": "What is the weather today?"} | python -m typopilot.noise.cli demo
"""

from __future__ import annotations

import argparse
import json
import sys

from .generator import PROFILES, NoiseGenerator


def _parse_mix(raw: str) -> dict[str, float]:
    mix = {}
    for part in raw.split(","):
        name, _, weight = part.partition("=")
        mix[name.strip()] = float(weight)
    unknown = set(mix) - set(PROFILES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown severities: {sorted(unknown)}")
    return mix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="typopilot-noise", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    corrupt = sub.add_parser("corrupt", help="corrupt a JSONL file of clean sentences")
    corrupt.add_argument("input", help="JSONL file with a 'text' field per line")
    corrupt.add_argument("output", help="destination JSONL of training pairs")
    corrupt.add_argument("--seed", type=int, default=13)
    corrupt.add_argument("--variants", type=int, default=1,
                         help="corrupted variants per clean sentence")
    corrupt.add_argument("--mix", type=_parse_mix, default=None,
                         help="severity mix, e.g. light=1,medium=2,heavy=2,extreme=1")
    corrupt.add_argument("--text-key", default="text")

    demo = sub.add_parser("demo", help="corrupt one sentence at every severity")
    demo.add_argument("text", nargs="?", help="sentence to corrupt (default: read stdin)")
    demo.add_argument("--seed", type=int, default=13)

    args = parser.parse_args(argv)
    generator = NoiseGenerator(seed=args.seed)

    if args.command == "corrupt":
        stats = generator.corrupt_file(
            args.input,
            args.output,
            severity_mix=args.mix,
            variants_per_sentence=args.variants,
            text_key=args.text_key,
        )
        json.dump(stats, sys.stdout, indent=2)
        print()
    elif args.command == "demo":
        text = args.text if args.text is not None else sys.stdin.read().strip()
        for severity in PROFILES:
            result = generator.corrupt(text, severity, example_id="demo")
            ops = ", ".join(entry["op"] for entry in result.ops_log) or "(none)"
            print(f"[{severity:7}] {result.noisy}")
            print(f"          ops: {ops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
