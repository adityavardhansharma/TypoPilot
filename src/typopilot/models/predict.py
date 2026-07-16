"""Correct text with a trained checkpoint — the local test interface (s2.9).

Interactive:
    python -m typopilot.models.predict --checkpoint experiments/byt5-small-pilot-001/checkpoint-best

One-shot:
    python -m typopilot.models.predict --checkpoint ... --text "waht Is Th e weahter"

Evaluate on the fixed self-test suite (writes predictions + report):
    python -m typopilot.models.predict --checkpoint ... --suite data/test_suite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Corrector:
    def __init__(self, checkpoint: str, device: str | None = None,
                 max_input_bytes: int = 320, max_new_tokens: int = 320):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.max_input_bytes = max_input_bytes
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def __call__(self, text: str) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt",
            max_length=self.max_input_bytes, truncation=True,
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", help="correct one string and exit")
    parser.add_argument("--suite", type=Path,
                        help="run the fixed self-test suite at this directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory for suite predictions/report "
                             "(default: alongside the checkpoint)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args(argv)

    load_started = time.perf_counter()
    corrector = Corrector(args.checkpoint, device=args.device)
    load_seconds = time.perf_counter() - load_started
    print(f"[model loaded in {load_seconds:.1f}s on {corrector.device}]", file=sys.stderr)

    if args.suite:
        from typopilot.eval.suite import run_suite
        out_dir = args.out or Path(args.checkpoint).parent / "suite_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = run_suite(
            corrector, args.suite,
            predictions_path=out_dir / "predictions.jsonl",
            system_name=str(args.checkpoint),
        )
        report["model_load_seconds"] = round(load_seconds, 1)
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["overall"], indent=2))
        print(f"per-category report: {out_dir / 'report.json'}")
        return 0

    if args.text is not None:
        print(corrector(args.text))
        return 0

    print("interactive mode — empty line to quit", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        started = time.perf_counter()
        corrected = corrector(line)
        latency_ms = (time.perf_counter() - started) * 1000
        print(f"{corrected}   [{latency_ms:.0f} ms]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
