"""Score every baseline on the fixed test suite.

    python -m typopilot.baselines.run --suite data/test_suite --out experiments/baselines

Baselines:
  identity        — return the input unchanged (the floor: any useful system
                    must beat doing nothing)
  rules           — regex spacing/capitalisation fixes only
  symspell        — SymSpell lookup_compound spelling correction
  symspell+rules  — SymSpell, then the rule pass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typopilot.eval.suite import run_suite

from .rules import rule_correct
from .symspell_baseline import SymSpellCorrector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("data/test_suite"))
    parser.add_argument("--out", type=Path, default=Path("experiments/baselines"))
    args = parser.parse_args(argv)

    symspell = SymSpellCorrector()
    systems = {
        "identity": lambda text: text,
        "rules": rule_correct,
        "symspell": symspell,
        "symspell+rules": lambda text: rule_correct(symspell(text)),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, system in systems.items():
        report = run_suite(
            system,
            args.suite,
            predictions_path=args.out / f"{name}.predictions.jsonl",
            system_name=name,
        )
        reports[name] = report
        overall = report["overall"]
        print(f"{name:15} CER {overall['cer']:.4f}  WER {overall['wer']:.4f}  "
              f"EM {overall['exact_match']:.3f}  improved {overall['improvement']:+.4f}  "
              f"overcorrect {overall['overcorrected']:.3f}")

    with (args.out / "baseline_report.json").open("w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print(f"\nFull report: {args.out / 'baseline_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
