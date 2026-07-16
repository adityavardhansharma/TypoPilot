"""Correct text with a fine-tuned causal LLM (merged HF format).

    python -m typopilot.models.predict_llm --checkpoint experiments/llama32-1b-lora-001/merged --text "waht is teh weahter"
    python -m typopilot.models.predict_llm --checkpoint ... --suite data/test_suite_commercial

Uses the SAME prompt template as training. For fast local CPU inference, convert
the merged model to GGUF (scripts/export_gguf.sh) and run it in LM Studio / llama.cpp
instead; this path is for GPU/HF-format evaluation and comparison against ByT5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train_llm import PROMPT_TEMPLATE


class LLMCorrector:
    def __init__(self, checkpoint: str, device: str | None = None, max_new_tokens: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def __call__(self, text: str) -> str:
        prompt = PROMPT_TEMPLATE.format(noisy=text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Decode only the newly generated tokens (strip the prompt).
        gen = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    corrector = LLMCorrector(args.checkpoint, device=args.device)
    print(f"[loaded in {time.perf_counter()-started:.1f}s on {corrector.device}]", file=sys.stderr)

    if args.suite:
        from typopilot.eval.suite import run_suite
        out_dir = args.out or Path(args.checkpoint).parent / "suite_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = run_suite(corrector, args.suite,
                           predictions_path=out_dir / "predictions.jsonl",
                           system_name=str(args.checkpoint))
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["overall"], indent=2))
        return 0

    if args.text is not None:
        print(corrector(args.text))
        return 0

    print("interactive mode — empty line to quit", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        s = time.perf_counter()
        out = corrector(line)
        print(f"{out}   [{(time.perf_counter()-s)*1000:.0f} ms]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
