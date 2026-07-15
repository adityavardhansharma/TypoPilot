# TypoPilot

Research project for correcting heavily mistyped English and Hinglish text.
Full specification: [idea.md](idea.md).

## Current status

- [x] **Step 1 — Synthetic noise generator** (`src/typopilot/noise/`)
- [ ] Step 2 — Pilot dataset from BEA-2019 (clean targets + real edit pairs)
- [ ] Step 3 — Baselines (SymSpell + rule-based)
- [ ] Step 4 — ByT5-small pilot fine-tune (FP32, 6GB VRAM config)
- [ ] Step 5 — Full training run
- [ ] Step 6 — Evaluation, error analysis, report

## Setup

Python 3.11+. No third-party dependencies yet (stdlib only); `pytest` for tests.

```powershell
$env:PYTHONPATH = "src"     # or: pip install -e .
python -m pytest tests -q
```

## Noise generator usage

Corrupt a JSONL file of clean sentences (`{"text": "..."}` per line) into
(noisy input -> clean target) training pairs:

```powershell
python -m typopilot.noise.cli corrupt data/clean.jsonl data/pairs.jsonl `
    --seed 13 --variants 2 --mix light=1,medium=2,heavy=2,extreme=1
```

Eyeball corruption at every severity:

```powershell
python -m typopilot.noise.cli demo "What is the weather today?"
```

Key properties:

- **Deterministic** — every pair is a pure function of `(seed, example_id, severity)`;
  re-running produces byte-identical output. Each record stores its `example_seed`
  and the exact list of operations applied.
- **Keyboard-aware** — nearby-key substitutions use a QWERTY adjacency map derived
  from physical key geometry (`g -> f,h,t,y,v,b`).
- **Recoverable** — per-severity caps on operations-per-word stop short inputs
  from being corrupted beyond recognition; meaning-affecting word operations only
  touch a safe-list of filler words.
- **Streaming** — files are processed line by line, never loaded into RAM.

18 operations across five categories (character, word-boundary, capitalisation,
punctuation, word-level) and four severity buckets (light / medium / heavy / extreme).
