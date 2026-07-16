# Experiment report — byt5-small-pilot-001

**Date:** 2026-07-16 · **Model:** google/byt5-small (300M) · **Hardware:** GTX 1660 Ti 6GB, FP32

## Setup

- Data: 17,100 train / 900 val pairs (75% synthetic corruption of BEA-2019 + DailyDialog
  clean text, 20% real BEA learner edits, 5% identity pairs), 192-byte caps.
- Training: 3 epochs, effective batch 16 (2 × accum 8), LR 3e-4 linear, gradient
  checkpointing, FP32 (T5 NaNs in fp16; Turing has no bf16).
- Duration: **7h 34m** (thermal throttling on the G14 roughly doubled the naive estimate).
  Peak VRAM 5.62GB. Train loss 14.3 → ~1.26 avg; eval loss 0.091, still declining slightly.

## Results — fixed self-test suite (800 examples, greedy decoding)

| System | CER | WER | Exact match | Overcorrected |
|---|---|---|---|---|
| identity (input unchanged) | 0.1010 | 0.3146 | 12.6% | 0% |
| rules | 0.0978 | 0.3057 | 12.8% | 6.8% |
| SymSpell | 0.1020 | 0.2868 | 8.5% | 44.5% |
| **ByT5 pilot** | **0.0501** | **0.1093** | **43.8%** | **9.3%** |

Per category (ByT5):

| Category | CER | EM | CER improvement | Overcorrected |
|---|---|---|---|---|
| english_light | 0.0225 | 68.0% | +0.030 | 10.7% |
| english_medium | 0.0295 | 47.3% | +0.054 | 4.0% |
| english_heavy | 0.0690 | 32.7% | +0.098 | 4.0% |
| english_extreme | 0.0806 | 25.0% | +0.136 | 5.0% |
| no_change_needed | 0.0032 | 91.0% | −0.003 | 9.0% |
| real_errors_bea_dev | 0.0902 | 8.0% | +0.001 | 21.3% |

## Latency (GPU, FP32, greedy, unoptimised)

- Median 1,265 ms · p90 2,561 ms · p95 3,100 ms · max 4,930 ms
- Short inputs (<60 chars): median 861 ms · Model load: 1.4 s

## Reading of the results

**The approach works.** The model halves the character error rate of untouched input,
more than triples every baseline's exact-match rate, and its improvement grows with
noise severity (+0.136 CER on extreme noise) — precisely the regime the project targets.
SymSpell, the strongest classical baseline, *damages* text at these noise levels
(44.5% overcorrection); the fine-tuned model overcorrects 9.3%.

**Identity discipline is decent but not free:** 91% of already-correct sentences pass
through untouched; 9% get needless edits despite the 5% identity-pair training share.

**Known failure modes** (from prediction inspection):
1. **Semantic guessing on extreme noise** — "I am opne t9 reoocat iin g." →
   "I am one to recommend it." (ref: "I am open to relocating."). Meaning-changing;
   the class the spec treats as most severe.
2. **Dropped clauses** — badly mangled trailing clauses are sometimes deleted rather
   than repaired ("ow much vit" → dropped).
3. **Nonword completions** — "Does r suitmee?" → "suitment".
4. **Word-order corrections beyond minimal edit** — reorders adverbs when the typo
   spans a word boundary.
5. **Real BEA grammar edits barely improved** (EM 8%) — grammar correction is a harder,
   more subjective task than typo repair and 20% training share wasn't enough.

**Latency misses the product target** (<500 ms short-sentence median) by ~1.7×
short / ~2.5× overall — expected for FP32 greedy generation with no Stage-F
optimisation (FP16 inference, ONNX, quantisation) applied yet.

## Verdict & next experiments

Pilot objective achieved: the pipeline trains, learns, and clearly beats all baselines.
Not yet product-threshold (§24). Recommended next steps, in order of expected value:

1. **FP16/ONNX inference pass (Stage F)** — likely 2–4× latency win, no retraining.
2. **Confidence guard (§22)** — reject outputs that differ too much from input or
   change negations/entities; directly attacks failure modes 1–2.
3. **Scale training data** to 50–150k pairs (the full clean pool supports it) with a
   higher real-pair share; eval loss was still declining at 3 epochs.
4. **Only then** revisit Hinglish (Stage C).
