# Model comparison — pilot-001 vs final-002

Two ByT5-small English correction models trained in this project, and how they
differ across **data, training, evaluation, and performance**.

| | **pilot-001** (old) | **final-002** (new) |
|---|---|---|
| Purpose | prove the pipeline works | shippable, commercial-clean model |
| Base model | google/byt5-small (300M) | google/byt5-small (300M) |
| Trained on | GTX 1660 Ti (local) | L40S (Lightning AI cloud) |
| Wall-clock | 7 h 34 m | 39 m |
| Licence of training data | non-commercial | **commercial-clean** |

---

## 1. Data

The biggest change. The pilot used research-only corpora; the final model uses
only commercially licensed sources, so its results reflect what a real product
could ship.

| | pilot-001 | final-002 |
|---|---|---|
| Clean-text source | BEA-2019 corrected side + DailyDialog | **Tatoeba** (CC-BY) |
| Real error pairs | BEA-2019 learner edits (non-comm.) | **C4_200M** grammar pairs (ODC-BY) |
| Misspelling model | none — mechanical noise only | **11,035-word dict** (client9/misspell MIT + Wikipedia CC BY-SA), 31,135 real misspellings |
| Train / val pairs | 17,100 / 900 | **113,670 / 2,320** |
| Composition | 75% synth single · 20% real · 5% identity | 48% synth single · **22% multi-sentence** · 26% C4 grammar · 5% identity |
| Multi-sentence inputs | none | **25,000 paragraphs** (2–4 sentences) |
| Max sequence | 192 bytes | **320 bytes** |
| Licence status | research only | **fully commercial** |

**Why it matters**

- **Commercial-clean**: BEA/DailyDialog/JFLEG are dropped from training (kept only
  as research artifacts). Nothing that ships is encumbered.
- **Real misspellings**: the pilot only learned mechanical keyboard noise, so it
  missed conventional human misspellings (`recieved`, `tommorow`). The dictionary
  op teaches those directly.
- **Multi-sentence + 320-byte cap**: the pilot was trained on single sentences
  ≤192 bytes, so long paragraphs were out-of-distribution. The final model trains
  on 2–4-sentence paragraphs.
- **6.6× more data** overall.

GitHub Typo Corpus was evaluated and **rejected**: its official download is dead
(2019 S3 bucket), and the only surviving mirror stripped the per-repo licence
field, making commercial-rights filtering impossible.

---

## 2. Training

| | pilot-001 | final-002 |
|---|---|---|
| Hardware | GTX 1660 Ti, 6 GB (Turing) | L40S, 46 GB (Ada) |
| Precision | **FP32** (Turing has no bf16; T5 NaNs in fp16) | **bf16** (Ada supports it) |
| Micro-batch × accum | 2 × 8 (eff. 16) | **32 × 1 (eff. 32)** |
| Gradient checkpointing | on (needed for 6 GB) | **off** (46 GB has headroom) |
| Learning rate | 3e-4 | 4e-4 (scaled for larger batch) |
| Epochs / steps | 3 / 3,207 | 1.5 / **5,330** |
| Peak VRAM | 5.62 / 6 GB | 24.5 / 46 GB |
| Duration | **454 min** | **39.5 min** |
| Throughput | ~1.9 examples/s | **72 examples/s** (~38×) |

**Why it matters**

- Moving to the L40S unlocked **bf16**, removing the FP32-only constraint that
  made local training slow and delicate. Training is both faster and more stable.
- The larger card allowed **batch 32 with no gradient checkpointing**, which alone
  is a large throughput gain over the local setup.
- Net effect: an 11.5× wall-clock speedup (7.5 h → 40 min) despite training on
  6.6× more data, well within the free L40S allocation.

---

## 3. Evaluation

| | pilot-001 | final-002 |
|---|---|---|
| Test suite | 800 ex, BEA/DailyDialog-derived | **900 ex, commercial-clean** (Tatoeba/C4) |
| New categories | — | **paragraphs_multi**, **real_errors_c4** |
| Final eval_loss | 0.091 | **0.0945** |

**Important caveat:** the two eval_loss numbers are **not directly comparable** —
they are measured on different test suites (the final suite is harder and includes
multi-sentence paragraphs and real C4 grammar errors, which the pilot suite lacked).
The final suite was rebuilt commercial-clean and is the benchmark for all future
runs. The pilot's suite is retired.

The clean apples-to-apples comparison is the **fixed 8-input manual test**, run on
both models below.

---

## 4. Performance — head-to-head on identical inputs

Both models corrected the same 8 inputs (1 paragraph + 7 sentences). Full outputs:
[manual-test-001.html](manual-test-001.html) (pilot) and
[manual-test-002-final.html](manual-test-002-final.html) (final).

| Input | pilot-001 | final-002 | Change |
|---|---|---|---|
| `recieved…mesage…tommorow` | missed `mesage` | **all fixed** ✅ | dict caught it |
| `…completly diffrent` | missed `completly` | **all fixed** ✅ | dict caught it |
| `…form the latset…` | missed `form` | fixed `form→from` ✅ | improved |
| paragraph `gud enogh` | **`guess enough`** ⚠️ meaning change | `gud enough` ✅ safe | **harmful error gone** |
| paragraph `modle resluts` | left `modle` uncorrected | fixed `model` ✅ | improved |

**The single most important result:** the pilot produced a **meaning-changing
("harmful") correction** on the paragraph (`gud enogh` → "guess enough"). The final
model preserves meaning. Harmful corrections are the failure class the project
treats as most severe (spec §13, §21).

**Remaining misses (final model):** some real misspellings still slip through
(`corectly`, `erors`, `changs`, `rongly`, `lastest`), and it under-capitalises
all-lowercase paragraphs. Real-misspelling coverage improved but is not complete.

### Latency (local GTX 1660 Ti, FP32 greedy, warm)

| | pilot-001 | final-002 |
|---|---|---|
| Short sentence (median) | ~861 ms | ~1,086 ms |
| Model load | 1.4 s | 1.3 s |

Latency is comparable (both models are the same size/architecture). The final
model is marginally slower per call due to slightly longer average outputs, and
both sit **above the 500 ms product target** — the FP16/ONNX/quantisation pass
(Stage F) is the next lever and is not yet applied. Cloud (L40S) latencies run
~1.7× faster than local but are irrelevant to the product, which runs locally.

---

## 5. Verdict

The final model is a clear, honest improvement on every axis that matters for a
product:

- **Legally shippable** — commercial-clean data end to end.
- **Fewer harmful errors** — the pilot's meaning-change failure is gone on the
  test example.
- **Better real-misspelling coverage** — two previously-missed inputs now perfect.
- **Trained on paragraphs** — long-input handling is now in-distribution.
- **Far cheaper to train** — 40 min on free cloud vs 7.5 h locally.

Still open before product-readiness (spec §24): latency optimisation to hit the
<500 ms target, a confidence/safety guard for the residual meaning-change risk,
and completing the full automated suite scoring against baselines (in progress).
