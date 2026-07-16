# Lightning AI runbook — final run (byt5-small-final-002)

Strategy: **set up and smoke-test on the cheap machine, switch to L40S only
for the real run.** Setup overhead (installs, downloads) costs the same
wall-clock on any GPU — don't burn L40S hours on it.

## 1. Create the Studio (CPU or T4 machine)

- lightning.ai → new Studio → machine: **CPU or T4**
- In the Studio terminal:

```bash
git clone https://github.com/adityavardhansharma/TypoPilot.git
cd TypoPilot
```

(Private repo: use a GitHub personal access token when prompted, or
`gh auth login`.)

## 2. Upload the dataset

Drag `data_final.zip` (created locally by `scripts/make_data_zip.ps1`,
~25MB) into the `TypoPilot/` folder via the Studio file panel. Then:

```bash
bash scripts/lightning_setup.sh
```

The script installs pinned deps, checks the GPU, unzips the data, and runs
the unit tests.

## 3. Smoke test (still on the cheap machine)

```bash
# NOTE: uses the LOCAL config — the cloud config's bf16 + batch 32 needs the L40S
PYTHONPATH=src python -m typopilot.models.train \
    --config configs/training/byt5_small_final.json --smoke
```

30 steps; verifies data loading, forward/backward, checkpoint save. After
switching to the L40S you can optionally smoke the cloud config there too
(~2 min) before the real run.

## 4. Switch machine to L40S and run

Studio settings → machine → **L40S**. Wait for restart, then:

```bash
cd TypoPilot
PYTHONPATH=src python -m typopilot.models.train \
    --config configs/training/byt5_small_final_cloud.json
```

Expected: ~30–60 min, bf16, batch 32. Checkpoints land every 500 steps in
`experiments/byt5-small-final-002/`. If the session dies, re-run with
`--resume`.

## 5. Bring the model home

Download `experiments/byt5-small-final-002/checkpoint-best/` (~1.2GB) via
the file panel (right-click → download, or zip it first:
`zip -r ckpt.zip experiments/byt5-small-final-002/checkpoint-best`).

**Switch the Studio back to CPU (or stop it) immediately** — don't idle on
L40S hours.

## 6. Local evaluation (on the laptop)

```powershell
$env:PYTHONPATH = "E:\TypoPilot\src"
E:\TypoPilot\.venv\Scripts\python.exe -m typopilot.models.predict `
    --checkpoint experiments\byt5-small-final-002\checkpoint-best `
    --suite data\test_suite_commercial `
    --out experiments\byt5-small-final-002\suite_eval
```

Latency numbers only count from the laptop — it is the deployment target.
