#!/usr/bin/env bash
# Lightning AI Studio setup for the TypoPilot final run.
# Run from the repo root after cloning. Safe to re-run.
set -euo pipefail

echo "== 1/4 dependencies =="
pip install -q torch transformers==5.5.4 "accelerate>=1.1.0" pyarrow numpy tensorboard pytest

echo "== 2/4 GPU check =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible - is the Studio on a GPU machine?"
name = torch.cuda.get_device_name(0)
bf16 = torch.cuda.is_bf16_supported()
print(f"GPU: {name}  bf16: {bf16}")
if not bf16:
    print("WARNING: no bf16 -> use configs/training/byt5_small_final.json (FP32) instead of the cloud config")
PY

echo "== 3/4 dataset =="
if [ ! -f data/final/train.jsonl ]; then
    if [ -f data_final.zip ]; then
        mkdir -p data && unzip -q data_final.zip -d data/
        echo "unzipped data_final.zip -> data/final/"
    else
        echo "ERROR: upload data_final.zip to the repo root first" >&2
        exit 1
    fi
fi
python -c "print(sum(1 for _ in open('data/final/train.jsonl')), 'train pairs')"

echo "== 4/4 unit tests =="
python -m pytest tests -q

cat <<'EOF'

Setup complete. Next steps:

  # 30-step smoke test (~2 min on L40S) - run this on the T4/CPU first if saving L40S hours:
  PYTHONPATH=src python -m typopilot.models.train --config configs/training/byt5_small_final_cloud.json --smoke

  # The real run (~30-60 min on L40S):
  PYTHONPATH=src python -m typopilot.models.train --config configs/training/byt5_small_final_cloud.json

  # If interrupted, resume with:
  PYTHONPATH=src python -m typopilot.models.train --config configs/training/byt5_small_final_cloud.json --resume

Afterwards download experiments/byt5-small-final-002/checkpoint-best/ (~1.2GB)
back to the laptop for local evaluation.
EOF
