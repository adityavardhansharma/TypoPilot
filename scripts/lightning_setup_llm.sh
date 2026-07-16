#!/usr/bin/env bash
# Extra deps for the LLM LoRA fine-tune (run after the repo + data are in place).
set -euo pipefail
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python

echo "== installing peft (+ sentencepiece for the Llama tokenizer) =="
$PY -m pip install -q "peft>=0.13" sentencepiece

echo "== GPU / bf16 check =="
$PY - <<'PYEOF'
import torch
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
print("-> if bf16 is False (e.g. T4), set bf16=false and fp16=true in the config")
PYEOF

echo "== data check =="
cd /teamspace/studios/this_studio/TypoPilot
wc -l data/final_250k/train.jsonl data/final_250k/val_2500.jsonl

cat <<'EOF'

Ready. Smoke test (30 steps, ~2 min):
  PYTHONPATH=src python -m typopilot.models.train_llm \
    --config configs/training/llama32_1b_lora.json --smoke

Real run (~1h on L40S):
  PYTHONPATH=src python -m typopilot.models.train_llm \
    --config configs/training/llama32_1b_lora.json

Outputs: experiments/llama32-1b-lora-001/merged  (full model, ready for GGUF)
EOF
