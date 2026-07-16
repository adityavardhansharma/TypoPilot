#!/usr/bin/env bash
# Convert the merged fine-tuned model to a quantised GGUF for LM Studio / llama.cpp.
# Run wherever you have the merged model (Studio or laptop). Produces a Q4_K_M
# GGUF (~0.8GB for a 1B model) that runs fast on CPU.
#
# Usage: bash scripts/export_gguf.sh experiments/llama32-1b-lora-001/merged out/typopilot-1b
set -euo pipefail

MERGED="${1:?path to the merged/ model dir}"
OUT="${2:-out/typopilot-1b}"
mkdir -p "$(dirname "$OUT")"

if [ ! -d llama.cpp ]; then
  echo "== cloning llama.cpp =="
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
  pip install -q -r llama.cpp/requirements.txt
fi

echo "== HF -> GGUF (f16) =="
python llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "${OUT}-f16.gguf" --outtype f16

echo "== quantise -> Q4_K_M =="
# Build the quantiser if needed (cmake), then quantise.
if [ ! -x llama.cpp/build/bin/llama-quantize ]; then
  cmake -S llama.cpp -B llama.cpp/build >/dev/null && cmake --build llama.cpp/build --target llama-quantize -j
fi
llama.cpp/build/bin/llama-quantize "${OUT}-f16.gguf" "${OUT}-Q4_K_M.gguf" Q4_K_M

echo "Done: ${OUT}-Q4_K_M.gguf"
echo "Load it in LM Studio. Use this exact prompt so it matches training:"
cat <<'EOF'

  Correct only the errors (spelling, spacing, capitalization, and obvious
  mistakes) in the text below. Keep the exact wording, meaning, and casual
  tone. Do not rephrase, translate, or add anything.

  Text: {your messy text}
  Corrected:
EOF
