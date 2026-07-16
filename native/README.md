# TypoPilot native inference harness

Small CPU harness for the fine-tuned Llama correction model. It embeds the
`llama.cpp` library directly; it does not require LM Studio, Python, or a local
HTTP server at runtime.

Build on Windows with Visual Studio Build Tools:

```powershell
$cmake = "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake -S native -B build/typopilot-native -A x64 `
  -DLLAMA_CPP_DIR="$PWD/build/llama.cpp" -DGGML_NATIVE=ON
& $cmake --build build/typopilot-native --config Release --target typopilot-harness -j 8
```

Single correction:

```powershell
build/typopilot-native/Release/typopilot-harness.exe `
  --model experiments/llama32-1b-lora-001/gguf/typopilot-llama32-1b-q4_k_m.gguf `
  --threads 8 --text "waht is teh weahter today?"
```

Batch benchmark input is UTF-8 TSV (`id<TAB>text`):

```powershell
typopilot-harness.exe --model model.gguf --threads 8 --input-tsv cases.tsv
```

Batch mode emits one JSON object per line with output text and latency. The
model is loaded once and kept resident across every request.

