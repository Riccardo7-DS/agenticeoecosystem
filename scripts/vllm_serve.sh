#!/usr/bin/env bash
set -euo pipefail

MODEL_BASE=/shared/home/rdercole/Riccardo/projects/agenticeoecosystem/models

usage() {
  echo "Usage: $0 --model <eve|mistral>"
  exit 1
}

MODEL_NAME=""
PORT=18000
GPUS=""
TENSOR_PARALLEL=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)              MODEL_NAME="$2";      shift 2 ;;
    --port)               PORT="$2";            shift 2 ;;
    --gpus)               GPUS="$2";            shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$MODEL_NAME" ]] && usage
[[ -n "$GPUS" ]] && export CUDA_VISIBLE_DEVICES="$GPUS"

case "$MODEL_NAME" in
  eve)
    MODEL_PATH="$MODEL_BASE/eve_hug/EVE-Instruct"
    SERVED_NAME="EVE-Instruct"
    ;;
  mistral)
    MODEL_PATH="$MODEL_BASE/mistral-small"
    SERVED_NAME="Mistral-Small-3.2-24B-Instruct-2506"
    ;;
  *) echo "Unknown model '$MODEL_NAME'. Choose 'eve' or 'mistral'."; exit 1 ;;
esac

FALLBACK_HEADERS=/tmp/libpydev3/usr/include
if [[ ! -f /usr/include/python3.12/Python.h && -f "$FALLBACK_HEADERS/python3.12/Python.h" ]]; then
  echo "System Python headers missing — using fallback from $FALLBACK_HEADERS"
  export CPATH="$FALLBACK_HEADERS/python3.12:$FALLBACK_HEADERS${CPATH:+:$CPATH}"
fi

echo "Serving $SERVED_NAME from $MODEL_PATH"

uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len 131072 \
  --max-num-batched-tokens 8000 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --dtype auto \
  --served-model-name "$SERVED_NAME"
