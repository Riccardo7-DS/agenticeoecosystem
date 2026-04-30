#!/usr/bin/env bash
# benchmark.sh — Bash orchestrator for the EO benchmark.
#
# Each model runs in its own Python process; GPU memory is explicitly freed
# between runs via pkill, so no vLLM state leaks across models.
#
# Usage:
#   cd /path/to/agenticeoecosystem
#   bash scripts/benchmark.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Argument parsing — mirrors benchmarking.py's --subset / --no-subset flag
#   bash scripts/benchmark.sh           → all runnable tasks (default)
#   bash scripts/benchmark.sh --subset  → curated 9-task subset only
# ---------------------------------------------------------------------------

SUBSET=false
for arg in "$@"; do
    case "$arg" in
        --subset)    SUBSET=true  ;;
        --no-subset) SUBSET=false ;;
    esac
done

# ---------------------------------------------------------------------------
# Configuration — edit these to match your run
# ---------------------------------------------------------------------------

# Both task files are loaded and merged for every run
TASKS_FILES=("eo_benchmark_tasks.json" "arxiv_gis_benchmark_tasks.json")

# ---------------------------------------------------------------------------
# 30-task curated subset — diverse coverage of all 6 MCP servers
#   EO Dashboard (STAC/narrative)  : 000001, 000016, 000005
#   GEE satellite download          : 000004, 000043, AG-000013, AG-000035, AG-000037
#   HybridRAG (ESA science strategy): 000014, 000064, 000066, 000072
#   ArXiv paper search              : AG-000001, AG-000002, AG-000005
#   GIS (geometry/raster/analysis)  : AG-000008, AG-000010, AG-000011, AG-000014, AG-000017, AG-000019
#   Cross-server (2+ agents)        : 000005, 000024, AG-000020–021, AG-000024, AG-000027, AG-000035–037
#   Scope Awareness                 : 000028, AG-000022
# ---------------------------------------------------------------------------
EO_STAC_IDS="TASK_260419_000001_100001,TASK_260419_000002_100002,TASK_260419_000004_100004,TASK_260419_000005_100005"
EO_RAG_IDS="TASK_260419_000014_100014,TASK_260419_000064_100064,TASK_260419_000066_100066,TASK_260419_000072_100072"
EO_SEQ_IDS="TASK_260419_000016_100016,TASK_260419_000043_100043"
EO_CROSS_IDS="TASK_260419_000024_100024"
EO_SCOPE_IDS="TASK_260419_000028_100028"
AG_ARXIV_IDS="TASK_260427_000001_200001,TASK_260427_000002_200002,TASK_260427_000005_200005"
AG_GIS_IDS="TASK_260427_000008_200008,TASK_260427_000010_200010,TASK_260427_000011_200011,TASK_260427_000014_200014,TASK_260427_000017_200017,TASK_260427_000019_200019"
AG_GEE_IDS="TASK_260427_000013_200013,TASK_260427_000035_200035,TASK_260427_000037_200037"
AG_CROSS_IDS="TASK_260427_000020_200020,TASK_260427_000021_200021,TASK_260427_000024_200024,TASK_260427_000027_200027,TASK_260427_000036_200036"
AG_SCOPE_IDS="TASK_260427_000022_200022"
SUBSET_TASK_IDS="${EO_STAC_IDS},${EO_RAG_IDS},${EO_SEQ_IDS},${EO_CROSS_IDS},${EO_SCOPE_IDS},${AG_ARXIV_IDS},${AG_GIS_IDS},${AG_GEE_IDS},${AG_CROSS_IDS},${AG_SCOPE_IDS}"

# All runnable task labels (excludes Placeholder / MOOC) — applied across both files
ALL_TASK_LABELS="Tool Grounding,Sequential Reasoning,Cross-agent Synthesis,Scope Awareness"

if [ "$SUBSET" = true ]; then
    TASK_SELECTOR_ARGS=(--task-ids "$SUBSET_TASK_IDS")
    TASK_MODE="subset (30 curated tasks)"
else
    TASK_SELECTOR_ARGS=(--task-labels "$ALL_TASK_LABELS")
    TASK_MODE="all runnable tasks"
fi

OUTPUT_DIR="$REPO_ROOT/output/geobenchX/run_$(date +%Y%m%d_%H%M%S)"
TIMEOUT=1800       # 12B models — NemotronH hybrid Mamba-attn needs ~20 min on first run
TIMEOUT_LARGE=1800 # 14B–24B models or slow NFS loading

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup_gpu() {
    log "  · Cleaning up GPU processes…"
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
}

run_generate() {
    local label="$1" model="$2" provider="$3" timeout="${4:-$TIMEOUT}"
    log "▶ Generate: $label  (model=$model, provider=$provider, timeout=${timeout}s)"
    if python "$SCRIPT_DIR/run_generate.py" \
            --label            "$label"    \
            --model            "$model"    \
            --provider         "$provider" \
            --tasks-file       "${TASKS_FILES[@]}" \
            "${TASK_SELECTOR_ARGS[@]}" \
            --output-dir       "$OUTPUT_DIR" \
            --vllm-startup-timeout "$timeout"; then
        log "  ✓ Generate complete: $label"
    else
        log "  ✗ Generate FAILED: $label (continuing with next model)"
    fi
    cleanup_gpu
}

run_evaluate() {
    local results_file="$1" eval_label="$2" eval_model="$3" eval_provider="$4" timeout="${5:-$TIMEOUT}"
    local gen_label
    gen_label="$(basename "${results_file%.json}")"
    log "▶ Evaluate: $gen_label  ×  $eval_label  (model=$eval_model, timeout=${timeout}s)"
    if python "$SCRIPT_DIR/run_evaluate.py" \
            --results-file         "$results_file" \
            --evaluator-label      "$eval_label"   \
            --evaluator-model      "$eval_model"   \
            --evaluator-provider   "$eval_provider" \
            --output-dir           "$OUTPUT_DIR"   \
            --vllm-startup-timeout "$timeout"; then
        log "  ✓ Evaluate complete: $gen_label × $eval_label"
    else
        log "  ✗ Evaluate FAILED: $gen_label × $eval_label (continuing)"
    fi
    cleanup_gpu
}

# ---------------------------------------------------------------------------
# Phase 1: Generate solutions
# Comment out models you don't want to run.
# ---------------------------------------------------------------------------

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Mode: $TASK_MODE"
log "Output: $OUTPUT_DIR"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Phase 1/2: Generating solutions"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# External API — always available, no vLLM needed
run_generate "EVE-Instruct"      "EVE-Instruct"      "eve"

# Free models via vLLM
run_generate "Nemotron-2-12B"    "nemotron-2-12b"    "vllm" $TIMEOUT
run_generate "Phi4-Mini"         "phi4-mini"         "vllm" $TIMEOUT
run_generate "Qwen3-14B"         "qwen3-14b"         "vllm" $TIMEOUT_LARGE
run_generate "Qwen2.5-14B"       "qwen2.5-14b"       "vllm" $TIMEOUT_LARGE
run_generate "DeepSeek-R1-14B"   "deepseek-r1-14b"   "vllm" $TIMEOUT_LARGE

# Gated models (comment out if licence not accepted on huggingface.co)
run_generate "Gemma3-12B"        "gemma3-12b"        "vllm" $TIMEOUT
# INT8-quantized via bitsandbytes (weights ~44 GiB exceed A6000 in BF16)
# run_generate "Mistral-Small-24B" "mistral-small-24b" "vllm" $TIMEOUT_LARGE
# run_generate "Llama3.1-8B"       "llama3.1-8b"       "vllm" $TIMEOUT_LARGE
# run_generate "Llama3.2-3B"       "llama3.2-3b"       "vllm" $TIMEOUT_LARGE

# ---------------------------------------------------------------------------
# Phase 2: Evaluate solutions
# Runs every evaluator over every results_*.json produced in Phase 1.
# ---------------------------------------------------------------------------

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Phase 2/2: Evaluating solutions"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for results_file in "$OUTPUT_DIR"/results_*.json; do
    [ -f "$results_file" ] || continue

    run_evaluate "$results_file" "Nemotron-2-12B" "nemotron-2-12b" "vllm" $TIMEOUT
    run_evaluate "$results_file" "Gemma3-12B"     "gemma3-12b"     "vllm" $TIMEOUT_LARGE
    run_evaluate "$results_file" "Qwen3-14B"      "qwen3-14b"      "vllm" $TIMEOUT_LARGE
done

# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Building comparison table"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "$SCRIPT_DIR/build_table.py" --output-dir "$OUTPUT_DIR"

log "Done. Results in $OUTPUT_DIR"
