#!/usr/bin/env bash
# benchmark.sh — Bash orchestrator for the EO benchmark.
#
# Each model runs in its own Python process; GPU memory is explicitly freed
# between runs via pkill, so no vLLM state leaks across models.
#
# Two pipeline architectures are exercised, each swept across every model in
# the Phase 1 list below:
#   - "multi"  : the google-adk orchestrator (agents/multi_agent.py) — the
#                current multi-agent system. provider="eve" (EVE-Instruct)
#                routes through the EVE gateway; provider="vllm" routes
#                through a local vLLM server (see utils/registry.py).
#   - "single" : one LangGraph agent with all tools, same routing rules.
#
# Usage:
#   cd /path/to/agenticeoecosystem
#   bash scripts/benchmark.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Argument parsing
#   bash scripts/benchmark.sh           → all runnable tasks (default)
#   bash scripts/benchmark.sh --subset  → quick smoke run (Tool Grounding only —
#                                         see NOTE below on the curated subset)
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

# Single merged, actively-maintained dataset (superseded eo_benchmark_tasks.json /
# arxiv_gis_benchmark_tasks.json, which are now stale and no longer updated).
TASKS_FILES=("benchmark_tasks.json")

# All runnable task labels (excludes Placeholder / MOOC)
ALL_TASK_LABELS="Tool Grounding,Sequential Reasoning,Cross-agent Synthesis,Scope Awareness"

# NOTE: the old 30-task curated --subset (TASK_260419_*/TASK_260427_* IDs) referenced
# an earlier task-ID scheme that no longer exists in benchmark_tasks.json — those IDs
# would silently resolve to an empty TaskSet. Until a curated subset is re-picked from
# the current dataset, --subset falls back to a quick label-filtered smoke run instead
# of failing silently.
if [ "$SUBSET" = true ]; then
    echo "⚠ --subset curated task-ID list is stale for the current benchmark_tasks.json — falling back to a 'Tool Grounding' smoke run instead." >&2
    TASK_SELECTOR_ARGS=(--task-labels "Tool Grounding")
    TASK_MODE="smoke (Tool Grounding only — curated subset needs regenerating)"
else
    TASK_SELECTOR_ARGS=(--task-labels "$ALL_TASK_LABELS")
    TASK_MODE="all runnable tasks"
fi

OUTPUT_DIR="$REPO_ROOT/output/ESAEOBench/run_$(date +%Y%m%d_%H%M%S)"
TIMEOUT=1800        # 12B models — NemotronH hybrid Mamba-attn needs ~20 min on first run
TIMEOUT_LARGE=1800  # 14B–24B models or slow NFS loading
TASK_TIMEOUT=2400   # Max wall-clock seconds per task; a stuck task (e.g. an MCP
                     # tool call that keeps getting retried) is abandoned and the
                     # run moves to the next task instead of hanging indefinitely.

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
    # keep_alive=true skips this call's vLLM teardown (both the Python-level
    # _terminate_vllm and this function's cleanup_gpu), so a following call
    # for the SAME model can reuse the still-warm server instead of paying
    # the full reload cost twice — see launch_vllm()'s skip-relaunch check in
    # utils/registry.py. Always pass keep_alive=false (or omit it) on the
    # LAST call for a given model so the GPU is freed before the next model.
    local pipeline="$1" label="$2" model="$3" provider="$4" timeout="${5:-$TIMEOUT}" keep_alive="${6:-false}"
    local run_label="${label} [${pipeline}]"
    local extra_args=()
    [ "$keep_alive" = true ] && extra_args+=(--keep-vllm-alive)
    log "▶ Generate: $run_label  (pipeline=$pipeline, model=$model, provider=$provider, timeout=${timeout}s)"
    if uv run python "$SCRIPT_DIR/run_generate.py" \
            --label            "$run_label" \
            --model            "$model"    \
            --provider         "$provider" \
            --pipeline         "$pipeline" \
            --tasks-file       "${TASKS_FILES[@]}" \
            "${TASK_SELECTOR_ARGS[@]}" \
            --output-dir       "$OUTPUT_DIR" \
            --vllm-startup-timeout "$timeout" \
            --task-timeout     "$TASK_TIMEOUT" \
            "${extra_args[@]}"; then
        log "  ✓ Generate complete: $run_label"
    else
        log "  ✗ Generate FAILED: $run_label (continuing with next model)"
    fi
    [ "$keep_alive" = true ] || cleanup_gpu
}

run_evaluate() {
    local results_file="$1" eval_label="$2" eval_model="$3" eval_provider="$4" timeout="${5:-$TIMEOUT}"
    local gen_label
    gen_label="$(basename "${results_file%.json}")"
    log "▶ Evaluate: $gen_label  ×  $eval_label  (model=$eval_model, timeout=${timeout}s)"
    if uv run python "$SCRIPT_DIR/run_evaluate.py" \
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

# Each model is swept across both pipelines. For vLLM-backed models, the
# "multi" call passes keep_alive=true so the server launched for it stays up
# for the "single" call right after (same model → launch_vllm() detects it's
# already serving and skips the reload); the "single" call then tears it
# down before the next model. EVE-Instruct needs no vLLM either way.

# External API — always available, no vLLM involved
run_generate multi  "EVE-Instruct"      "EVE-Instruct"      "eve"
run_generate single "EVE-Instruct"      "EVE-Instruct"      "eve"

# Free models via vLLM
run_generate multi  "Nemotron-2-12B"    "nemotron-2-12b"    "vllm" $TIMEOUT       true
run_generate single "Nemotron-2-12B"    "nemotron-2-12b"    "vllm" $TIMEOUT

run_generate multi  "Phi4-Mini"         "phi4-mini"         "vllm" $TIMEOUT       true
run_generate single "Phi4-Mini"         "phi4-mini"         "vllm" $TIMEOUT

run_generate multi  "Qwen3-14B"         "qwen3-14b"         "vllm" $TIMEOUT_LARGE true
run_generate single "Qwen3-14B"         "qwen3-14b"         "vllm" $TIMEOUT_LARGE

run_generate multi  "Qwen2.5-14B"       "qwen2.5-14b"       "vllm" $TIMEOUT_LARGE true
run_generate single "Qwen2.5-14B"       "qwen2.5-14b"       "vllm" $TIMEOUT_LARGE

run_generate multi  "DeepSeek-R1-14B"   "deepseek-r1-14b"   "vllm" $TIMEOUT_LARGE true
run_generate single "DeepSeek-R1-14B"   "deepseek-r1-14b"   "vllm" $TIMEOUT_LARGE

# Gated models (comment out if licence not accepted on huggingface.co)
run_generate multi  "Gemma3-12B"        "gemma3-12b"        "vllm" $TIMEOUT       true
run_generate single "Gemma3-12B"        "gemma3-12b"        "vllm" $TIMEOUT
# INT8-quantized via bitsandbytes (weights ~44 GiB exceed A6000 in BF16)
# run_generate multi  "Mistral-Small-24B" "mistral-small-24b" "vllm" $TIMEOUT_LARGE true
# run_generate single "Mistral-Small-24B" "mistral-small-24b" "vllm" $TIMEOUT_LARGE
# run_generate multi  "Llama3.1-8B"       "llama3.1-8b"       "vllm" $TIMEOUT_LARGE true
# run_generate single "Llama3.1-8B"       "llama3.1-8b"       "vllm" $TIMEOUT_LARGE
# run_generate multi  "Llama3.2-3B"       "llama3.2-3b"       "vllm" $TIMEOUT_LARGE true
# run_generate single "Llama3.2-3B"       "llama3.2-3b"       "vllm" $TIMEOUT_LARGE

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
uv run python "$SCRIPT_DIR/build_table.py" --output-dir "$OUTPUT_DIR"

log "Done. Results in $OUTPUT_DIR"
