#!/usr/bin/env bash
# benchmark.sh — Bash orchestrator for the EO benchmark.
#
# Each model runs in its own Python process; GPU memory is explicitly freed
# between runs via pkill, so no vLLM state leaks across models.
#
# Two pipeline architectures are exercised:
#   - "multi"  : the google-adk orchestrator (agents/multi_agent.py) — the
#                current multi-agent system. It always runs EVE-Instruct via
#                the EVE endpoint regardless of --model/--provider, so it is
#                generated ONCE (not swept across every model below).
#   - "single" : one LangGraph agent with all tools, swept across every model
#                in the Phase 1 list below, same as before.
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

run_generate_single() {
    local label="$1" model="$2" provider="$3" timeout="${4:-$TIMEOUT}"
    local run_label="${label} [single]"
    log "▶ Generate: $run_label  (model=$model, provider=$provider, timeout=${timeout}s)"
    if python "$SCRIPT_DIR/run_generate.py" \
            --label            "$run_label" \
            --model            "$model"    \
            --provider         "$provider" \
            --pipeline         single \
            --tasks-file       "${TASKS_FILES[@]}" \
            "${TASK_SELECTOR_ARGS[@]}" \
            --output-dir       "$OUTPUT_DIR" \
            --vllm-startup-timeout "$timeout"; then
        log "  ✓ Generate complete: $run_label"
    else
        log "  ✗ Generate FAILED: $run_label (continuing with next model)"
    fi
    cleanup_gpu
}

run_generate_multi_once() {
    # pipeline="multi" (google-adk orchestrator) always runs EVE-Instruct via
    # the EVE endpoint regardless of --model/--provider — sweeping it across
    # every model in the loop below would just relaunch vLLM for nothing and
    # silently re-run EVE-Instruct each time, so it's generated once here.
    local run_label="EVE-Instruct [multi]"
    log "▶ Generate: $run_label  (google-adk orchestrator, EVE-Instruct only)"
    if python "$SCRIPT_DIR/run_generate.py" \
            --label      "$run_label" \
            --model      "EVE-Instruct" \
            --provider   "eve" \
            --pipeline   multi \
            --tasks-file "${TASKS_FILES[@]}" \
            "${TASK_SELECTOR_ARGS[@]}" \
            --output-dir "$OUTPUT_DIR"; then
        log "  ✓ Generate complete: $run_label"
    else
        log "  ✗ Generate FAILED: $run_label"
    fi
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

# Multi-agent (google-adk orchestrator) — EVE-Instruct only, runs once
run_generate_multi_once

# Single-agent sweep across models
# External API — always available, no vLLM needed
run_generate_single "EVE-Instruct"      "EVE-Instruct"      "eve"

# Free models via vLLM
run_generate_single "Nemotron-2-12B"    "nemotron-2-12b"    "vllm" $TIMEOUT
run_generate_single "Phi4-Mini"         "phi4-mini"         "vllm" $TIMEOUT
run_generate_single "Qwen3-14B"         "qwen3-14b"         "vllm" $TIMEOUT_LARGE
run_generate_single "Qwen2.5-14B"       "qwen2.5-14b"       "vllm" $TIMEOUT_LARGE
run_generate_single "DeepSeek-R1-14B"   "deepseek-r1-14b"   "vllm" $TIMEOUT_LARGE

# Gated models (comment out if licence not accepted on huggingface.co)
run_generate_single "Gemma3-12B"        "gemma3-12b"        "vllm" $TIMEOUT
# INT8-quantized via bitsandbytes (weights ~44 GiB exceed A6000 in BF16)
# run_generate_single "Mistral-Small-24B" "mistral-small-24b" "vllm" $TIMEOUT_LARGE
# run_generate_single "Llama3.1-8B"       "llama3.1-8b"       "vllm" $TIMEOUT_LARGE
# run_generate_single "Llama3.2-3B"       "llama3.2-3b"       "vllm" $TIMEOUT_LARGE

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
