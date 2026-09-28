#!/usr/bin/env bash
# run_sample_agent_judge.sh — Generate + agent-judge a sample of benchmark tasks.
#
# Generalized version of run_sample5_agent_judge.sh: the task set, sample name,
# models and per-task timeout are all env-overridable, so the same script serves
# a 5-task smoke run, this 20-task run, or a larger one later.
#
# Two phases:
#   1. Generate solutions via the "multi" orchestrator pipeline (EVE-Instruct).
#   2. Score them with the tool-using agent-as-judge (EVE-Instruct), writing a
#      full Markdown transcript per task so you can see exactly what the judge
#      inspected and why it scored the way it did.
#
# Defaults to a 10-task, non-ERA5 sample covering all 9 remaining domains
# (eodash, cdse, gee, eumetsat, vlm, gis, graphrag, arxiv, mooc) across all four
# task categories (2 Tool Grounding / 4 Cross-agent Synthesis / 2 Sequential
# Reasoning / 2 Scope Awareness). The first five are the same tasks the
# earlier 5-task run used, so results stay comparable. Includes two
# MOOC/graphrag tasks deliberately, to confirm the MOOC RAG server (local
# Qdrant storage) works when only one run is active — see IMPORTANT below.
#
# IMPORTANT — never run this script (or run_generate.py / run_agent_judge.py)
# more than once concurrently. The MOOC RAG MCP server uses Qdrant in
# embedded/local-file mode (src/agenticeoecosystem/servers/esa_moocs/qdrant_storage/),
# which only allows ONE process to hold that storage folder open at a time.
# Two overlapping runs will make the second one's mooc_rag_agent fail with
# "Storage folder ... is already accessed by another instance of Qdrant
# client." — this already happened once and is not a broken tool, just a
# concurrency conflict. Check `ps aux | grep run_generate.py` before starting
# a new run if you're unsure whether a previous one is still alive.
#
# Usage:
#   bash scripts/run_sample_agent_judge.sh
#
# Override anything:
#   SAMPLE_NAME=Sample50 TASK_IDS="..." TASK_TIMEOUT=900 bash scripts/run_sample_agent_judge.sh
#
# Scheduling note: cron/launchd do NOT inherit your interactive shell's exports.
# Either put EVE_ENDPOINT/EVE_USERNAME/EVE_PASSWORD in the repo's .env, or source
# a credentials file in the crontab entry:
#   0 6 * * *  /bin/bash -c 'source ~/.eve_env && /path/to/run_sample_agent_judge.sh' >> ~/cron.log 2>&1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Configuration — edit or override via env vars before invoking
# ---------------------------------------------------------------------------
SAMPLE_NAME="${SAMPLE_NAME:-Sample10}"
TASKS_FILE="${TASKS_FILE:-benchmark_tasks.json}"

# 10 non-ERA5 tasks, still covering all 9 domains. Two (TASK_...027 and
# TASK_...068) touch the MOOC RAG server — see the concurrency note above.
TASK_IDS="${TASK_IDS:-TASK_260419_000001_100001,TASK_260419_000002_100002,TASK_260419_000004_100004,TASK_260419_000007_100007,TASK_260419_000008_100008,TASK_260419_000027_100027,TASK_260716_000001_700013,TASK_260713_000044_500023,TASK_260419_000068_100068,TASK_260419_000028_100028}"

GEN_MODEL="${GEN_MODEL:-EVE-Instruct}"
GEN_PROVIDER="${GEN_PROVIDER:-eve}"

# The judge runs on EVE too — claude-sonnet-4-6 404s on this gateway.
JUDGE_MODEL="${JUDGE_MODEL:-EVE-Instruct}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-eve}"
JUDGE_LABEL="${JUDGE_LABEL:-agent_judge}"
MAX_TURNS="${MAX_TURNS:-8}"

# Per-task wall-clock cap during generation — a stuck MCP call gets abandoned
# and the run moves on, instead of hanging a 20-task run indefinitely.
TASK_TIMEOUT="${TASK_TIMEOUT:-600}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
# tr, not ${VAR,,} — macOS ships bash 3.2, which has no lowercase expansion.
SAMPLE_SLUG="$(printf '%s' "$SAMPLE_NAME" | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/${SAMPLE_SLUG}_run_$RUN_STAMP}"
TRANSCRIPT_DIR="$OUTPUT_DIR/judge_transcripts"
RUN_LOG="$REPO_ROOT/logs/${SAMPLE_NAME}_agent_judge_$RUN_STAMP.log"

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs"

# ---------------------------------------------------------------------------
# Helpers & preflight
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

# If EVE creds aren't already exported, try the repo .env.
if [ -z "${EVE_ENDPOINT:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

for var in EVE_ENDPOINT EVE_USERNAME EVE_PASSWORD; do
    if [ -z "${!var:-}" ]; then
        log "✗ $var is not set in this process's environment."
        log "  cron/launchd don't inherit your interactive shell's exports — either add it"
        log "  to $REPO_ROOT/.env, or source a credentials file before calling this script."
        exit 1
    fi
done

# Deliberately NOT `uv run`: pyproject pins vllm==0.26.0, whose CUDA-only deps
# have no macOS wheels, so `uv`'s auto-sync fails before running anything —
# even though the eve pipeline never imports vllm/torch. This conda env has
# everything the eve path needs, built independently of uv.
PYTHON_BIN="${PYTHON_BIN:-/Users/kp/miniforge3/envs/lat/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
    log "✗ PYTHON_BIN not found or not executable: $PYTHON_BIN"
    exit 1
fi

# Heads-up on credentials that gate whether artifacts actually get produced —
# without these, imagery tasks fail at the tool level and the judge has no
# image to look at (it should score them NO_MATCH, which is correct but only
# exercises its failure path, not its vision path).
for var in CDSE_CLIENT_ID CDSE_CLIENT_SECRET; do
    [ -z "${!var:-}" ] && log "⚠ $var unset — CDSE imagery (fetch_sentinel2 etc.) will fail with 'invalid_client'."
done

N_TASKS="$(awk -F',' '{print NF}' <<< "$TASK_IDS")"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Phase 1/2: Generating solutions — $N_TASKS task(s)"
log "Model:   $GEN_MODEL ($GEN_PROVIDER, pipeline=multi)  task-timeout=${TASK_TIMEOUT}s"
log "Output:  $OUTPUT_DIR"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! "$PYTHON_BIN" "$SCRIPT_DIR/run_generate.py" \
        --label        "$SAMPLE_NAME" \
        --model        "$GEN_MODEL" \
        --provider     "$GEN_PROVIDER" \
        --pipeline     multi \
        --tasks-file   "$TASKS_FILE" \
        --task-ids     "$TASK_IDS" \
        --task-timeout "$TASK_TIMEOUT" \
        --output-dir   "$OUTPUT_DIR" 2>&1 | tee -a "$RUN_LOG"; then
    log "✗ Generation FAILED — aborting before the judge phase."
    exit 1
fi
log "✓ Generation complete"

RESULTS_FILE="$OUTPUT_DIR/results_${SAMPLE_NAME}.json"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Phase 2/2: Agent-as-judge — $JUDGE_MODEL ($JUDGE_PROVIDER)  max_turns=$MAX_TURNS"
log "Transcripts: $TRANSCRIPT_DIR"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! "$PYTHON_BIN" "$SCRIPT_DIR/run_agent_judge.py" \
        --results-file       "$RESULTS_FILE" \
        --evaluator-label    "$JUDGE_LABEL" \
        --evaluator-model    "$JUDGE_MODEL" \
        --evaluator-provider "$JUDGE_PROVIDER" \
        --max-turns          "$MAX_TURNS" \
        --transcript-dir     "$TRANSCRIPT_DIR" \
        --no-skip-evaluated 2>&1 | tee -a "$RUN_LOG"; then
    log "✗ Agent-judge scoring FAILED"
    exit 1
fi

log "✓ Done — scores in $RESULTS_FILE"
log "  Transcripts: $TRANSCRIPT_DIR"
log "  Run log:     $RUN_LOG"
