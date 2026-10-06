#!/usr/bin/env bash
# Refresh local judge history and rejudging inventory; never launch evaluations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
shopt -s nullglob

registry=data/judge_runs
strategies=data/tasks/task_evaluation_strategies.json
benchmark=data/tasks/benchmark_tasks.json
mkdir -p "$registry"/{historical,pending,current}
generated_at=$(date -u +%FT%TZ)

link_source() {
    local source=$1 destination=$2
    mkdir -p "$(dirname "$destination")"
    ln -sfn "$(realpath --relative-to="$(dirname "$destination")" "$source")" "$destination"
}

for source in data/evaluation_runs/* data/rubric_runs/*; do
    [[ -d "$source" ]] || continue
    name=$(basename "$source")
    case "$name" in
        *reference*) category=superseded_reference; status=superseded; reason='Generic reference-workflow matching is retired; exclude from current rankings.' ;;
        numerical*) category=numerical_diagnostics; status=partial_diagnostic; reason='Historical extraction/reference audit; reuse only after validating question, answer, reference and evidence fingerprints.' ;;
        rubric*) category=historical_rubrics; status=historical_protocol; reason='Historical rubric scores; trace and evidence profiles are separate. Revised or rerun answers require fresh scoring.' ;;
        *) category=unclassified; status=review_required; reason='Method and provenance require review before use.' ;;
    esac
    destination="$registry/historical/$category/$name"
    mkdir -p "$destination"
    link_source "$source" "$destination/outputs"
    jq -n --arg source "$source" --arg status "$status" --arg reason "$reason" \
        '{source:$source,status:$status,eligible_for_current_ranking:false,reason:$reason}' > "$destination/status.json"
done

# Preserve original log paths and give historical judge logs a categorized view.
for source in logs/evaluate_*.log logs/judges_batch_*.log logs/rubric_judges_batch_*.log logs/agent_judge_*.log; do
    case "$(basename "$source")" in
        rubric_judges*) category=historical_rubrics ;;
        agent_judge*) category=aaj_review_required ;;
        *) category=superseded_reference ;;
    esac
    link_source "$source" "$registry/historical/$category/logs/$(basename "$source")"
done

# Build one inventory per candidate, overlaying latest rerun checkpoints by ID.
# Score dictionaries are deliberately excluded from pending inventories.
for base in data/evaluation_runs/all_non_scope_reference_20261002/results_*_Mistral-Small-3.2_reference_all.json; do
    name=$(basename "$base")
    candidate=${name#results_}
    candidate=${candidate%_Mistral-Small-3.2_reference_all.json}
    reruns=(data/candidate_reruns/missing_and_required_20261003/*/"results_${candidate}_missing_rerun.json"
        data/candidate_reruns/missing_and_required_20261003/openrouter/"$candidate"/"results_${candidate}_missing_rerun.json")
    sources=("$base" "${reruns[@]}")
    source_records=()
    for source in "${sources[@]}"; do
        [[ -f "$source" ]] || continue
        # Fingerprint the exact JSON read, including a checkpoint being updated.
        snapshot=$(mktemp /tmp/judge-inventory.XXXXXX)
        cp "$source" "$snapshot"
        source_hash=$(sha256sum "$snapshot")
        source_hash=${source_hash%% *}
        source_records+=("$(jq --arg path "$source" --arg hash "$source_hash" \
            '{path:$path,sha256:$hash,metadata:.metadata,tasks:[.tasks[]|
              {task_ID,task_text,
               generated_solution:(if .generated_solution==null then null else true end),
               has_artifact_records:((.candidate_artifacts//[]|length)>0)}]}' "$snapshot")")
        unlink "$snapshot"
    done
    printf '%s\n' "${source_records[@]}" | jq -s \
        --slurpfile strategies "$strategies" --slurpfile benchmark "$benchmark" \
        --arg candidate "$candidate" --arg timestamp "$generated_at" '
        ($strategies[0].tasks|map({key:.task_ID,value:.})|from_entries) as $cfg |
        ($benchmark[0].tasks|map({key:.task_ID,value:.task_text})|from_entries) as $questions |
        . as $sources |
        (reduce .[] as $source ({}; reduce $source.tasks[] as $task (.;
            .[$task.task_ID]={task:$task,source:$source.path}))) as $latest |
        {candidate:$candidate,generated_at:$timestamp,status:"needs_current_protocol_judging",
         strategies_file:"data/tasks/task_evaluation_strategies.json",
         reference_comparison_retired:true,
         sources:[$sources[]|{path,sha256}],
         tasks:[$latest[] | .task as $task |
            {task_ID:$task.task_ID,answer_source:.source,
             status:(if $task.generated_solution==null then "missing_answer" else "needs_rejudge" end),
             current_question_matches:($task.task_text==$questions[$task.task_ID]),
             methods:($cfg[$task.task_ID].evaluation_methods//[]),
             primary_evaluator:($cfg[$task.task_ID].primary_evaluator//null),
             has_artifact_records:$task.has_artifact_records}]|
             sort_by(.task_ID)} |
        .summary={questions:(.tasks|length),
          needs_rejudge:([.tasks[]|select(.status=="needs_rejudge")]|length),
          missing_answers:([.tasks[]|select(.status=="missing_answer")]|length),
          question_version_review:([.tasks[]|select(.current_question_matches|not)]|length),
          aaj_assigned:([.tasks[]|select(.methods|index("agent_as_judge"))]|length),
          gis_assigned:([.tasks[]|select(.methods|index("gis_verification_agent"))]|length)}
        ' > "$registry/pending/$candidate.json"
done

# Register actual unified-runner reports, keeping incomplete coverage visible.
current_records=()
for report in "$registry"/current/*/summary.json; do
    current_records+=("$(jq --arg path "$report" '
      select(.protocol_version=="task_specific_v1" and .dry_run==false) |
      {report:$path,incomplete,candidates:[.candidates[]|
       {source,output,questions,task_statuses,method_coverage,validated_means}]}' "$report")")
done
current_json=$(printf '%s\n' "${current_records[@]}" | jq -s '.')
jq -s --arg timestamp "$generated_at" --argjson current "$current_json" \
    '{generated_at:$timestamp,current_protocol_runs_registered:($current|length),current_runs:$current,
      candidates:map({candidate,status,summary,sources})}' \
    "$registry"/pending/*.json > "$registry/index.json"

cat > "$registry/README.md" <<'EOF'
# Judge run index

Start with `index.json` and `pending/`. Refresh with
`bash scripts/reporting/organize_judge_runs.sh` as candidate generation progresses.

- `historical/superseded_reference/`: retired workflow-matching scores and logs.
- `historical/historical_rubrics/`: older rubric outputs; preserve their profile.
- `historical/numerical_diagnostics/`: partial numerical audits, not full rankings.
- `pending/`: per-question source, required methods and rejudging/version status.
- `current/`: reserved for explicitly registered current-protocol evaluations.

Historical views link to original files so existing scripts keep working. No
original results or logs are moved, deleted or edited. Judge logs inside an
output folder remain accessible through its `outputs` link.

Pending inventories overlay latest rerun answers on historical candidate tasks;
they contain no inherited scores. A null rerun answer stays missing. Historical
answers with different question wording require version review before scoring.
Configured methods are requirements, not evidence that a validator exists or ran.

The GLM checkpoint may still be changing. Refresh after generation finishes.
`needs_rejudge` means an answer is recorded, not that it is correct or complete.
See `index.json` for registered current runs and their incomplete-coverage status.
EOF

printf 'Judge inventory refreshed: %s/index.json\n' "$registry"
