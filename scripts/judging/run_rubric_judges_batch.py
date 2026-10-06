#!/usr/bin/env python
"""
run_rubric_judges_batch.py — Score candidate-run results files against the
selected open-ended rubric profile, with one judge model, reusing a
single vLLM launch across every --results-file (same pattern as
run_judges_batch.py, but rubric_evaluator.py's per-criterion judge instead
of evaluator.py's reference-comparison judge).

Only the tasks present in --rubrics-file are scored; each output file
contains just that subset (self-contained per-experiment file), not the
full candidate results file.

Example:
    uv run python scripts/judging/run_rubric_judges_batch.py \
        --results-file /shared/projects/philab-eve/ESAEOBench/candidate_runs/results_EVE-Instruct.json \
        --results-file /shared/projects/philab-eve/ESAEOBench/candidate_runs/results_Qwen3.6-27B.json \
        --rubrics-file /shared/projects/philab-eve/ESAEOBench/judges_runs/rubric_83q_v3/rubrics_83q_v3.json \
        --evaluator-label Nemotron-2-12B \
        --evaluator-model nemotron-2-12b \
        --output-dir /shared/projects/philab-eve/ESAEOBench/judges_runs/rubric_83q_v3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from agenticeoecosystem.utils.logging import setup_logging


def load_rubrics(path: str) -> dict[str, list[dict]]:
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for entry in data:
        crit = entry.get("rubric", {}).get("criteria")
        if crit and len(crit) == 3:
            out[entry["task_ID"]] = crit
    return out


def load_rubric_protocols(path: str) -> dict[str, dict]:
    """Read explicit v4 instructions without changing legacy v3 inputs."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    protocols = {}
    for entry in entries:
        rubric = entry.get("rubric", {})
        if "protocol_version" not in rubric:
            continue
        if rubric.get("evidence_mode") not in {"trace", "evidence"}:
            raise ValueError(f"Invalid rubric profile for {entry['task_ID']}")
        if not rubric.get("scoring_instructions"):
            raise ValueError(f"Missing scoring instructions for {entry['task_ID']}")
        protocols[entry["task_ID"]] = {
            key: rubric[key] for key in ("protocol_version", "evidence_mode", "scoring_instructions")
        }
    return protocols


def main() -> int:
    parser = argparse.ArgumentParser(description="Rubric-judge several candidate results files with one evaluator, one vLLM launch")
    parser.add_argument("--results-file", required=True, action="append", dest="results_files",
                        help="Full path to a candidate results_*.json file; repeat for each candidate run")
    parser.add_argument("--rubrics-file", required=True, help="Path to the rubrics JSON (task_ID -> rubric.criteria)")
    parser.add_argument("--evaluator-label", required=True, help="Short name for this evaluator (used as score key + output suffix)")
    parser.add_argument("--evaluator-model", required=True, help="Evaluator model key (vLLM) or model ID")
    parser.add_argument("--evaluator-provider", default="vllm", help="Provider: vllm, eve, openai, …")
    parser.add_argument("--output-dir", required=True, help="Directory to write judged results into")
    parser.add_argument("--vllm-startup-timeout", type=int, default=1800)
    parser.add_argument("--skip-evaluated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inspect-evidence", action="store_true",
                        help="Run offline AAJ artifact inspection before rubric scoring (additional judge calls)")
    parser.add_argument("--inspection-evaluator-label", help="Reuse candidate-bound AAJ evidence under this label")
    parser.add_argument("--artifact-manifest", help="Candidate-owned artifact manifest; allowed with exactly one results file")
    args = parser.parse_args()
    if args.artifact_manifest and len(args.results_files) != 1:
        parser.error("Use one results file per artifact manifest to preserve candidate ownership.")
    artifact_manifest = json.loads(Path(args.artifact_manifest).read_text()) if args.artifact_manifest else None
    if artifact_manifest is not None and (not isinstance(artifact_manifest, dict) or any(
        not isinstance(paths, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths)
        for paths in artifact_manifest.values())):
        parser.error("Artifact manifest must map task IDs to lists of absolute paths.")

    setup_logging(log_file=f"rubric_judges_batch_{args.evaluator_label.replace(' ', '_')}")
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rubrics_by_id = load_rubrics(args.rubrics_file)
    rubric_protocols = load_rubric_protocols(args.rubrics_file)
    modes = {p["evidence_mode"] for p in rubric_protocols.values()}
    if len(modes) > 1 or (rubric_protocols and len(rubric_protocols) != len(rubrics_by_id)):
        parser.error("Use one consistent rubric profile per scoring batch.")
    if "trace" in modes and (args.inspect_evidence or args.inspection_evaluator_label or args.artifact_manifest):
        parser.error("Trace scoring uses uniform calls/arguments/answers; run artifact verification separately.")
    logger.info("Rubric protocol: %s", next(iter(modes), "legacy evidence-required"))
    logger.info("Loaded %d rubrics from %s", len(rubrics_by_id), args.rubrics_file)
    rubric_task_ids = list(rubrics_by_id.keys())

    from agenticeoecosystem.evaluation.dataclasses import TaskSet
    from agenticeoecosystem.evaluation.judges.rubric_evaluator import rubric_evaluate_solutions

    proc = None
    failures = []
    try:
        model_id = args.evaluator_model
        eval_provider = args.evaluator_provider
        if args.evaluator_provider == "vllm":
            from agenticeoecosystem.utils.registry import launch_vllm, get_vllm_model_name
            proc = launch_vllm(args.evaluator_model, startup_timeout=args.vllm_startup_timeout)
            os.environ["VLLM_PORT"] = "8000"
            os.environ["EVALUATOR_PROVIDER"] = "vllm"
            model_id = get_vllm_model_name()
            logger.info("vLLM serving model as: %s", model_id)

        for results_file in args.results_files:
            results_path = Path(results_file)
            logger.info("=== Rubric-judging %s ===", results_path.name)
            try:
                full_tasks = TaskSet.read_from_file(results_path.name, folder=str(results_path.parent))
                tasks = full_tasks.select_by_ids(rubric_task_ids)
                logger.info("Loaded %d task(s), %d overlap the rubric set", len(full_tasks), len(tasks))
                if len(tasks) == 0:
                    logger.warning("No overlapping tasks — skipping %s", results_path.name)
                    continue

                inspection_label = args.inspection_evaluator_label
                if args.inspect_evidence:
                    from agenticeoecosystem.evaluation.judges.agent_judge import evaluate_solutions_agentic
                    inspection_label = inspection_label or f"{args.evaluator_label}_evidence"
                    evaluate_solutions_agentic(tasks, model=model_id, provider=eval_provider,
                                              evaluator_label=inspection_label, artifact_manifest=artifact_manifest,
                                              skip_evaluated=False)
                tasks = rubric_evaluate_solutions(
                    tasks,
                    rubrics_by_id=rubrics_by_id,
                    model=model_id,
                    skip_evaluated=args.skip_evaluated,
                    evaluator_label=args.evaluator_label,
                    provider=eval_provider,
                    inspection_label=inspection_label,
                    rubric_protocols=rubric_protocols,
                )

                evaluator_safe = args.evaluator_label.replace(" ", "_")
                out_filename = f"results_{results_path.stem.removeprefix('results_')}_{evaluator_safe}.json"
                tasks.save_to_file(out_filename, folder=str(output_dir))
                logger.info("Saved -> %s/%s", output_dir, out_filename)

                raw_scores = [t.evaluator_scores_raw.get(args.evaluator_label) for t in tasks]
                raw_scores = [s for s in raw_scores if s is not None]
                buckets = [t.evaluator_scores.get(args.evaluator_label) for t in tasks]
                buckets = [s for s in buckets if s is not None]
                unverified = sum(any(c.get("score") is None for c in
                                     t.evaluator_criteria.get(args.evaluator_label, [])) for t in tasks)
                logger.info("Evidence coverage [%s]: %d/%d complete rubric scores; %d have unverified criteria.",
                            args.evaluator_label, len(raw_scores), len(tasks), unverified)
                if raw_scores:
                    logger.info(
                        "Summary [%s] %s - scored: %d/%d  mean: %.3f/2.0  match: %d  partial: %d  no_match: %d",
                        args.evaluator_label, results_path.stem, len(raw_scores), len(tasks),
                        sum(raw_scores) / len(raw_scores),
                        sum(1 for s in buckets if s == 2), sum(1 for s in buckets if s == 1), sum(1 for s in buckets if s == 0),
                    )
            except Exception as exc:
                logger.error("Rubric evaluation failed for '%s' with evaluator '%s': %s",
                             results_path.name, args.evaluator_label, exc, exc_info=True)
                failures.append(results_path.name)

        return 1 if failures else 0

    finally:
        if proc is not None:
            import signal
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
