#!/usr/bin/env python
"""
run_judges_batch.py — Judge a batch of candidate-run results files with one
local vLLM-served evaluator model, without reloading the model between files.

Mirrors run_evaluate.py's single-file flow, but launches the vLLM server once
and reuses it across every --results-file passed in, which matters for models
like Nemotron whose hybrid Mamba-attention architecture takes ~20 min to load.

Example:
    uv run python scripts/run_judges_batch.py \
        --results-file /shared/projects/philab-eve/ESAEOBench/candidate_runs/results_EVE-Instruct.json \
        --results-file /shared/projects/philab-eve/ESAEOBench/candidate_runs/results_Qwen3.6-27B.json \
        --evaluator-label Nemotron-2-12B \
        --evaluator-model nemotron-2-12b \
        --output-dir /shared/projects/philab-eve/ESAEOBench/judges_runs
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from agenticeoecosystem.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge several results files with one evaluator, one vLLM launch")
    parser.add_argument("--results-file", required=True, action="append", dest="results_files",
                        help="Full path to a results_*.json file; repeat for each candidate run")
    parser.add_argument("--evaluator-label", required=True, help="Short name for this evaluator (used as score key)")
    parser.add_argument("--evaluator-model", required=True, help="Evaluator model key (vLLM) or model ID")
    parser.add_argument("--evaluator-provider", default="vllm", help="Provider: vllm, eve, openai, …")
    parser.add_argument("--output-dir", required=True, help="Directory to write judged results into")
    parser.add_argument("--vllm-startup-timeout", type=int, default=1800)
    parser.add_argument("--skip-evaluated", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    setup_logging(log_file=f"judges_batch_{args.evaluator_label.replace(' ', '_')}")
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from agenticeoecosystem.evaluation.dataclasses import TaskSet
    from agenticeoecosystem.evaluation.evaluator import evaluate_solutions

    proc = None
    failures = []
    try:
        model_id = args.evaluator_model
        if args.evaluator_provider == "vllm":
            from agenticeoecosystem.utils.registry import launch_vllm, get_vllm_model_name
            proc = launch_vllm(args.evaluator_model, startup_timeout=args.vllm_startup_timeout)
            os.environ["VLLM_PORT"] = "8000"
            os.environ["EVALUATOR_PROVIDER"] = "vllm"
            model_id = get_vllm_model_name()
            logger.info("vLLM serving model as: %s", model_id)
        else:
            os.environ["EVALUATOR_PROVIDER"] = args.evaluator_provider

        for results_file in args.results_files:
            results_path = Path(results_file)
            gen_label = results_path.stem
            logger.info("=== Judging %s ===", results_path.name)
            try:
                tasks = TaskSet.read_from_file(results_path.name, folder=str(results_path.parent))
                logger.info("Loaded %d task(s) from %s", len(tasks), results_path)

                tasks = evaluate_solutions(
                    tasks,
                    model=model_id,
                    output_filename=None,  # avoid clobbering other files' intermediate checkpoints in GEOBENCH_RESULTS
                    skip_evaluated=args.skip_evaluated,
                    evaluator_label=args.evaluator_label,
                )
                # Distinct filename per evaluator — never overwrite another
                # judge's output file for the same candidate run.
                evaluator_safe = args.evaluator_label.replace(" ", "_")
                out_filename = f"{results_path.stem}_{evaluator_safe}.json"
                tasks.save_to_file(out_filename, folder=str(output_dir))
                logger.info("Updated results saved -> %s/%s", output_dir, out_filename)

                ev = tasks.score_summary_for_evaluator(args.evaluator_label)
                logger.info(
                    "Summary [%s] %s - evaluated: %d  match: %d  partial: %d  no_match: %d  accuracy: %.1f%%",
                    args.evaluator_label, gen_label, ev.get("n", 0),
                    ev.get("match", 0), ev.get("partial", 0), ev.get("no_match", 0),
                    ev.get("accuracy_full", 0.0) * 100,
                )
            except Exception as exc:
                logger.error("Evaluation failed for '%s' with evaluator '%s': %s",
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
