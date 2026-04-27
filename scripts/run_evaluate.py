#!/usr/bin/env python
"""
run_evaluate.py — Single-evaluator scoring pass on one results file.

Reads results_<label>.json, runs evaluate_solutions with one evaluator,
saves the updated file in place, then exits.

Exits 0 on success, 1 on failure.
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from agenticeoecosystem.utils.logging import setup_logging


def _terminate_vllm(proc) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one results file with one evaluator")
    parser.add_argument("--results-file",       required=True, help="Full path to results_*.json")
    parser.add_argument("--evaluator-label",    required=True, help="Short name for this evaluator (used as score key)")
    parser.add_argument("--evaluator-model",    required=True, help="Evaluator model key or HuggingFace ID")
    parser.add_argument("--evaluator-provider", default="vllm", help="Provider: vllm, eve, openai, …")
    parser.add_argument("--output-dir",         default=None,
                        help="Save updated results here (default: same folder as results-file)")
    parser.add_argument("--vllm-startup-timeout", type=int, default=600)
    parser.add_argument("--skip-evaluated",     action=argparse.BooleanOptionalAction, default=True,
                        help="Skip tasks already scored by this evaluator (default: True)")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    gen_label    = results_path.stem  # e.g. "results_Qwen3-14B"
    eval_safe    = args.evaluator_label.replace(" ", "_")
    setup_logging(log_file=f"evaluate_{gen_label}_{eval_safe}")
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    from agenticeoecosystem.evaluation.dataclasses import TaskSet
    tasks = TaskSet.read_from_file(results_path.name, folder=str(results_path.parent))
    logger.info("Loaded %d task(s) from %s", len(tasks), results_path)

    proc = None
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

        from agenticeoecosystem.evaluation.evaluator import evaluate_solutions
        tasks = evaluate_solutions(
            tasks,
            model=model_id,
            output_filename=results_path.name,
            skip_evaluated=args.skip_evaluated,
            evaluator_label=args.evaluator_label,
        )
        tasks.save_to_file(results_path.name, folder=str(output_dir))
        logger.info("Updated results saved → %s/%s", output_dir, results_path.name)

        ev = tasks.score_summary_for_evaluator(args.evaluator_label)
        logger.info(
            "Summary [%s] — evaluated: %d  match: %d  partial: %d  no_match: %d  accuracy: %.1f%%",
            args.evaluator_label, ev.get("n", 0),
            ev.get("match", 0), ev.get("partial", 0), ev.get("no_match", 0),
            ev.get("accuracy_full", 0.0) * 100,
        )
        return 0

    except Exception as exc:
        logger.error("Evaluation failed for '%s' with evaluator '%s': %s",
                     results_path.name, args.evaluator_label, exc, exc_info=True)
        return 1

    finally:
        _terminate_vllm(proc)


if __name__ == "__main__":
    sys.exit(main())
