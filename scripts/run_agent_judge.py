#!/usr/bin/env python
"""
run_agent_judge.py — Tool-using agent-as-judge scoring pass on one results file.

Same shape as run_evaluate.py, but calls evaluation/agent_judge.py's
tool-calling judge instead of evaluator.py's one-shot LLM-as-judge: the
judge can list output/<server>/, open files the trace produced, and
actually look at image/GeoTIFF previews before scoring.

Reads results_<label>.json, runs evaluate_solutions_agentic with one
evaluator label, saves the updated file in place, then exits.

Exits 0 on success, 1 on failure.
"""

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from agenticeoecosystem.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one results file with the tool-using agent-as-judge")
    parser.add_argument("--results-file",       required=True, help="Full path to results_*.json")
    parser.add_argument("--evaluator-label",    default="agent_judge", help="Short name for this evaluator (used as score key)")
    parser.add_argument("--evaluator-model",    required=True, help="Vision-capable judge model key")
    parser.add_argument("--evaluator-provider", default="eve", help="Provider: eve, openai, custom, vllm, …")
    parser.add_argument("--max-turns",          type=int, default=8, help="Max tool-call turns per task before giving up")
    parser.add_argument("--output-dir",         default=None,
                        help="Save updated results here (default: same folder as results-file)")
    parser.add_argument("--transcript-dir",     default=None,
                        help="Write a full Markdown transcript per task (every tool call, full tool "
                             "result, final verdict) here. Default: <output-dir>/judge_transcripts/")
    parser.add_argument("--skip-evaluated",     action=argparse.BooleanOptionalAction, default=True,
                        help="Skip tasks already scored by this evaluator (default: True)")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    gen_label    = results_path.stem
    eval_safe    = args.evaluator_label.replace(" ", "_")
    setup_logging(log_file=f"agent_judge_{gen_label}_{eval_safe}")
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = args.transcript_dir or str(output_dir / "judge_transcripts")

    from agenticeoecosystem.evaluation.dataclasses import TaskSet
    tasks = TaskSet.read_from_file(results_path.name, folder=str(results_path.parent))
    logger.info("Loaded %d task(s) from %s", len(tasks), results_path)

    try:
        from agenticeoecosystem.evaluation.agent_judge import evaluate_solutions_agentic
        tasks = evaluate_solutions_agentic(
            tasks,
            model=args.evaluator_model,
            provider=args.evaluator_provider,
            max_turns=args.max_turns,
            output_filename=results_path.name,
            output_folder=str(output_dir),
            skip_evaluated=args.skip_evaluated,
            evaluator_label=args.evaluator_label,
            transcript_dir=transcript_dir,
        )
        logger.info("Judge transcripts written → %s", transcript_dir)
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
        logger.error("Agent-judge evaluation failed for '%s' with evaluator '%s': %s",
                     results_path.name, args.evaluator_label, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
