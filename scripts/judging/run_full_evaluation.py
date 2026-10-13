#!/usr/bin/env python
"""Run configured task-specific judges; never invoke reference-workflow matching."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agenticeoecosystem.evaluation.runners.full_evaluation import FullEvaluationConfig, FullEvaluationRunner


def parse_config() -> FullEvaluationConfig:
    """Parse the batch CLI without launching models or evaluating candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", required=True, action="append", type=Path,
                        help="Candidate JSON; repeat for multiple candidates")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evaluator-model", required=True, help="Served judge model ID")
    parser.add_argument("--evaluator-provider", default="eve",
                        help="eve, openrouter, openai, custom or vllm; vllm server must already be running")
    parser.add_argument("--evaluator-label", default="full_evaluation_v1")
    parser.add_argument("--evaluation-strategies", type=Path, default=FullEvaluationConfig.strategies_file)
    parser.add_argument("--ground-truths", type=Path, default=FullEvaluationConfig.ground_truths_file)
    parser.add_argument("--rubrics-file", type=Path, default=FullEvaluationConfig.rubrics_file)
    parser.add_argument("--benchmark-file", type=Path, default=FullEvaluationConfig.benchmark_file)
    parser.add_argument("--tool-agent-mapping", type=Path, default=FullEvaluationConfig.tool_agent_mapping,
                        help="Tool ownership contract for derived agent trajectory metrics")
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--reference-validations", type=Path,
                        help="Candidate-bound source/support validity attestations, keyed by task ID")
    parser.add_argument("--candidate-references", type=Path,
                        help="One candidate's recovered references (recover_candidate_numeric_evidence.py), keyed by task ID")
    parser.add_argument("--task-ids", nargs="+", default=[])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--gis-max-turns", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Write the dispatch plan without model calls")
    args = parser.parse_args()
    return FullEvaluationConfig(
        results_files=tuple(args.results_file), output_dir=args.output_dir,
        model=args.evaluator_model, provider=args.evaluator_provider, label=args.evaluator_label,
        strategies_file=args.evaluation_strategies, ground_truths_file=args.ground_truths,
        rubrics_file=args.rubrics_file, benchmark_file=args.benchmark_file,
        tool_agent_mapping=args.tool_agent_mapping,
        artifact_manifest=args.artifact_manifest, reference_validations=args.reference_validations,
        candidate_references=args.candidate_references,
        task_ids=tuple(args.task_ids), temperature=args.temperature, seed=args.seed,
        max_turns=args.max_turns, gis_max_turns=args.gis_max_turns,
        resume=args.resume, dry_run=args.dry_run)


def main() -> int:
    """Write durable results and return 2 for incomplete coverage, 1 for failure."""
    if "--recover-saved-results" in sys.argv:
        from agenticeoecosystem.evaluation.runners.recovery import RecoveryConfig, SavedEvaluationRecovery

        parser = argparse.ArgumentParser(description="Recover saved deterministic metrics without model calls")
        parser.add_argument("--recover-saved-results", required=True, type=Path,
                            help="Root containing saved evaluation summary.json files")
        parser.add_argument("--output-dir", required=True, type=Path)
        args = parser.parse_args()
        report = SavedEvaluationRecovery(RecoveryConfig(args.recover_saved_results, args.output_dir)).run()
        print(f"Recovered {len(report['candidates'])} candidate evaluations into {args.output_dir}")
        return 0
    from dotenv import load_dotenv

    config = parse_config()
    if not config.dry_run:
        load_dotenv(ROOT / ".env")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(config.output_dir / "judging.log", encoding="utf-8")],
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        report = FullEvaluationRunner(config).run()
    except (OSError, ValueError, RuntimeError) as error:
        logging.exception("Full evaluation failed: %s", error)
        return 1
    print(f"{'Plan' if config.dry_run else 'Results'} saved in {config.output_dir}")
    return 0 if config.dry_run or not report["incomplete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
