#!/usr/bin/env python
"""
run_generate.py — Single-model generation run.

Launches vLLM (if provider=vllm), runs generate_solutions, saves results, then exits.
Each invocation is a fresh Python process, so the GPU is fully released on exit.

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
    parser = argparse.ArgumentParser(description="Generate solutions for one model")
    parser.add_argument("--label",                required=True,  help="Human-readable label (used in output filename)")
    parser.add_argument("--model",                required=True,  help="Model key or HuggingFace ID")
    parser.add_argument("--provider",             default="eve",  help="LangChain provider: eve, vllm, openai, …")
    parser.add_argument("--temperature",          type=float, default=0.0)
    parser.add_argument("--tasks-file",           required=True, nargs='+', help="Tasks JSON filename(s); multiple files are merged")
    parser.add_argument("--tasks-folder",         default=None,   help="Folder containing tasks file (default: data/tasks)")
    parser.add_argument("--task-ids",             default=None,   help="Comma-separated task IDs to select")
    parser.add_argument("--task-labels",          default=None,   help="Comma-separated task labels to select (e.g. 'Tool Grounding,Sequential Reasoning')")
    parser.add_argument("--output-dir",           required=True,  help="Directory for results_<label>.json")
    parser.add_argument("--vllm-startup-timeout", type=int, default=600)
    parser.add_argument("--skip-solved",          action="store_true", help="Skip tasks that already have a solution")
    parser.add_argument("--pipeline",             default="multi", choices=["multi", "single_adk", "single", "multi_legacy"],
                        help="Pipeline mode: 'multi' (google-adk orchestrator, the current "
                             "multi-agent system — EVE-Instruct only), 'single_adk' (google-adk, "
                             "ONE agent with every sub-agent's toolset attached directly, no "
                             "routing — EVE-Instruct only, same as 'multi'), 'single' (one "
                             "LangChain agent, all tools, sweepable across any model), or "
                             "'multi_legacy' (older LangGraph supervisor + sub-agents in "
                             "client.py, superseded by 'multi')")
    parser.add_argument("--agent-models",         default=None,
                        help='JSON per-agent model overrides for pipeline=multi_legacy, e.g. \'{"gis": ["mistral-small-24b", "vllm"]}\'')
    args = parser.parse_args()

    label_safe = args.label.replace(" ", "_")
    setup_logging(log_file=f"generate_{label_safe}_{args.pipeline}")
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from agenticeoecosystem.definitions import TASKS_FOLDER
    from agenticeoecosystem.evaluation.dataclasses import TaskSet

    folder = args.tasks_folder or str(TASKS_FOLDER)
    tasks = None
    for tf in args.tasks_file:
        ts = TaskSet.read_from_file(tf, folder=folder)
        if tasks is None:
            tasks = ts
        else:
            tasks.extend(ts.tasks)
    if args.task_ids:
        ids = [x.strip() for x in args.task_ids.split(",")]
        tasks = tasks.select_by_ids(ids)
    elif args.task_labels:
        labels = [x.strip() for x in args.task_labels.split(",")]
        tasks = tasks.select_by_labels(labels)
    logger.info("Loaded %d task(s) from %d file(s) in %s", len(tasks), len(args.tasks_file), folder)

    import json as _json
    agent_models = None
    if args.agent_models:
        raw = _json.loads(args.agent_models)
        agent_models = {k: tuple(v) for k, v in raw.items()}
        if args.pipeline == "single":
            logger.warning("--agent-models is ignored when --pipeline=single")

    proc = None
    try:
        if args.provider == "vllm":
            from agenticeoecosystem.utils.registry import launch_vllm
            proc = launch_vllm(args.model, startup_timeout=args.vllm_startup_timeout)
            os.environ["VLLM_PORT"] = "8000"

        from agenticeoecosystem.evaluation.pipeline_runner import generate_solutions
        output_filename = f"results_{label_safe}.json"
        tasks, in_tok, out_tok = generate_solutions(
            tasks,
            model=args.model,
            temperature=args.temperature,
            model_provider=args.provider,
            output_filename=output_filename,
            skip_solved=args.skip_solved,
            pipeline=args.pipeline,
            agent_models=agent_models,
        )
        # Authoritative save to the requested output_dir
        tasks.save_to_file(output_filename, folder=str(output_dir))
        logger.info("Results saved → %s/%s  tokens: %d in / %d out",
                    output_dir, output_filename, in_tok, out_tok)
        return 0

    except Exception as exc:
        logger.error("Generation failed for '%s': %s", args.label, exc, exc_info=True)
        return 1

    finally:
        _terminate_vllm(proc)


if __name__ == "__main__":
    sys.exit(main())
