"""Prepare candidate rerun task snapshots and run Mistral pipelines in order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agenticeoecosystem.evaluation.dataclasses import Task, TaskSet


@dataclass(frozen=True)
class RerunConfig:
    """Historical evaluation inputs and isolated rerun directory."""

    source: Path
    output: Path
    task_timeout: float = 1800
    include_changed: bool = False


class MissingCandidateReruns:
    """Select missing candidates and optionally revised questions, preserving provenance."""

    MISTRAL_ORDER = (
        ("Mistral-Small-2506", "multi"),
        ("Mistral-Small-3.2-24B-single", "single"),
    )
    OPENROUTER_ORDER = ("GLM-5.3-Flash-OR", "Nemotron-3-Super-OR")

    def __init__(self, config: RerunConfig) -> None:
        self.config = config

    def load_prepared(self, directory: Path) -> dict:
        """Load portable task inputs without requiring historical result files."""
        directory = directory.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        for candidate, entry in manifest["candidates"].items():
            source = (directory / entry["input"]).resolve()
            if source.parent != directory:
                raise ValueError(f"Prepared input must be inside its task directory: {candidate}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry["input_sha256"]:
                raise ValueError(f"Prepared input hash mismatch: {candidate}")
            tasks = TaskSet.model_validate_json(source.read_text()).tasks
            if [task.task_ID for task in tasks] != entry["task_ids"]:
                raise ValueError(f"Prepared input task IDs differ: {candidate}")
            if any(task.generated_solution is not None or task.evaluator_scores or task.candidate_artifacts for task in tasks):
                raise ValueError(f"Prepared inputs contain candidate results: {candidate}")
            entry["input"] = str(source)
        self.config.output.mkdir(parents=True, exist_ok=True)
        return manifest

    def prepare(self) -> dict:
        """Write validated, clean task inputs and a provenance manifest."""
        current = json.loads((ROOT / "data/tasks/benchmark_tasks.json").read_text())
        current_tasks = {task["task_ID"]: task for task in current["tasks"]}
        rerun_source = ROOT / "data/tasks/benchmark_rerun_required.md"
        required_ids = set(re.findall(r"^\| (TASK_\w+) \|", rerun_source.read_text(), re.MULTILINE)) if self.config.include_changed else set()
        unknown_ids = required_ids - current_tasks.keys()
        if unknown_ids:
            raise ValueError(f"Rerun-required IDs absent from current benchmark: {sorted(unknown_ids)}")
        selection = "missing_or_rerun_required" if self.config.include_changed else "missing_only"
        versions = "current for rerun-required tasks; historical for other missing tasks" if self.config.include_changed else "historical"
        manifest = {"selection": selection,
                    "task_versions": versions,
                    "rerun_required_source": str(rerun_source) if self.config.include_changed else None,
                    "rerun_required_ids": sorted(required_ids),
                    "current_benchmark_sha256": hashlib.sha256((ROOT / "data/tasks/benchmark_tasks.json").read_bytes()).hexdigest(),
                    "source": str(self.config.source), "candidates": {}}
        inputs = self.config.output / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        for source in sorted(self.config.source.glob("results_*_Nemotron-2-12B_reference_all.json")):
            candidate = source.name.removeprefix("results_").removesuffix("_Nemotron-2-12B_reference_all.json")
            historical = json.loads(source.read_text())
            tasks = []
            changed = []
            missing_ids = []
            selected_required = []
            for entry in historical["tasks"]:
                missing = entry["generated_solution"] is None
                required = entry["task_ID"] in required_ids
                if not (missing or required) or "Scope Awareness" in entry.get("task_labels", []):
                    continue
                if missing:
                    missing_ids.append(entry["task_ID"])
                if required:
                    selected_required.append(entry["task_ID"])
                selected_entry = current_tasks[entry["task_ID"]] if required else entry
                # Reconstruct only task/reference fields, clearing all candidate and judge state.
                task = Task.model_validate({key: selected_entry[key] for key in (
                    "task_ID", "task_text", "task_labels", "reference_solution_description",
                    "reference_solutions", "GT") if key in selected_entry})
                tasks.append(task)
                latest = current_tasks.get(task.task_ID)
                latest_task = Task.model_validate(latest).model_dump() if latest is not None else None
                historical_task = task.model_dump()
                if latest_task is None or any(latest_task.get(key) != historical_task.get(key) for key in (
                    "task_text", "reference_solution_description", "reference_solutions")):
                    changed.append(task.task_ID)
            target = inputs / f"tasks_{candidate}.json"
            metadata = {"rerun_source_candidate": candidate, "selection": selection,
                        "historical_model": historical["metadata"]["model"],
                        "task_versions": versions, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            snapshot = TaskSet(metadata=metadata, tasks=tasks)
            contents = snapshot.model_dump_json(indent=2)
            if target.exists() and target.read_text() != contents:
                raise ValueError(f"Prepared inputs differ; choose a fresh output directory: {target}")
            target.write_text(contents)
            manifest["candidates"][candidate] = {
                "count": len(tasks), "input": str(target), "model": historical["metadata"]["model"],
                "task_ids": [task.task_ID for task in tasks],
                "missing_task_ids": missing_ids,
                "rerun_required_task_ids": selected_required,
                "overlap_count": len(set(missing_ids) & set(selected_required)),
                "differs_from_current_task_version": changed,
                "input_sha256": hashlib.sha256(contents.encode()).hexdigest(),
            }
        if not manifest["candidates"]:
            raise ValueError("No historical candidate files found")
        (self.config.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def run_mistral(self, manifest: dict, model: str | None,
                    pipelines: list[str] | None = None) -> None:
        """Execute multi then single, resuming from their isolated checkpoints."""
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from agenticeoecosystem.config.credentials import get_openai_client
        client = get_openai_client("eve").with_options(timeout=20, max_retries=0)
        available = {item.id for item in client.models.list().data}
        for candidate, pipeline in self.MISTRAL_ORDER:
            if pipelines is not None and pipeline not in pipelines:
                continue
            entry = manifest["candidates"][candidate]
            selected_model = model or entry["model"]
            if selected_model not in available:
                raise ValueError(f"Required model {selected_model!r} is not served; available: {sorted(available)}")
            self.run_candidate(manifest, candidate, pipeline, "eve", selected_model, self.config.output / pipeline)

    def run_openrouter(self, manifest: dict) -> None:
        """Run benchmark OpenRouter candidates serially, excluding Maverick."""
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is not configured")
        for candidate in self.OPENROUTER_ORDER:
            self.run_candidate(manifest, candidate, "multi", "openrouter",
                               manifest["candidates"][candidate]["model"],
                               self.config.output / "openrouter" / candidate)

    def run_candidate(self, manifest: dict, candidate: str, pipeline: str,
                      provider: str, selected_model: str, output: Path) -> None:
        """Resume one candidate and record completion independently of provider."""
        entry = manifest["candidates"][candidate]
        output.mkdir(parents=True, exist_ok=True)
        self._run_candidate(entry, candidate, pipeline, provider, selected_model, output)

    def _run_candidate(self, entry: dict, candidate: str, pipeline: str,
                       provider: str, selected_model: str, output: Path) -> None:
        """Execute a checkpointed candidate in a fresh process."""
        label = f"{candidate}_missing_rerun"
        checkpoint = output / f"results_{label}.json"
        task_file = Path(entry["input"])
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["metadata"].get("model") != selected_model:
                raise ValueError(f"Checkpoint belongs to a different model: {checkpoint}")
            if [task["task_ID"] for task in saved["tasks"]] != entry["task_ids"]:
                raise ValueError(f"Checkpoint task IDs differ: {checkpoint}")
            task_file = checkpoint
        env = dict(os.environ)
        env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
        command = [sys.executable, str(ROOT / "scripts/generation/run_generate.py"),
                   "--label", label, "--model", selected_model, "--provider", provider,
                   "--pipeline", pipeline, "--tasks-folder", str(task_file.parent),
                   "--tasks-file", task_file.name, "--output-dir", str(output),
                   "--task-timeout", str(self.config.task_timeout),
                   "--archive-evidence", "--skip-solved"]
        print(f"Starting {candidate} ({pipeline}, {provider}): {entry['count']} tasks; model={selected_model}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        result = json.loads(checkpoint.read_text())
        missing = [task["task_ID"] for task in result["tasks"] if task["generated_solution"] is None]
        (output / "completion.json").write_text(json.dumps({
            "model": selected_model, "tasks": len(result["tasks"]),
            "saved_candidates": len(result["tasks"]) - len(missing),
            "still_missing": missing,
        }, indent=2) + "\n")
        print(f"Finished {pipeline}: {len(missing)} still missing", flush=True)


def main() -> None:
    """Prepare snapshots; execute only when explicitly selected on the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/evaluation_runs/all_non_scope_reference_20261002")
    parser.add_argument("--prepared-inputs", type=Path,
                        help="Portable task-input directory containing manifest.json; no historical results needed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-timeout", type=float, default=1800)
    parser.add_argument("--execute-mistral", action="store_true")
    parser.add_argument("--mistral-pipelines", nargs="+", choices=["multi", "single"],
                        help="Restrict Mistral phases; defaults to multi then single")
    parser.add_argument("--execute-openrouter", action="store_true", help="Run GLM Flash then Nemotron Super, after Mistral if also selected")
    parser.add_argument("--model", help="Explicit served model for both pipelines; otherwise require original models")
    parser.add_argument("--include-changed", action="store_true", help="Also select rerun-required IDs using their current task versions")
    args = parser.parse_args()
    if args.task_timeout <= 0:
        parser.error("--task-timeout must be positive")
    runner = MissingCandidateReruns(RerunConfig(args.source.resolve(), args.output_dir.resolve(), args.task_timeout, args.include_changed))
    manifest = runner.load_prepared(args.prepared_inputs) if args.prepared_inputs else runner.prepare()
    for candidate, entry in manifest["candidates"].items():
        print(f"{candidate}: {entry['count']} selected tasks ({len(entry['missing_task_ids'])} missing, {len(entry['rerun_required_task_ids'])} rerun-required, {entry['overlap_count']} overlap)")
    if args.execute_mistral:
        runner.run_mistral(manifest, args.model, args.mistral_pipelines)
    if args.execute_openrouter:
        runner.run_openrouter(manifest)


if __name__ == "__main__":
    main()
