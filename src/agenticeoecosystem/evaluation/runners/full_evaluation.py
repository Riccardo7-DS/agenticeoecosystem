"""Checkpointed orchestration of the current task-specific evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from agenticeoecosystem.utils.output_paths import candidate_output_roots
from typing import Any

from agenticeoecosystem.definitions import ROOT_DIR
from ..evidence.candidate_evidence import CandidateEvidence, EvaluationRequirements
from ..dataclasses import Task, TaskSet
from ..judges.deterministic_judge import DeterministicAnswerJudge

LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "task_specific_v1"
NUMERIC_METHODS = {"unit_normalized_error", "numeric_tolerance_accuracy", "exact_count_match",
                   "coordinate_distance_error"}
INSPECTION_METHODS = {"agent_as_judge", "gis_verification_agent", "aaj_image_inspection",
                      "numerical_artifact_validation", "aligned_series_mae_rmse",
                      "clipped_extent_comparison"}
TERMINAL_PHASES = {"scored", "unverified", "diagnostic", "unsupported"}
TASK_FIELDS = ("evaluator_scores", "evaluator_scores_raw", "evaluator_reasoning",
               "evaluator_tool_calls", "evaluator_evidence", "evaluator_criteria")


def digest(value: Any) -> str:
    """Hash a canonical JSON value, including configuration and answer content."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace a checkpoint atomically, leaving its predecessor readable on failure."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class FullEvaluationConfig:
    """Explicit inputs, judge settings and output location for one batch."""

    results_files: tuple[Path, ...]
    output_dir: Path
    model: str
    provider: str = "eve"
    label: str = "full_evaluation_v1"
    strategies_file: Path = ROOT_DIR / "data/tasks/task_evaluation_strategies.json"
    ground_truths_file: Path = ROOT_DIR / "data/tasks/ground_truths.json"
    rubrics_file: Path = ROOT_DIR / "data/tasks/rubrics_108q_v4_evidence.json"
    benchmark_file: Path = ROOT_DIR / "data/tasks/benchmark_tasks.json"
    artifact_manifest: Path | None = None
    reference_validations: Path | None = None
    task_ids: tuple[str, ...] = ()
    temperature: float = 0.0
    seed: int = 42
    max_turns: int = 8
    gis_max_turns: int = 5
    resume: bool = True
    dry_run: bool = False


class FullEvaluationRunner:
    """Dispatch configured passes, preserve provenance, and expose coverage gaps."""

    def __init__(self, config: FullEvaluationConfig) -> None:
        self.config = config
        if config.provider not in {"eve", "openrouter", "openai", "custom", "vllm"}:
            raise ValueError("Unsupported judge provider")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.label):
            raise ValueError("Evaluator label must contain only letters, digits, dot, dash or underscore")
        if not config.results_files or len({p.stem for p in config.results_files}) != len(config.results_files):
            raise ValueError("Provide candidate files with distinct filenames")
        if config.artifact_manifest and len(config.results_files) != 1:
            raise ValueError("An artifact manifest is allowed for exactly one candidate file")
        if config.max_turns < 1 or config.gis_max_turns < 1:
            raise ValueError("Judge turn limits must be positive")
        self.strategies = self._index(self._read(config.strategies_file)["tasks"])
        self.ground_truths = self._read(config.ground_truths_file)
        self.benchmark = self._index(self._read(config.benchmark_file)["tasks"])
        self.rubrics = self._index(self._read(config.rubrics_file))
        modes = {entry["rubric"].get("evidence_mode") for entry in self.rubrics.values()}
        if len(modes) != 1 or not modes <= {"trace", "evidence"}:
            raise ValueError("Select one explicit v4 trace or evidence rubric profile")
        self.rubric_mode = next(iter(modes))
        for entry in self.rubrics.values():
            rubric = entry["rubric"]
            if not rubric.get("protocol_version") or not rubric.get("scoring_instructions") or len(rubric.get("criteria", [])) != 3:
                raise ValueError(f"Incomplete rubric protocol: {entry['task_ID']}")
        self.manifest = self._read(config.artifact_manifest) if config.artifact_manifest else {}
        if not isinstance(self.manifest, dict) or any(
                not isinstance(paths, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths)
                for paths in self.manifest.values()):
            raise ValueError("Artifact manifest must map task IDs to absolute path lists")
        self.validations = self._read(config.reference_validations) if config.reference_validations else {}
        if not isinstance(self.validations, dict):
            raise ValueError("Reference validations must be a task-ID-keyed object")
        self.requirements = EvaluationRequirements(config.strategies_file)
        self.answer_judge = DeterministicAnswerJudge(config.model, config.provider, config.temperature)
        self.artifact_state_hashes: dict[str, str] = {}
        self.protocol_label = f"{config.label}.protocol"
        self.labels = {phase: f"{config.label}.{phase}" for phase in ("aaj", "numeric", "rubric", "deterministic")}
        self.configuration_hash = digest({
            "protocol": PROTOCOL_VERSION, "model": config.model, "provider": config.provider,
            "temperature": config.temperature, "seed": config.seed,
            "max_turns": config.max_turns, "gis_max_turns": config.gis_max_turns,
            "strategies": self.strategies, "gt": self.ground_truths,
            "rubrics": self.rubrics, "benchmark": self.benchmark,
            "implementation": self._implementation_digests(),
        })

    @staticmethod
    def _implementation_digests() -> dict[str, str]:
        """Include every evaluation module in checkpoint invalidation."""
        package_root = Path(__file__).parent.parent
        return {
            str(path.relative_to(package_root)): CandidateEvidence.file_digest(path)
            for path in sorted(package_root.rglob("*.py"))
        }

    @staticmethod
    def _read(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {entry["task_ID"]: entry for entry in entries}
        if len(result) != len(entries):
            raise ValueError("Duplicate task IDs in configuration")
        return result

    def _artifact_state(self, task: Task) -> dict[str, str]:
        """Bind resume to actual file content and missing-file state, not path alone."""
        paths = [a.get("archived_path", a.get("path", "")) for a in task.candidate_artifacts]
        paths.extend(self.manifest.get(task.task_ID, []))
        # Include trace-bound local files for historical runs without archives.
        paths.extend(str(p) for p in CandidateEvidence.artifact_paths(task, candidate_output_roots()))
        result = {}
        for name in sorted(set(paths) - {""}):
            try:
                result[name] = CandidateEvidence.file_digest(Path(name))
            except OSError:
                result[name] = "unavailable"
        return result

    def _key(self, task: Task) -> str:
        artifact_state = self._artifact_state(task)
        self.artifact_state_hashes[task.task_ID] = digest(artifact_state)
        return digest({"configuration": self.configuration_hash,
                       "candidate": CandidateEvidence.fingerprint(task),
                       "artifacts": artifact_state,
                       "manifest": self.manifest.get(task.task_ID, []),
                       "reference_validation": self.validations.get(task.task_ID)})

    def _reference_verified(self, task: Task, gt: dict[str, Any], strategy: dict[str, Any]) -> bool:
        """Allow fixed contracts or explicit candidate/reference-bound validity records."""
        if (strategy.get("reference_policy") or "").startswith("fixed_"):
            return True
        validation = self.validations.get(task.task_ID, {})
        if (validation.get("status") == "verified" and validation.get("reason")
                and validation.get("candidate_fingerprint") == CandidateEvidence.fingerprint(task)
                and validation.get("ground_truth_sha256") == digest(gt)
                and (self.artifact_state_hashes.get(task.task_ID) == digest({})
                     or validation.get("artifact_state_sha256") == self.artifact_state_hashes.get(task.task_ID))):
            return True
        provenance = gt.get("provenance", {})
        if (strategy.get("reference_policy") or "").startswith("versioned_server_catalogue"):
            module = provenance.get("server_module")
            path = (ROOT_DIR / module).resolve() if module else None
            if path and path.is_relative_to(ROOT_DIR) and path.is_file():
                return CandidateEvidence.file_digest(path) == provenance.get("server_sha256")
        return False

    def plan(self, task: Task) -> dict[str, Any]:
        """Validate question versions and list only currently configured phases."""
        strategy = self.strategies.get(task.task_ID)
        if strategy is None:
            status = "out_of_scope" if "Scope Awareness" in task.task_labels else "missing_strategy"
            return {"status": status, "methods": [], "phases": []}
        methods = list(strategy.get("evaluation_methods", []))
        current = self.benchmark.get(task.task_ID)
        if current is None or current["task_text"] != task.task_text:
            return {"status": "question_version_mismatch", "methods": methods, "phases": []}
        rubric = self.rubrics.get(task.task_ID)
        if "rubrics" in methods and rubric and rubric.get("task_text", task.task_text) != task.task_text:
            return {"status": "rubric_question_mismatch", "methods": methods, "phases": []}
        if self.config.task_ids and task.task_ID not in self.config.task_ids:
            return {"status": "not_selected", "methods": methods, "phases": []}
        phases = []
        if set(methods) & INSPECTION_METHODS or ("rubrics" in methods and self.rubric_mode == "evidence"):
            phases.append("aaj")
        if set(methods) & NUMERIC_METHODS:
            phases.append("numeric")
        if set(methods) - INSPECTION_METHODS - NUMERIC_METHODS - {"rubrics", "llm_judge"}:
            phases.append("deterministic")
        if "rubrics" in methods:
            phases.append("rubric")
        available = INSPECTION_METHODS | NUMERIC_METHODS | {"rubrics"}
        if "rubrics" in methods:
            available |= {"llm_judge"}
        available |= self.answer_judge.supported(strategy, self.ground_truths.get(task.task_ID, {}))
        unsupported = set(methods) - available
        return {"status": "missing_answer" if task.generated_solution is None else "pending",
                "methods": methods, "phases": phases, "unsupported_methods": sorted(unsupported)}

    def _aaj(self, task: Task, folder: Path) -> dict[str, Any]:
        from ..judges.agent_judge import judge_task_agentic

        label = self.labels["aaj"]
        judge_task_agentic(task, model=self.config.model, provider=self.config.provider,
                          temperature=self.config.temperature, max_turns=self.config.max_turns,
                          gis_max_turns=self.config.gis_max_turns, evaluator_label=label,
                          transcript_dir=str(folder / "transcripts"),
                          artifact_paths=self.manifest.get(task.task_ID, []),
                          evaluation_requirements=self.requirements)
        score = task.evaluator_scores.get(label)
        if score is None:
            return {"status": "error", "reason": task.evaluator_reasoning.get(label) or "No AAJ verdict"}
        available = CandidateEvidence.context(task, label)["available"]
        return {"status": "scored" if available else "unverified", "score": score,
                "evidence_available": available,
                "reason": "AAJ verdict; verification coverage is recorded separately"}

    def _numeric(self, task: Task) -> dict[str, Any]:
        from ..judges.numerical_judge import numeric_score_task_solution

        strategy = self.strategies[task.task_ID]
        gt = self.ground_truths.get(task.task_ID, {})
        result = numeric_score_task_solution(
            task, self.strategies, self.ground_truths, model=self.config.model,
            provider=self.config.provider, temperature=self.config.temperature,
            evaluator_label=self.labels["numeric"], inspection_label=self.labels["aaj"])
        if result is None:
            return {"status": "unverified", "reason": "No usable GT, selected numeric fields, or verified runtime reference"}
        evidence = task.evaluator_evidence[self.labels["numeric"]]
        comparable = bool(evidence.get("runtime_reference")) or self._reference_verified(task, gt, strategy)
        evidence["reference_validated"] = comparable
        record = {"status": "scored" if comparable else "diagnostic",
                "hit_at_tol": evidence["hit_at_tol"], "num_score": evidence["num_score"],
                "fields": task.evaluator_criteria[self.labels["numeric"]],
                "reference_validated": comparable,
                "reason": "Comparable reference" if comparable else
                          "Reference source/support validity is unverified; excluded from validated averages"}
        if "coordinate_distance_error" in strategy.get("evaluation_methods", []):
            fields = {field["field"]: field for field in record["fields"]}
            lat, lon = fields.get("lat", {}), fields.get("lon", {})
            if all(field.get("numeric") and not field.get("invalid") and not field.get("missing")
                   for field in (lat, lon)) and all(abs(lat[key]) <= 90 and abs(lon[key]) <= 180 for key in ("y", "y_hat")):
                from pyproj import Geod
                _, _, distance = Geod(ellps="WGS84").inv(lon["y"], lat["y"], lon["y_hat"], lat["y_hat"])
                if math.isfinite(distance):
                    record["coordinate_distance_error_m"] = abs(distance)
        return record

    def _deterministic(self, task: Task) -> dict[str, Any]:
        strategy = self.strategies[task.task_ID]
        gt = self.ground_truths.get(task.task_ID, {})
        supported = self.answer_judge.supported(strategy, gt) & set(strategy["evaluation_methods"])
        if not supported:
            return {"status": "unsupported", "reason": "No executable deterministic answer contract", "metrics": {}}
        values = self.answer_judge.extract(task)
        metrics = self.answer_judge.score(values, strategy, gt)
        metrics = {method: metrics.get(method) for method in sorted(supported)}
        comparable = self._reference_verified(task, gt, strategy)
        record = {"status": "scored" if comparable else "diagnostic", "metrics": metrics,
                  "reference_validated": comparable, "extracted_values": values,
                  "reason": "Explicit deterministic contract" if comparable else "Source snapshot validity unverified"}
        task.evaluator_evidence[self.labels["deterministic"]] = record
        return record

    def _rubric(self, task: Task) -> dict[str, Any]:
        from ..judges.rubric_evaluator import rubric_score_task_solution

        entry = self.rubrics.get(task.task_ID)
        if entry is None:
            return {"status": "unsupported", "reason": "No rubric in selected profile"}
        rubric = entry["rubric"]
        protocol = {key: rubric[key] for key in ("protocol_version", "evidence_mode", "scoring_instructions")}
        label = self.labels["rubric"]
        rubric_score_task_solution(task, rubric["criteria"], model=self.config.model,
                                   provider=self.config.provider, temperature=self.config.temperature,
                                   evaluator_label=label, rubric_protocol=protocol,
                                   inspection_label=self.labels["aaj"] if self.rubric_mode == "evidence" else None)
        criteria = task.evaluator_criteria.get(label)
        if criteria is None:
            return {"status": "error", "reason": "Rubric judge returned no parseable criteria"}
        total = task.evaluator_scores_raw.get(label)
        return {"status": "scored" if total is not None else "unverified", "weighted_total": total,
                "criteria": criteria, "profile": self.rubric_mode}

    def _methods(self, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        """Expose each assigned method without equating consultation with validation."""
        phases = record.get("phase_results", {})
        tools = task.evaluator_tool_calls.get(self.labels["aaj"], [])
        gis_calls = [call for tool in tools for call in tool.get("evidence", {}).get("tool_calls", [])
                     if isinstance(call.get("result"), dict) and "error" not in call["result"]]
        methods = {}
        for method in record["methods"]:
            if self.config.dry_run and record["status"] == "pending" and method in record.get("unsupported_methods", []):
                methods[method] = {"status": "unsupported", "reason": "No executable contract"}
            elif self.config.dry_run and record["status"] == "pending":
                methods[method] = {"status": "planned"}
            elif record["status"] not in {"evaluated", "pending"}:
                methods[method] = {"status": record["status"]}
            elif method in {"rubrics", "llm_judge"} and "rubrics" in record["methods"]:
                methods[method] = phases.get("rubric", {"status": "unsupported"})
            elif method == "agent_as_judge":
                methods[method] = phases.get("aaj", {"status": "unsupported"})
            elif method in NUMERIC_METHODS:
                methods[method] = phases.get("numeric", {"status": "unsupported"})
                if method == "coordinate_distance_error":
                    numeric = phases.get("numeric", {})
                    distance = numeric.get("coordinate_distance_error_m")
                    methods[method] = {"status": numeric.get("status") if distance is not None else "unverified",
                                       "value": distance, "units": "m", "ellipsoid": "WGS84"}
                elif method == "exact_count_match" and phases.get("numeric", {}).get("fields"):
                    numeric = phases["numeric"]
                    fields = numeric["fields"]
                    methods[method] = {"status": numeric["status"],
                                       "accuracy": sum(field.get("absolute_error") == 0 for field in fields) / len(fields)}
            elif method == "gis_verification_agent":
                methods[method] = {"status": "inspected" if gis_calls else "unverified", "measurements": gis_calls}
            elif method == "aaj_image_inspection":
                preview = any(t.get("observation", {}).get("visual_preview_used_by_aaj") for t in tools)
                methods[method] = {"status": "inspected" if preview else "unverified"}
            elif method in {"numerical_artifact_validation", "aligned_series_mae_rmse", "clipped_extent_comparison"}:
                name = "inspect_extent" if method == "clipped_extent_comparison" else "compare_netcdf"
                measurements = [call["result"] for call in gis_calls if call.get("tool") == name]
                validated = [measurement for measurement in measurements if isinstance(measurement.get("verified"), bool)]
                methods[method] = {"status": "scored" if validated else "unverified", "measurements": measurements}
            elif method in phases.get("deterministic", {}).get("metrics", {}):
                phase = phases["deterministic"]
                value = phase["metrics"][method]
                methods[method] = {"status": phase["status"] if value is not None else "unverified", "value": value}
            else:
                methods[method] = {"status": "unsupported", "reason": "Mapped validator has no executable contract; no legacy fallback"}
        return methods

    def _load(self, path: Path) -> tuple[TaskSet, str]:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        tasks = TaskSet(metadata=payload.get("metadata", {}), tasks=payload["tasks"])
        if len({task.task_ID for task in tasks}) != len(tasks):
            raise ValueError(f"Duplicate candidate task IDs: {path}")
        missing = set(self.config.task_ids) - {task.task_ID for task in tasks}
        if missing:
            raise ValueError(f"Selected task IDs absent from {path}: {sorted(missing)}")
        for task in tasks:
            for field in TASK_FIELDS:
                getattr(task, field).clear()
            task.match_score_LLM = None
            task.match_reasoning_LLM = None
        return tasks, hashlib.sha256(text.encode()).hexdigest()

    def _evaluate_candidate(self, source: Path) -> dict[str, Any]:
        tasks, source_hash = self._load(source)
        output = self.config.output_dir / f"{source.stem}_{self.config.label}.json"
        if output.resolve() == source.resolve():
            raise ValueError("Evaluation output must not overwrite its candidate input")
        folder = self.config.output_dir / f"{source.stem}_{self.config.label}"
        folder.mkdir(exist_ok=True)
        previous = {}
        if self.config.resume and output.exists() and not self.config.dry_run:
            saved = self._read(output)
            previous = {t["task_ID"]: Task.model_validate(t) for t in saved["tasks"]}
        metadata = {"protocol_version": PROTOCOL_VERSION, "configuration_sha256": self.configuration_hash,
                    "source": str(source.resolve()), "source_sha256": source_hash,
                    "model": self.config.model, "provider": self.config.provider,
                    "rubric_profile": self.rubric_mode, "seed": self.config.seed,
                    "temperature": self.config.temperature, "started_at": datetime.now(timezone.utc).isoformat(),
                    "legacy_scores_excluded": True, "dry_run": self.config.dry_run,
                    "configuration": {key: str(value) for key, value in vars(self.config).items()}}
        try:
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            metadata["git_commit"] = "unavailable"
        tasks.metadata["unified_evaluation"] = metadata
        for task in tasks:
            record = self.plan(task)
            record.update(cache_key=None if self.config.dry_run else self._key(task), phase_results={})
            record["artifact_state_sha256"] = self.artifact_state_hashes.get(task.task_ID)
            older = previous.get(task.task_ID)
            if older and older.evaluator_evidence.get(self.protocol_label, {}).get("cache_key") == record["cache_key"]:
                for field in TASK_FIELDS:
                    target = getattr(task, field)
                    target.update({label: value for label, value in getattr(older, field).items()
                                   if label in self.labels.values()})
                record["phase_results"] = older.evaluator_evidence[self.protocol_label].get("phase_results", {})
            if record["status"] == "pending" and not self.config.dry_run:
                for phase in record["phases"]:
                    result = record["phase_results"].get(phase, {})
                    if result.get("status") in TERMINAL_PHASES:
                        continue
                    # A new inspection invalidates scores that used the previous inspection.
                    if phase == "aaj":
                        for dependent in ("numeric", "rubric"):
                            record["phase_results"].pop(dependent, None)
                            for field in TASK_FIELDS:
                                getattr(task, field).pop(self.labels[dependent], None)
                    for field in TASK_FIELDS:
                        getattr(task, field).pop(self.labels[phase], None)
                    LOGGER.info("%s %s: %s", source.name, task.task_ID, phase)
                    try:
                        if phase == "aaj":
                            result = self._aaj(task, folder)
                        else:
                            result = getattr(self, f"_{phase}")(task)
                    except Exception as error:
                        LOGGER.exception("Evaluation phase failed: %s %s", task.task_ID, phase)
                        result = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
                    record["phase_results"][phase] = result
                    task.evaluator_evidence[self.protocol_label] = record
                    write_json(output, tasks.model_dump(mode="json"))
                record["status"] = "evaluated"
            record["method_results"] = self._methods(task, record)
            task.evaluator_evidence[self.protocol_label] = record
            if not self.config.dry_run:
                write_json(output, tasks.model_dump(mode="json"))
        summary = self._summary(tasks)
        summary.update(source=str(source), source_sha256=source_hash, output=str(output), metadata=metadata)
        write_json(folder / ("plan.json" if self.config.dry_run else "summary.json"), summary)
        return summary

    def _summary(self, tasks: TaskSet) -> dict[str, Any]:
        records = [task.evaluator_evidence[self.protocol_label] for task in tasks]
        methods = sorted({method for record in records for method in record["methods"]})
        coverage = {method: dict(Counter(record["method_results"][method]["status"]
                                        for record in records if method in record["method_results"]))
                    for method in methods}
        means = {}
        for phase, metric in (("numeric", "hit_at_tol"), ("numeric", "num_score"),
                              ("rubric", "weighted_total"), ("aaj", "score")):
            values = [record["phase_results"].get(phase, {}).get(metric) for record in records
                      if record["phase_results"].get(phase, {}).get("status") == "scored"]
            values = [value for value in values if value is not None]
            means[metric] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
        method_means = {}
        for method in methods:
            results = [record["method_results"].get(method, {}) for record in records]
            values = [result.get("value", result.get("accuracy")) for result in results
                      if result.get("status") == "scored"]
            values = [value for value in values if isinstance(value, (int, float))]
            if values:
                method_means[method] = {"n": len(values), "mean": sum(values) / len(values)}
        issues = {"error", "unsupported", "unverified", "diagnostic", "question_version_mismatch",
                  "rubric_question_mismatch", "missing_strategy", "missing_answer"}
        return {"questions": len(tasks), "task_statuses": dict(Counter(r["status"] for r in records)),
                "method_coverage": coverage, "validated_means": means, "validated_method_means": method_means,
                "incomplete": (any(status in issues for counts in coverage.values() for status in counts)
                               or any(record["status"] in issues for record in records)),
                "tasks": [{"task_ID": task.task_ID, **record} for task, record in zip(tasks, records)]}

    def run(self) -> dict[str, Any]:
        """Evaluate each frozen input snapshot and save a batch coverage report."""
        import fcntl
        import numpy as np

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.config.output_dir / ".evaluation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another evaluation is writing this output directory") from error
            summaries = [self._evaluate_candidate(path) for path in self.config.results_files]
            report = {"protocol_version": PROTOCOL_VERSION, "dry_run": self.config.dry_run,
                      "incomplete": any(summary["incomplete"] for summary in summaries),
                      "candidates": summaries}
            write_json(self.config.output_dir / ("plan.json" if self.config.dry_run else "summary.json"), report)
        return report
