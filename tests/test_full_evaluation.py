"""Offline orchestration checks: dispatch, isolation, resume and metric contracts."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Load only evaluation modules, without candidate-agent package initialization.
PACKAGE = ModuleType("full_evaluation_test")
PACKAGE.__path__ = [str(ROOT / "src/agenticeoecosystem/evaluation")]
sys.modules[PACKAGE.__name__] = PACKAGE
runner_module = importlib.import_module(f"{PACKAGE.__name__}.runners.full_evaluation")
schema = importlib.import_module(f"{PACKAGE.__name__}.dataclasses")
deterministic = importlib.import_module(f"{PACKAGE.__name__}.judges.deterministic_judge")


class FullEvaluationTests(unittest.TestCase):
    """Exercise real checkpoint orchestration with mocked phase model calls."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        task = {"task_ID": "q1", "task_text": "Report the measured area.",
                "generated_solution": {"steps": [], "final_answer": "Area is 2 km²."},
                "evaluator_scores": {"old_reference": 2}}
        self.source = self.save("candidate.json", {"metadata": {}, "tasks": [task]})
        self.strategies = self.save("strategies.json", {"tasks": [{"task_ID": "q1",
            "evaluation_methods": ["agent_as_judge", "gis_verification_agent", "rubrics",
                                   "llm_judge", "unit_normalized_error", "numeric_tolerance_accuracy", "set_f1"]}]})
        self.gt = self.save("gt.json", {"q1": {"area_km2": 2}})
        self.benchmark = self.save("benchmark.json", {"tasks": [task]})
        criteria = [{"type": kind, "weight": weight, "description": kind}
                    for kind, weight in zip(("scientific_correctness", "grounded_and_complete", "internal_consistency"), (.5, .3, .2))]
        self.rubrics = self.save("rubrics.json", [{"task_ID": "q1", "task_text": task["task_text"],
            "rubric": {"criteria": criteria, "protocol_version": "4.0", "evidence_mode": "evidence",
                       "scoring_instructions": "Judge the requested result."}}])
        self.config = runner_module.FullEvaluationConfig(
            results_files=(self.source,), output_dir=self.root / "judged", model="offline",
            strategies_file=self.strategies, ground_truths_file=self.gt,
            rubrics_file=self.rubrics, benchmark_file=self.benchmark)
        self.calls = []

    def tearDown(self) -> None:
        self.directory.cleanup()

    def save(self, filename: str, value: object) -> Path:
        path = self.root / filename
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def phases(self) -> dict[str, object]:
        def phase(name: str) -> object:
            def evaluate(task: object, *args: object) -> dict:
                self.calls.append(name)
                return {"status": "scored", "score": 2, "hit_at_tol": 1.0,
                        "num_score": 1.0, "weighted_total": 1.8, "metrics": {"set_f1": 1.0}}
            return evaluate
        return {f"_{name}": phase(name) for name in ("aaj", "numeric", "deterministic", "rubric")}

    def run_mocked(self, config: object = None) -> dict:
        runner = runner_module.FullEvaluationRunner(config or self.config)
        with patch.multiple(runner, **self.phases()):
            return runner.run()

    def test_dispatch_order_and_input_isolation(self) -> None:
        original = self.source.read_bytes()
        report = self.run_mocked()
        self.assertEqual(self.calls, ["aaj", "numeric", "deterministic", "rubric"])
        output = json.loads(Path(report["candidates"][0]["output"]).read_text())
        self.assertNotIn("old_reference", output["tasks"][0]["evaluator_scores"])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(report["candidates"][0]["validated_means"]["num_score"]["mean"], 1.0)

    def test_resume_and_changed_answer(self) -> None:
        self.run_mocked()
        self.calls.clear()
        self.run_mocked()
        self.assertEqual(self.calls, [])
        payload = json.loads(self.source.read_text())
        payload["tasks"][0]["generated_solution"]["final_answer"] = "Area is 3 km²."
        self.source.write_text(json.dumps(payload))
        self.run_mocked()
        self.assertEqual(self.calls, ["aaj", "numeric", "deterministic", "rubric"])

    def test_changed_ground_truth_invalidates_resume(self) -> None:
        self.run_mocked()
        self.calls.clear()
        self.gt.write_text(json.dumps({"q1": {"area_km2": 3}}))
        self.run_mocked()
        self.assertEqual(len(self.calls), 4)

    def test_changed_manifest_artifact_invalidates_resume(self) -> None:
        artifact = self.root / "artifact.txt"
        artifact.write_text("original")
        manifest = self.save("manifest.json", {"q1": [str(artifact)]})
        config = replace(self.config, artifact_manifest=manifest)
        self.run_mocked(config)
        self.calls.clear()
        artifact.write_text("changed")
        self.run_mocked(config)
        self.assertEqual(len(self.calls), 4)

    def test_failed_aaj_retries_and_invalidates_dependent_scores(self) -> None:
        runner = runner_module.FullEvaluationRunner(self.config)
        phases = self.phases()
        phases["_aaj"] = lambda *args: {"status": "error", "reason": "timeout"}
        with patch.multiple(runner, **phases):
            runner.run()
        self.calls.clear()
        self.run_mocked()
        self.assertEqual(self.calls, ["aaj", "numeric", "rubric"])

    def test_old_question_and_missing_answer_are_explicit(self) -> None:
        payload = json.loads(self.source.read_text())
        payload["tasks"][0]["task_text"] = "An earlier question."
        self.source.write_text(json.dumps(payload))
        report = self.run_mocked()
        self.assertEqual(self.calls, [])
        self.assertTrue(report["incomplete"])
        self.assertEqual(report["candidates"][0]["task_statuses"], {"question_version_mismatch": 1})

    def test_missing_strategy_is_incomplete_without_silent_success(self) -> None:
        self.strategies.write_text(json.dumps({"tasks": []}))
        report = self.run_mocked()
        self.assertTrue(report["incomplete"])
        self.assertEqual(self.calls, [])

    def test_missing_answer_has_no_fabricated_scores(self) -> None:
        payload = json.loads(self.source.read_text())
        payload["tasks"][0]["generated_solution"] = None
        self.source.write_text(json.dumps(payload))
        report = self.run_mocked()
        self.assertEqual(self.calls, [])
        self.assertEqual(report["candidates"][0]["task_statuses"], {"missing_answer": 1})
        self.assertTrue(report["incomplete"])

    def test_no_resume_forces_all_phases(self) -> None:
        self.run_mocked()
        self.calls.clear()
        self.run_mocked(replace(self.config, resume=False))
        self.assertEqual(len(self.calls), 4)

    def test_dry_run_never_calls_phases_or_hashes_artifacts(self) -> None:
        runner = runner_module.FullEvaluationRunner(replace(self.config, dry_run=True))
        with patch.multiple(runner, **self.phases()), patch.object(runner, "_artifact_state", side_effect=AssertionError):
            report = runner.run()
        self.assertEqual(self.calls, [])
        self.assertFalse(list(self.config.output_dir.glob("candidate_full*.json")))
        self.assertEqual(report["candidates"][0]["method_coverage"]["agent_as_judge"], {"planned": 1})

    def test_partial_numerical_scores_excluded_from_validated_average(self) -> None:
        runner = runner_module.FullEvaluationRunner(self.config)
        phases = self.phases()
        phases["_numeric"] = lambda *args: {"status": "diagnostic", "num_score": .8, "hit_at_tol": .5}
        with patch.multiple(runner, **phases):
            report = runner.run()
        self.assertIsNone(report["candidates"][0]["validated_means"]["num_score"]["mean"])
        self.assertTrue(report["incomplete"])

    def test_gis_consultation_without_measurement_is_unverified(self) -> None:
        report = self.run_mocked()
        self.assertEqual(report["candidates"][0]["method_coverage"]["gis_verification_agent"], {"unverified": 1})

    def test_real_numerical_metric_and_reference_gate(self) -> None:
        self.strategies.write_text(json.dumps({"tasks": [{"task_ID": "q1",
            "evaluation_methods": ["unit_normalized_error", "numeric_tolerance_accuracy"],
            "numerical_field_tolerances": {"area_km2": {"abs_tol": .1, "rel_tol": 0, "floor_scale": 0}}}]}))
        numeric = importlib.import_module(f"{PACKAGE.__name__}.judges.numerical_judge")
        with patch.object(numeric, "extract_candidate_values", return_value={"area_km2": 2.2}):
            report = runner_module.FullEvaluationRunner(self.config).run()
        result = report["candidates"][0]["tasks"][0]["phase_results"]["numeric"]
        self.assertEqual(result["status"], "diagnostic")
        self.assertEqual(result["hit_at_tol"], 0)
        self.assertAlmostEqual(result["num_score"], .5)
        payload = json.loads(self.source.read_text())
        task = schema.Task.model_validate(payload["tasks"][0])
        validations = self.save("validations.json", {"q1": {
            "status": "verified", "reason": "Independent fixed geometry/support check",
            "candidate_fingerprint": runner_module.CandidateEvidence.fingerprint(task),
            "ground_truth_sha256": runner_module.digest({"area_km2": 2})}})
        with patch.object(numeric, "extract_candidate_values", return_value={"area_km2": 2.2}):
            report = runner_module.FullEvaluationRunner(replace(self.config, reference_validations=validations)).run()
        self.assertAlmostEqual(report["candidates"][0]["validated_means"]["num_score"]["mean"], .5)

    def test_real_runtime_distance_unavailable_remains_unverified(self) -> None:
        self.strategies.write_text(json.dumps({"tasks": [{"task_ID": "q1",
            "evaluation_strategy": "numerical_answer",
            "evaluation_methods": ["unit_normalized_error", "numeric_tolerance_accuracy"]}]}))
        self.gt.write_text(json.dumps({"q1": {"reference_type": "source_dependent_runtime"}}))
        report = runner_module.FullEvaluationRunner(self.config).run()
        self.assertEqual(report["candidates"][0]["method_coverage"]["numeric_tolerance_accuracy"], {"unverified": 1})

    def test_geodetic_position_error_is_distinct_from_coordinate_scalar_score(self) -> None:
        self.strategies.write_text(json.dumps({"tasks": [{"task_ID": "q1",
            "evaluation_methods": ["coordinate_distance_error", "numeric_tolerance_accuracy"],
            "numerical_field_tolerances": {key: {"abs_tol": .02, "rel_tol": 0, "floor_scale": 0}
                                           for key in ("lat", "lon")}}]}))
        self.gt.write_text(json.dumps({"q1": {"lat": 0.0, "lon": 0.0}}))
        numeric = importlib.import_module(f"{PACKAGE.__name__}.judges.numerical_judge")
        with patch.object(numeric, "extract_candidate_values", return_value={"lat": 0.0, "lon": .01}):
            report = runner_module.FullEvaluationRunner(self.config).run()
        method = report["candidates"][0]["tasks"][0]["method_results"]["coordinate_distance_error"]
        self.assertEqual(method["status"], "diagnostic")
        self.assertAlmostEqual(method["value"], 1113.1949, places=3)

    def test_real_deterministic_extraction_and_scoring(self) -> None:
        self.strategies.write_text(json.dumps({"tasks": [{"task_ID": "q1",
            "reference_policy": "fixed_identifier", "evaluation_methods": ["canonicalized_crs_equivalence"],
            "deterministic_checks": [{"type": "canonicalized_crs_equivalence",
                "expected": {"authority": "EPSG", "code": 32722}}]}]}))
        values = {"items": [], "pairs": {}, "crs": "EPSG:32722"}
        with patch.object(deterministic.DeterministicAnswerJudge, "extract", return_value=values):
            report = runner_module.FullEvaluationRunner(self.config).run()
        method = report["candidates"][0]["tasks"][0]["method_results"]["canonicalized_crs_equivalence"]
        self.assertEqual(method, {"status": "scored", "value": 1.0})

    def test_set_normalization_and_duplicate_items(self) -> None:
        gt = {"reference_set": [{"id": "ET-01", "name": "Open Science"}, {"id": "ET-02", "name": "Education"}]}
        values = {"items": ["open science", "ET-01", "EDUCATION"], "pairs": {}}
        scores = deterministic.DeterministicAnswerJudge.score(values, {}, gt)
        self.assertEqual(scores["set_f1"], 1.0)
        self.assertEqual(scores["exact_set_match"], 1.0)

    def test_swapped_pair_labels_and_ambiguous_paraphrases(self) -> None:
        gt = {"reference_dictionary": {"a": "projections", "b": "forecasts"}}
        strategy = {"evaluation_methods": ["dataset_classification_accuracy"]}
        values = {"items": ["a", "b"], "pairs": {"a": "forecasts", "b": "projections"}}
        scores = deterministic.DeterministicAnswerJudge.score(values, strategy, gt)
        self.assertEqual(scores["exact_set_match"], 1.0)
        self.assertEqual(scores["dataset_classification_accuracy"], 0.0)
        values["pairs"]["a"] = "unrecognized description"
        self.assertIsNone(deterministic.DeterministicAnswerJudge.score(values, strategy, gt)["dataset_classification_accuracy"])

    def test_crs_equivalence_and_separate_hemisphere(self) -> None:
        strategy = {"deterministic_checks": [
            {"type": "canonicalized_crs_equivalence", "expected": {"authority": "EPSG", "code": 32722}},
            {"type": "utm_zone_hemisphere_match", "expected": {"zone": 22, "hemisphere": "S"}}]}
        values = {"items": [], "pairs": {}, "crs": "EPSG:32722", "utm_zone": 22, "utm_hemisphere": "north"}
        scores = deterministic.DeterministicAnswerJudge.score(values, strategy, {})
        self.assertEqual(scores["canonicalized_crs_equivalence"], 1.0)
        self.assertEqual(scores["utm_zone_hemisphere_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
