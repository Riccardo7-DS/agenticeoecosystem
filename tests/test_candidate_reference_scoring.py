"""Offline propagation of candidate-bound recomputed references into numeric scores."""

from __future__ import annotations

import hashlib
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
PACKAGE = ModuleType("candidate_reference_test")
PACKAGE.__path__ = [str(ROOT / "src/agenticeoecosystem/evaluation")]
sys.modules[PACKAGE.__name__] = PACKAGE
runner_module = importlib.import_module(f"{PACKAGE.__name__}.runners.full_evaluation")
schema = importlib.import_module(f"{PACKAGE.__name__}.dataclasses")
numeric = importlib.import_module(f"{PACKAGE.__name__}.judges.numerical_judge")

TOLERANCE = {"abs_tol": .01, "rel_tol": 0, "floor_scale": 0}


class CandidateReferenceScoringTests(unittest.TestCase):
    """The static GT is replaced only by evidence bound to this candidate and artifact state."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.raster = self.root / "archive" / "ndvi.tif"
        self.raster.parent.mkdir()
        self.raster.write_bytes(b"candidate raster")
        digest = hashlib.sha256(self.raster.read_bytes()).hexdigest()
        self.task = {"task_ID": "q1", "task_text": "Report NDVI mean and max.",
                     "generated_solution": {"steps": [], "final_answer": "Mean 0.40, max 0.90."},
                     "candidate_artifacts": [{"original_path": "/elsewhere/ndvi.tif",
                                              "archived_path": str(self.raster), "sha256": digest}]}
        self.source = self.save("candidate.json", {"metadata": {}, "tasks": [self.task]})
        self.strategies = self.save("strategies.json", {"tasks": [{"task_ID": "q1",
            "evaluation_methods": ["unit_normalized_error", "numeric_tolerance_accuracy"],
            "numerical_field_tolerances": {"ndvi_stats.mean": TOLERANCE, "ndvi_stats.max": TOLERANCE}}]})
        # The static GT disagrees with the candidate's own valid input.
        self.gt = self.save("gt.json", {"q1": {"ndvi_stats": {"mean": .2, "max": .5}}})
        self.benchmark = self.save("benchmark.json", {"tasks": [self.task]})
        criteria = [{"type": kind, "weight": weight, "description": kind}
                    for kind, weight in zip(("scientific_correctness", "grounded_and_complete",
                                             "internal_consistency"), (.5, .3, .2))]
        self.rubrics = self.save("rubrics.json", [{"task_ID": "q1", "task_text": self.task["task_text"],
            "rubric": {"criteria": criteria, "protocol_version": "4.0", "evidence_mode": "trace",
                       "scoring_instructions": "Judge."}}])
        self.config = runner_module.FullEvaluationConfig(
            results_files=(self.source,), output_dir=self.root / "judged", model="offline",
            strategies_file=self.strategies, ground_truths_file=self.gt,
            rubrics_file=self.rubrics, benchmark_file=self.benchmark)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def save(self, filename: str, value: object) -> Path:
        path = self.root / filename
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def entry(self, **changes: object) -> dict:
        runner = runner_module.FullEvaluationRunner(self.config)
        task = schema.TaskSet(tasks=[self.task])[0]
        entry = {"task_ID": "q1", "model": "m", "status": "recomputed",
                 "candidate_fingerprint": runner_module.CandidateEvidence.fingerprint(task),
                 "artifact_state_sha256": runner_module.digest(runner._artifact_state(task)),
                 "recipe_version": "candidate-numeric-recovery/1.0",
                 "reference_values": {"ndvi_stats": {"mean": .4, "max": .9}},
                 "validity": {"status": "valid", "findings": []},
                 "answer_trace_consistency": {"consistent_fraction": 1.0},
                 "evidence_status": "independent_recomputation_from_archived_candidate_inputs"}
        entry.update(changes)
        return entry

    def run_with(self, entry: dict | None, stated: dict) -> tuple[dict, dict]:
        config = self.config
        if entry is not None:
            config = replace(config, candidate_references=self.save("references.json", {"q1": entry}))
        with patch.object(numeric, "extract_candidate_values", return_value=stated) as extract:
            report = runner_module.FullEvaluationRunner(config).run()
        self.extract_calls = extract.call_count
        candidate = report["candidates"][0]
        return candidate["tasks"][0]["phase_results"]["numeric"], candidate

    def test_valid_reference_replaces_static_gt_and_reaches_validated_mean(self) -> None:
        result, summary = self.run_with(self.entry(), {"ndvi_stats.mean": .4, "ndvi_stats.max": .9})
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["reference_kind"], "candidate_bound_recomputation")
        self.assertEqual(result["num_score"], 1.0)
        self.assertEqual(summary["validated_means"]["num_score"], {"n": 1, "mean": 1.0})
        self.assertEqual(summary["candidate_evidence"]["validity"], {"valid": 1})
        self.assertEqual(summary["candidate_evidence"]["answer_trace_consistency"]["mean"], 1.0)

    def test_matching_a_stored_number_without_binding_is_not_credited(self) -> None:
        result, summary = self.run_with(None, {"ndvi_stats.mean": .2, "ndvi_stats.max": .5})
        self.assertEqual(result["status"], "diagnostic")
        self.assertIsNone(summary["validated_means"]["num_score"]["mean"])

    def test_other_candidate_solution_is_rejected(self) -> None:
        result, _ = self.run_with(self.entry(candidate_fingerprint="0" * 64), {"ndvi_stats.mean": .4})
        self.assertEqual(result["candidate_evidence"]["binding"], "rejected")
        self.assertEqual(result["status"], "diagnostic")
        self.assertNotIn("reference_kind", result)

    def test_changed_artifact_content_is_rejected(self) -> None:
        entry = self.entry()
        self.raster.write_bytes(b"overwritten by a later run")
        result, _ = self.run_with(entry, {"ndvi_stats.mean": .4, "ndvi_stats.max": .9})
        self.assertEqual(result["candidate_evidence"]["binding"], "rejected")
        self.assertIn("artifact state", result["candidate_evidence"]["reason"])

    def test_invalid_operation_scores_zero_with_diagnostic_agreement(self) -> None:
        validity = {"status": "invalid", "findings": [
            {"code": "resolution", "detail": "60 m for a 10 m task", "invalidates": True}]}
        result, summary = self.run_with(self.entry(validity=validity),
                                        {"ndvi_stats.mean": .4, "ndvi_stats.max": .9})
        self.assertEqual((result["status"], result["num_score"], result["hit_at_tol"]), ("scored", 0.0, 0.0))
        self.assertEqual(result["independent_agreement"]["num_score"], 1.0)
        self.assertEqual(summary["candidate_evidence"]["validity"], {"invalid": 1})
        self.assertIsNone(summary["candidate_evidence"]["independent_agreement_num_score"]["mean"])

    def test_trace_demonstrated_invalidity_without_values_needs_no_extraction(self) -> None:
        entry = self.entry(status="demonstrated_invalid", reference_values=None,
                           validity={"status": "invalid", "findings": [
                               {"code": "product", "detail": "RSS instead of HRY", "invalidates": True}]})
        result, _ = self.run_with(entry, {"ndvi_stats.mean": .4})
        self.assertEqual((result["status"], result["num_score"]), ("scored", 0.0))
        self.assertIsNone(result["independent_agreement"])
        self.assertEqual(self.extract_calls, 0)

    def test_field_without_candidate_input_scores_zero_even_when_stated(self) -> None:
        entry = self.entry(reference_values={"ndvi_stats": {"mean": .4}})
        result, _ = self.run_with(entry, {"ndvi_stats.mean": .4, "ndvi_stats.max": .9})
        self.assertEqual(result["fields_not_produced"], ["ndvi_stats.max"])
        self.assertEqual(result["num_score"], .5)

    def test_non_finite_reference_value_is_not_a_reference(self) -> None:
        entry = self.entry(reference_values={"ndvi_stats": {"mean": .4, "max": float("nan")}})
        result, _ = self.run_with(entry, {"ndvi_stats.mean": .4, "ndvi_stats.max": .9})
        self.assertEqual(result["fields_not_produced"], ["ndvi_stats.max"])

    def test_unstated_value_is_missing_not_recovered_from_trace(self) -> None:
        result, _ = self.run_with(self.entry(), {"ndvi_stats.mean": .4, "ndvi_stats.max": None})
        missing = [field for field in result["fields"] if field["missing"]]
        self.assertEqual([field["field"] for field in missing], ["ndvi_stats.max"])
        self.assertEqual(result["num_score"], .5)

    def test_unarchived_inputs_get_labelled_conservative_zero(self) -> None:
        entry = self.entry(status="reconstructable", reference_values=None,
                           validity={"status": "unverified", "findings": []})
        result, summary = self.run_with(entry, {"ndvi_stats.mean": .2, "ndvi_stats.max": .5})
        self.assertEqual((result["status"], result["num_score"]), ("scored", 0.0))
        self.assertEqual(result["zero_credit_basis"], "unverifiable_conservative_zero")
        self.assertEqual(self.extract_calls, 0)
        self.assertEqual(summary["candidate_evidence"]["zero_credit_basis"],
                         {"unverifiable_conservative_zero": 1})

    def test_no_supporting_trace_is_reported_but_never_a_reference(self) -> None:
        entry = self.entry(status="no_supporting_trace", reference_values=None,
                           validity={"status": "unverified", "findings": []})
        result, summary = self.run_with(entry, {"ndvi_stats.mean": .2, "ndvi_stats.max": .5})
        self.assertEqual(result["status"], "diagnostic")
        self.assertEqual(summary["candidate_evidence"]["unverified"], 1)

    def test_changed_reference_invalidates_resume(self) -> None:
        stated = {"ndvi_stats.mean": .4, "ndvi_stats.max": .9}
        self.run_with(self.entry(), stated)
        result, _ = self.run_with(self.entry(reference_values={"ndvi_stats": {"mean": .4, "max": .3}}), stated)
        self.assertLess(result["num_score"], 1.0)

    def test_references_require_a_single_candidate_and_matching_keys(self) -> None:
        references = self.save("references.json", {"q1": self.entry()})
        with self.assertRaises(ValueError):
            runner_module.FullEvaluationRunner(replace(
                self.config, results_files=(self.source, self.save("other.json", {"tasks": []})),
                candidate_references=references))
        with self.assertRaises(ValueError):
            runner_module.FullEvaluationRunner(replace(
                self.config, candidate_references=self.save("bad.json", {"q2": self.entry()})))

    def test_runtime_distance_task_ignores_candidate_reference(self) -> None:
        self.gt.write_text(json.dumps({"q1": {"reference_type": "source_dependent_runtime"}}))
        result, _ = self.run_with(self.entry(), {"ndvi_stats.mean": .4, "ndvi_stats.max": .9})
        self.assertEqual(result["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
