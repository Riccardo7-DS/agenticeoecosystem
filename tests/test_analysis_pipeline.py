"""Analysis pipeline: candidate publication, meta score and evidence-based recovery."""
import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

from scripts.analysis.current_judging_analysis import MetaScoreAnalysis
from test_full_evaluation import PACKAGE, schema

path = Path(__file__).resolve().parents[1] / 'scripts/generation/publish_candidate_results.py'
spec = importlib.util.spec_from_file_location('candidate_publication', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
recovery = importlib.import_module(f"{PACKAGE.__name__}.evidence.candidate_evidence")


class CandidatePublicationTests(unittest.TestCase):
    def test_missing_retry_cannot_erase_answer_and_missing_slots_remain(self) -> None:
        cohort = [{'task_ID': key, 'task_text': key} for key in ('a', 'b', 'c')]
        saved = {'task_ID': 'a', 'task_text': 'historical question',
                 'generated_solution': {'final_answer': 'saved'}, 'evaluator_scores': {'old': 2}}
        rows, provenance = module.CandidatePublisher.merge(cohort, [
            {'tasks': [saved]}, {'tasks': [{'task_ID': 'a', 'generated_solution': None}]}], ['base', 'retry'])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['generated_solution']['final_answer'], 'saved')
        self.assertEqual(rows[0]['task_text'], 'historical question')
        self.assertEqual(rows[0]['evaluator_scores'], {})
        self.assertIsNone(rows[1]['generated_solution'])
        self.assertEqual(provenance['a'], 'base')
        self.assertEqual(saved['evaluator_scores'], {'old': 2})

    def test_latest_saved_answer_takes_precedence(self) -> None:
        cohort = [{'task_ID': 'a'}]
        rows, provenance = module.CandidatePublisher.merge(cohort, [
            {'tasks': [{'task_ID': 'a', 'generated_solution': {'final_answer': 'old'}}]},
            {'tasks': [{'task_ID': 'a', 'generated_solution': {'final_answer': 'new'}}]}], ['old', 'new'])
        self.assertEqual(rows[0]['generated_solution']['final_answer'], 'new')
        self.assertEqual(provenance['a'], 'new')

    def test_duplicates_and_outside_cohort_are_rejected(self) -> None:
        for tasks in ([{'task_ID': 'a'}, {'task_ID': 'a'}], [{'task_ID': 'outside'}]):
            with self.assertRaises(ValueError):
                module.CandidatePublisher.merge([{'task_ID': 'a'}], [{'tasks': tasks}], ['source'])


class MetaScoreTests(unittest.TestCase):
    """Check the score formula without external snapshots or API calls."""

    def setUp(self) -> None:
        self.analysis = object.__new__(MetaScoreAnalysis)

    def test_hybrid_missing_validation_halves_aaj(self) -> None:
        record = {'phases': ['aaj', 'numeric'], 'phase_results': {'aaj': {'status': 'scored', 'score': 2}}}
        self.assertEqual(self.analysis.task_score('hybrid', record)['score'], 1)

    def test_numeric_requires_validated_reference(self) -> None:
        record = {'phase_results': {'numeric': {'status': 'diagnostic', 'num_score': 1}}}
        self.assertEqual(self.analysis.task_score('numerical_answer', record)['score'], 0)
        record['phase_results']['numeric']['status'] = 'scored'
        self.assertEqual(self.analysis.task_score('numerical_answer', record)['score'], 2)

    def test_unevaluated_category_stays_zero(self) -> None:
        self.assertEqual(self.analysis.task_score('qualitative', {})['score'], 0)
        self.assertFalse(self.analysis.task_score('qualitative', {})['evaluated'])
        self.assertEqual(sum(self.analysis.weights.values()), 265)
        self.assertEqual(sum(self.analysis.weights[k] for k in self.analysis.included_categories), 160)
        score = sum(weight * (2 if category == 'agent_as_judge' else 0)
                    for category, weight in self.analysis.weights.items()) / 265
        self.assertAlmostEqual(score, 72 / 265)

    def test_inspection_only_hybrid_validation_is_zero(self) -> None:
        record = {'phases': ['aaj'], 'phase_results': {'aaj': {'status': 'scored', 'score': 2}}}
        self.assertEqual(self.analysis.task_score('hybrid', record)['score'], 1)

    def test_hybrid_common_scale(self) -> None:
        record = {'phases': ['aaj', 'numeric', 'deterministic'], 'phase_results': {
            'aaj': {'status': 'scored', 'score': 2},
            'numeric': {'status': 'scored', 'num_score': 0.5},
            'deterministic': {'status': 'scored', 'metrics': {'exact_match': 0.5}},
        }}
        self.assertAlmostEqual(self.analysis.task_score('hybrid', record)['score'], 4 / 3)

    def test_candidate_archive_relocation_does_not_invalidate_scores(self) -> None:
        old = {'task_text': 'Question', 'generated_solution': {'final_answer': 'Answer'},
               'candidate_artifacts': [{'sha256': 'same', 'archived_path': '/old'}]}
        current = {**old, 'candidate_artifacts': [{'sha256': 'same', 'archived_path': '/new'}]}
        self.assertTrue(self.analysis.same_candidate_answer(old, current))
        current['generated_solution'] = {'final_answer': 'Changed answer'}
        self.assertFalse(self.analysis.same_candidate_answer(old, current))

    def test_candidate_evidence_change_invalidates_scores(self) -> None:
        old = {'task_text': 'Question', 'generated_solution': {'final_answer': 'Answer'},
               'candidate_artifacts': [{'sha256': 'old'}]}
        current = {**old, 'candidate_artifacts': [{'sha256': 'new'}]}
        self.assertFalse(self.analysis.same_candidate_answer(old, current))


class RecoveryTests(unittest.TestCase):
    def test_only_completely_missing_or_invalid_fields_allow_reference_independent_zero(self) -> None:
        self.assertTrue(recovery.all_numeric_outputs_missing([{"missing": True}, {"invalid": True}]))
        self.assertFalse(recovery.all_numeric_outputs_missing([]))
        self.assertFalse(recovery.all_numeric_outputs_missing([{"missing": True}, {"missing": False, "invalid": False}]))

    def task(self, text: str, error: bool = False) -> object:
        return schema.Task(task_ID="TASK_260419_000001_100001", task_text="List themes",
            generated_solution=schema.Solution(final_answer="1. **invented**", steps=[schema.Step(
                function_name="list_themes", tool_result={"isError": error,
                "structuredContent": {"result": text}})]))

    def test_reference_comes_from_source_and_requires_complete_count(self) -> None:
        task = self.task("Available themes (2):\n- oceans (9 narratives)\n- atmosphere (6 narratives)\n")
        self.assertEqual(recovery.recorded_theme_reference(task)["reference_set"], ["oceans", "atmosphere"])
        self.assertIsNone(recovery.recorded_theme_reference(self.task("Available themes (2):\n- oceans (9 narratives)\n")))
        self.assertIsNone(recovery.recorded_theme_reference(self.task("Available themes (1):\n- oceans (9 narratives)\n", True)))

    def test_catalogue_claim_in_final_answer_cannot_validate_reference(self) -> None:
        task = self.task("Available themes (1):\n- oceans (9 narratives)\n")
        gt = {"catalogue_snapshot": {"x": {"description": "future climate"}},
              "provenance": {"tool": "list_cds_datasets"}}
        task.generated_solution.final_answer = "[x] future climate"
        self.assertFalse(recovery.recorded_catalogue_matches(task, gt))
        task.generated_solution.steps[0].function_name = "list_cds_datasets"
        task.generated_solution.steps[0].tool_result = {"structuredContent": {"result": "[x]\nfuture climate"}}
        self.assertTrue(recovery.recorded_catalogue_matches(task, gt))
        task.generated_solution.steps[0].tool_result = {"isError": True, "structuredContent": {"result": "[x]\nfuture climate"}}
        self.assertFalse(recovery.recorded_catalogue_matches(task, gt))
