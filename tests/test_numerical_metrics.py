"""Offline numerical scoring and judge integration regression checks."""
import ast
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import typing
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/agenticeoecosystem/evaluation/numerical_metrics.py'
spec = importlib.util.spec_from_file_location('scalar_metrics_test', SOURCE)
nm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nm
spec.loader.exec_module(nm)


class ScalarMetricTests(unittest.TestCase):
    def test_tolerance_maximum_and_negative_ground_truth(self):
        spec = nm.ToleranceSpec(abs_tol=2, rel_tol=.1, floor_scale=3)
        self.assertEqual(nm.effective_tolerance(-100., spec), 10.)
        self.assertEqual(nm.effective_tolerance(0., spec), 3.)
        self.assertEqual(nm.effective_tolerance(10., nm.ToleranceSpec(abs_tol=4)), 4.)

    def test_boundary_and_exponential_decay(self):
        spec = nm.ToleranceSpec(abs_tol=2.)
        for prediction, expected_hit, expected_score in [(10, 1, 1), (12, 1, 1), (14, 0, .5), (16, 0, .25)]:
            with self.subTest(prediction=prediction):
                result = nm.score_scalar(prediction, 10., spec)
                self.assertEqual(result.hit, expected_hit)
                self.assertEqual(result.num_score, expected_score)
                self.assertEqual(result.absolute_error, abs(prediction - 10))
                self.assertEqual(result.normalized_error, abs(prediction - 10) / 2)

    def test_missing_and_invalid_predictions(self):
        spec = nm.ToleranceSpec(abs_tol=1.)
        invalid = [None, True, False, [], {}, 'bad', '5 metres', '', float('nan'), float('inf'), -float('inf'), 'NaN', 'Infinity']
        for value in invalid:
            with self.subTest(value=value):
                result = nm.score_scalar(value, 5., spec)
                self.assertEqual((result.hit, result.num_score), (0, 0))
                self.assertIsNone(result.absolute_error)
                self.assertIsNone(result.normalized_error)
                json.dumps(result.to_dict(), allow_nan=False)
        self.assertEqual(nm.num_score(' 5e0 ', 5., spec), 1.)
        self.assertEqual(nm.hit_at_tol('bad', 5., spec), 0)

    def test_key_alignment_extra_predictions_and_missing_denominator(self):
        result = nm.score_item({'b': 14., 'a': 10., 'extra': 999}, {'a': 10., 'b': 10., 'c': 10.},
                               nm.uniform_tolerances(['a', 'b', 'c'], abs_tol=2.))
        self.assertEqual([field.field for field in result.fields], ['a', 'b', 'c'])
        self.assertAlmostEqual(result.hit_at_tol, 1 / 3)
        self.assertEqual(result.num_score, .5)
        self.assertTrue(result.fields[2].missing)

    def test_distance_uses_the_same_scalar_metric_and_units(self):
        metres = nm.score_scalar(1004., 1000., nm.ToleranceSpec(abs_tol=2.))
        kilometres = nm.score_scalar(1.004, 1., nm.ToleranceSpec(abs_tol=.002))
        self.assertAlmostEqual(metres.normalized_error, kilometres.normalized_error)
        self.assertAlmostEqual(metres.num_score, .5)
        self.assertAlmostEqual(metres.num_score, kilometres.num_score)

    def test_invalid_tolerances_and_reference_are_configuration_errors(self):
        for params in [{'abs_tol': -1}, {'rel_tol': math.nan}, {'floor_scale': math.inf}, {}, {'abs_tol': True}]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                nm.ToleranceSpec(**params)
        for ground_truth in [math.nan, math.inf, True, '1']:
            with self.subTest(ground_truth=ground_truth), self.assertRaises(ValueError):
                nm.score_scalar(1., ground_truth, nm.ToleranceSpec(abs_tol=1.))
        with self.assertRaises(ValueError):
            nm.score_scalar(0., 0., nm.ToleranceSpec(rel_tol=.1))
        with self.assertRaises(KeyError):
            nm.score_item({}, {'distance_m': 100.}, {})

    def test_legacy_constructor_helpers_and_exact_matching(self):
        spec = nm.ToleranceSpec(tau_abs=2., tau_rel=.1, floor=.01)
        self.assertEqual((spec.abs_tol, spec.rel_tol, spec.floor_scale), (2., .1, .01))
        self.assertEqual(nm.uniform_tolerances(['x'], tau_abs=2.)['x'], nm.ToleranceSpec(abs_tol=2.))
        with self.assertRaises(ValueError):
            nm.ToleranceSpec(abs_tol=1., tau_abs=2.)
        self.assertEqual(nm.score_item({'name': 'Rome'}, {'name': 'Rome'}, {}).num_score, 1.)

    def test_field_selection_supports_all_scalar_quantities_and_literal_dots(self):
        gt = {'distance_m': 1000., 'area_km2': 2., 'transform': {'x': 200., 'y': 400.},
              'pct_pixels_ndvi_gt_0.5': 30., 'unrequested': 999.}
        fields = ['distance_m', 'area_km2', 'transform.x', 'transform.y', 'pct_pixels_ndvi_gt_0.5']
        values, tolerances = nm.extract_scorable_fields(gt, nm.uniform_tolerances(fields, abs_tol=1.))
        self.assertEqual(set(values), set(fields))
        self.assertEqual(nm.score_item(values, values, tolerances).num_score, 1.)
        with self.assertRaises(KeyError):
            nm.extract_scorable_fields(gt, {'missing': nm.ToleranceSpec(abs_tol=1.)})
        self.assertEqual(nm.extract_scorable_fields(gt, {}), ({}, {}))

    def test_legacy_presets_keep_existing_results(self):
        values, tolerances = nm.extract_scorable_fields({'band': {'min': -.2, 'max': .8}, 'lat': 40., 'lon': -3.})
        self.assertEqual(set(values), {'band.min', 'band.max', 'lat', 'lon'})
        self.assertEqual(nm.score_item(values, values, tolerances).hit_at_tol, 1.)
        self.assertEqual(tolerances['lat'].abs_tol, .02)

    def test_cross_item_means_and_explicit_positional_compatibility(self):
        spec = nm.ToleranceSpec(abs_tol=1.)
        good = nm.score_item({'different_name': 1.}, {'x': 1.}, {'x': spec}, align_by_key=False)
        missing = nm.score_item({}, {'x': 1.}, {'x': spec})
        aggregate = nm.aggregate_items([good, missing])
        self.assertEqual((aggregate['hit_at_tol'], aggregate['num_score']), (.5, .5))


class NumericalJudgeTests(unittest.TestCase):
    def setUp(self):
        # Load the actual scoring function without evaluation/__init__ starting agents.
        source = ROOT / 'src/agenticeoecosystem/evaluation/numerical_judge.py'
        names = {'numeric_score_task_solution', '_is_usable_gt', '_has_numeric_component', '_flatten_keys'}
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        self.requests = []
        self.prediction = {'distance_m': 1004.}

        def extract(task, fields, **kwargs):
            self.requests.append(fields)
            return self.prediction

        self.namespace = {'nm': nm, 'Any': typing.Any, 'Dict': typing.Dict, 'Optional': typing.Optional,
                          'Task': SimpleNamespace, 'MODEL_CLAUDE_SONNET': 'offline',
                          'logger': logging.getLogger('offline_numeric'),
                          'extract_candidate_values': extract,
                          'CandidateEvidence': SimpleNamespace(fingerprint=lambda task: 'offline')}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), self.namespace)
        self.task = SimpleNamespace(task_ID='distance', generated_solution=object(), evaluator_scores={},
                                    evaluator_scores_raw={}, evaluator_reasoning={}, evaluator_criteria={}, evaluator_evidence={})
        self.plan = {'evaluation_strategy': 'numerical_answer', 'numerical_field_tolerances': {
            'distance_m': {'abs_tol': 2., 'rel_tol': 0., 'floor_scale': 0.}}}

    def score(self, gt=None, plan=None):
        return self.namespace['numeric_score_task_solution'](
            self.task, {'distance': self.plan if plan is None else plan},
            {'distance': {'distance_m': 1000., 'endpoint': {'lat': 40., 'lon': -3.}} if gt is None else gt})

    def test_distance_dispatch_and_unrounded_diagnostics(self):
        self.score()
        self.assertEqual(self.requests, [['distance_m']])
        self.assertEqual(self.task.evaluator_scores_raw['numeric_tolerance'], .5)
        evidence = self.task.evaluator_evidence['numeric_tolerance']
        self.assertEqual((evidence['hit_at_tol'], evidence['num_score']), (0., .5))
        field = self.task.evaluator_criteria['numeric_tolerance'][0]
        self.assertEqual((field['absolute_error'], field['normalized_error'], field['effective_tol']), (4., 2., 2.))

    def test_no_solution_scores_missing_fields_without_model_call(self):
        self.task.generated_solution = None
        self.score()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.task.evaluator_scores_raw['numeric_tolerance'], 0.)
        self.assertTrue(self.task.evaluator_criteria['numeric_tolerance'][0]['missing'])

    def test_gt_config_fallback_and_status_placeholder_skip(self):
        plan = {'evaluation_strategy': 'numerical_answer'}
        self.score(gt={'distance_m': 1000., 'field_tolerances': self.plan['numerical_field_tolerances']}, plan=plan)
        self.assertEqual(self.requests, [['distance_m']])
        self.requests.clear()
        self.assertIsNone(self.score(gt={'_status': 'runtime_reference_required'}))
        self.assertEqual(self.requests, [])

    def test_invalid_prediction_scores_zero_and_serializes_safely(self):
        self.prediction = {'distance_m': math.nan}
        self.score()
        field = self.task.evaluator_criteria['numeric_tolerance'][0]
        self.assertTrue(field['invalid'])
        self.assertEqual(self.task.evaluator_scores_raw['numeric_tolerance'], 0.)
        json.dumps(self.task.evaluator_criteria, allow_nan=False)

    def test_judge_preserves_full_precision(self):
        self.prediction = {'distance_m': 1004.3}
        self.score()
        expected = 2 ** (-(abs(1004.3 - 1000.) / 2. - 1.))
        field = self.task.evaluator_criteria['numeric_tolerance'][0]
        self.assertEqual(field['num_score'], expected)
        self.assertNotEqual(field['num_score'], round(expected, 4))

    def test_hybrid_scalar_component_uses_identical_scoring(self):
        plan = dict(self.plan, evaluation_strategy='hybrid', evaluation_components=[
            {'type': 'numerical_answer'}, {'type': 'qualitative'}])
        self.score(plan=plan)
        self.assertEqual(self.task.evaluator_scores_raw['numeric_tolerance'], .5)

    def test_runtime_distance_uses_candidate_bound_reference_and_same_metric(self):
        self.namespace['CandidateEvidence'].runtime_distance_reference = lambda task, label: {
            'distance_m': 1000., 'source_sha256': 'verified-source'}
        self.score(gt={'_status': 'runtime_reference', 'reference_type': 'source_dependent_runtime'})
        self.assertEqual(self.task.evaluator_scores_raw['numeric_tolerance'], .5)
        self.assertEqual(self.requests, [['distance_m']])
        self.namespace['CandidateEvidence'].runtime_distance_reference = lambda task, label: None
        self.assertIsNone(self.score(gt={'_status': 'runtime_reference', 'reference_type': 'source_dependent_runtime'}))


if __name__ == '__main__':
    unittest.main()
