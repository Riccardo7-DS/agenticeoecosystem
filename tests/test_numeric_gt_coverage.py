"""Regression checks for requested-field coverage in the numerical benchmark."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('coverage_metrics', ROOT / 'src/agenticeoecosystem/evaluation/metrics/numerical_metrics.py')
METRICS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = METRICS
SPEC.loader.exec_module(METRICS)


class NumericCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.strategies = {e['task_ID']: e for e in json.loads((ROOT / 'data/tasks/task_evaluation_strategies.json').read_text())['tasks']}
        cls.references = json.loads((ROOT / 'data/tasks/ground_truths.json').read_text())

    def fields(self, suffix: str) -> set[str]:
        entry = next(e for k, e in self.strategies.items() if k.endswith('_' + suffix))
        return set(entry['numerical_field_tolerances'])

    def test_previously_omitted_quantities_are_selected(self) -> None:
        self.assertEqual(self.fields('200014'), {'transform.x', 'transform.y'})
        self.assertEqual(self.fields('400015'), {'transform.x', 'transform.y'})
        self.assertEqual(self.fields('400016'), {'transform.x', 'transform.y'})
        self.assertEqual(self.fields('200016'), {'distance_m'})
        self.assertEqual(self.fields('400020'), {'area_km2'})
        self.assertIn('pct_pixels_ndvi_gt_0.6', self.fields('200010'))
        self.assertIn('pct_pixels_ndvi_gt_0.5', self.fields('200012'))

    def test_unrequested_statistics_are_excluded(self) -> None:
        self.assertEqual(self.fields('200010'), {'ndvi_stats.min', 'ndvi_stats.max', 'pct_pixels_ndvi_gt_0.6'})
        self.assertEqual(self.fields('200013'), {'ndvi_stats.mean', 'ndvi_stats.max'})
        self.assertEqual(self.fields('200035'), {'ndvi_stats.mean', 'ndvi_stats.max', 'ndvi_stats.std'})
        self.assertNotIn('ndvi_stats.std', self.fields('200031'))

    def test_configured_reference_paths_resolve_and_missing_fields_count(self) -> None:
        checked = 0
        for task_id, entry in self.strategies.items():
            config = entry.get('numerical_field_tolerances')
            reference = self.references.get(task_id)
            if not config or not reference or '_status' in reference:
                continue
            specs = {k: METRICS.ToleranceSpec(**v) for k, v in config.items()}
            values, tolerances = METRICS.extract_scorable_fields(reference, specs)
            self.assertEqual(METRICS.score_item(values, values, tolerances).hit_at_tol, 1)
            absent = METRICS.score_item({}, values, tolerances)
            self.assertEqual((absent.hit_at_tol, absent.num_score), (0, 0))
            checked += 1
        self.assertGreaterEqual(checked, 19)


if __name__ == '__main__':
    unittest.main()
