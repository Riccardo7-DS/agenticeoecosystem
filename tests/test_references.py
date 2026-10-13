"""Reference discovery and completion: mathematics and isolation without network calls."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/references'))
discovery = importlib.import_module('ground_truth_discovery')
completion = importlib.import_module('reference_completion')


class DiscoveryTests(unittest.TestCase):
    def test_literal_catalogue_does_not_execute_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'server.py'
            path.write_text('raise RuntimeError("must not execute")\nCATALOGUE = {"x": [1, 2]}\n')
            self.assertEqual(discovery.literal_constant(path, 'CATALOGUE'), {'x': [1, 2]})
            path.write_text('CATALOGUE = dict(x=[1, 2])')
            with self.assertRaises(ValueError):
                discovery.literal_constant(path, 'CATALOGUE')
            path.write_text('BANDS = ["red"]\nCATALOGUE = {"bands": BANDS + ["nir"]}\n')
            self.assertEqual(discovery.literal_constant(path, 'CATALOGUE'), {'bands': ['red', 'nir']})

    def test_excluded_scope_and_complete_snapshot_do_not_copy_candidate_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = discovery.GroundTruthDiscovery(discovery.DiscoveryConfig(Path(directory)))
            self.assertEqual(len(worker.strategies), 104)
            task_id = 'TASK_260419_000001_100001'
            worker.records[task_id] = {}
            worker.cache['dashboard'] = ([{'title': 'A', 'theme': 'oceans'},
                                          {'title': 'B', 'theme': ['atmosphere']}], {'snapshot_sha256': 'test'})
            worker.discover_set(task_id)
            self.assertEqual(worker.proposals[task_id]['reference_set'], ['atmosphere', 'oceans'])
            self.assertEqual(worker.proposals[task_id]['status'], 'proposed_requires_review')
            self.assertFalse((Path(directory) / 'ground_truths.json').exists())

    def test_classification_task_selects_projection_and_forecast_not_reanalysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = discovery.GroundTruthDiscovery(discovery.DiscoveryConfig(Path(directory)))
            task_id = 'TASK_260713_000014_600014'
            worker.records[task_id] = {}
            worker.cache['_CDS_DATASETS'] = ({'projections-cmip6': {},
                'seasonal-original-single-levels': {}, 'reanalysis-era5-single-levels': {}}, {})
            worker.discover_set(task_id)
            self.assertEqual(set(worker.proposals[task_id]['reference_dictionary']),
                             {'projections-cmip6', 'seasonal-original-single-levels'})


class ReferenceCompletionTests(unittest.TestCase):
    def test_fixed_geometry_and_output_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sources').mkdir()
            worker = completion.ReferenceCompletion(completion.CompletionConfig(root))
            task = next(key for key in worker.tasks if key.endswith('_400020'))
            reference = worker.geometry(task)
            self.assertAlmostEqual(reference['area_km2'], 4733663.738810978, places=5)
            arguments = worker.resolve_arguments({'destination': 'output/task/a.tif'}, {})
            self.assertTrue(Path(arguments['destination']).is_relative_to(root))
            with self.assertRaises(ValueError):
                worker.resolve_arguments({'source': '<missing>'}, {})

    def test_axis_order_and_tool_errors(self) -> None:
        point = completion.ReferenceCompletion.projected({'lon': 12, 'lat': 0}, 'EPSG:32633')
        self.assertAlmostEqual(point['x'], 166021.44308, places=3)
        self.assertAlmostEqual(point['y'], 0)
        with self.assertRaises(ValueError):
            completion.decode_result({'isError': True})

    def test_aaj_routing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = completion.ReferenceCompletion(completion.CompletionConfig(root))
            worker.prepare_profile()
            self.assertEqual(json.loads((root / 'aaj_routing.json').read_text())['count'], 48)
            profile = json.loads((root / 'evaluation_strategies_aaj.json').read_text())
            inventory = next(row for row in profile['tasks'] if row['task_ID'].endswith('_200005'))
            self.assertEqual(inventory['evaluation_methods'], ['agent_as_judge'])
            self.assertFalse((root / 'ground_truths.json').exists())

    def test_evi_formula_and_mask(self) -> None:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / 'input.tif', Path(directory) / 'evi.tif'
            data = np.array([[[.1, -9999]], [[.2, -9999]], [[.6, -9999]]], dtype='float32')
            with rasterio.open(source, 'w', driver='GTiff', width=2, height=1, count=3,
                               dtype='float32', nodata=-9999, crs='EPSG:4326',
                               transform=from_origin(2, 13, .01, .01)) as dataset:
                dataset.write(data)
            completion.ReferenceCompletion.local_raster_operation('compute_evi', {
                'source': str(source), 'destination': str(destination), 'blue_band_index': 1,
                'red_band_index': 2, 'nir_band_index': 3})
            with rasterio.open(destination) as dataset:
                result = dataset.read(1, masked=True)
                self.assertAlmostEqual(float(result[0, 0]), 2.5 * (.6 - .2) / (.6 + 6*.2 - 7.5*.1 + 1), places=6)
                self.assertTrue(result.mask[0, 1])

    def test_empty_inventory_requires_successful_evidence(self) -> None:
        solution = {'steps': [{'function_name': 'list_papers', 'tool_result': {
            'content': [{'type': 'text', 'text': '{"papers": []}'}], 'isError': False}}]}
        reference = completion.ReferenceReviewExporter.inventory_reference(solution)
        self.assertEqual(reference['reference_set'], [])
        legacy = {'steps': [{'function_name': 'list_papers', 'tool_result': [
            {'type': 'text', 'text': '{"papers": ["2112.02447v2"]}'}]}]}
        self.assertEqual(completion.ReferenceReviewExporter.inventory_reference(legacy)['reference_set'], ['2112.02447v2'])
        solution['steps'][0]['tool_result']['isError'] = True
        self.assertEqual(completion.ReferenceReviewExporter.inventory_reference(solution)['status'], 'unverified')
