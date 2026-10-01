"""Numerical evidence, delegation and candidate-isolation checks for AAJ."""
import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[1] / 'src/agenticeoecosystem/evaluation/gis_judge_agent.py'
spec = importlib.util.spec_from_file_location('gis_judge_agent', MODULE)
gis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gis)


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_rejects_other_candidate_and_symlink(self):
        allowed = self.root / 'allowed.nc'
        allowed.touch()
        other = self.root / 'other.nc'
        other.touch()
        alias = self.root / 'alias.nc'
        alias.symlink_to(other)
        inspector = gis.ArtifactInspector([allowed])
        self.assertEqual(inspector.resolve(str(allowed)), allowed)
        for path in (other, alias):
            with self.assertRaises(ValueError):
                inspector.resolve(str(path))

    def test_extent_uses_centres_and_explicit_tolerance(self):
        import xarray as xr
        path = self.root / 'extent.nc'
        xr.Dataset(coords={'latitude': [42., 41.], 'longitude': [351., 352.]}).to_netcdf(path)
        inspector = gis.ArtifactInspector([path])
        result = inspector.inspect_extent(str(path), [-9., 41., -8., 42.])
        self.assertEqual(result['extent'], [-9., 41., -8., 42.])
        self.assertIsNone(result['verified'])
        self.assertTrue(inspector.inspect_extent(str(path), result['extent'], atol=0)['verified'])
        self.assertFalse(inspector.inspect_extent(str(path), [-10., 41., -8., 42.], atol=.1)['verified'])

    def test_monthly_validation_accepts_equivalent_calendar_labels(self):
        import numpy as np
        import xarray as xr
        source, output = self.root / 'daily.nc', self.root / 'monthly.nc'
        xr.Dataset({'t': ('valid_time', [2., 4.])}, coords={
            'valid_time': np.array(['2023-01-01', '2023-01-02'], dtype='datetime64[D]')}).to_netcdf(source)
        xr.Dataset({'t': ('time', [3.])}, coords={
            'time': np.array(['2023-01-01'], dtype='datetime64[D]')}).to_netcdf(output)
        inspector = gis.ArtifactInspector([source, output])
        self.assertTrue(inspector.validate_netcdf('resample', [str(source)], str(output),
                                               't', frequency='ME', method='mean')['verified'])

    def test_histogram_mask_counts_and_edges(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        path = self.root / 'index.tif'
        with rasterio.open(path, 'w', driver='GTiff', width=3, height=2,
                           count=1, dtype='float32', nodata=-9999,
                           crs='EPSG:4326', transform=from_origin(0, 2, 1, 1)) as dst:
            dst.write(np.array([[0, .2, .4], [.6, .8, -9999]], dtype='float32'), 1)
        inspector = gis.ArtifactInspector([path])
        for bins in (100, 256):
            result = inspector.inspect_raster(str(path), bins=bins)
            self.assertEqual(len(result['histogram']), bins)
            self.assertEqual(len(result['bin_edges']), bins + 1)
            self.assertEqual(sum(result['histogram']), 5)

    def test_affine_monthly_flux_and_incorrect_output(self):
        import numpy as np
        import xarray as xr
        src, dst = self.root / 'source.nc', self.root / 'output.nc'
        xr.Dataset({'slhf': ('x', [-86400., 172800., np.nan])}).to_netcdf(src)
        xr.Dataset({'slhf': ('x', [-1., 2., np.nan], {'units': 'W m**-2'})}).to_netcdf(dst)
        inspector = gis.ArtifactInspector([src, dst])
        args = dict(operation='affine', input_paths=[str(src)], output_path=str(dst),
                    variable='slhf', factor=1/86400, expected_units='W m**-2')
        self.assertTrue(inspector.validate_netcdf(**args)['verified'])
        args['factor'] = 1/3600
        self.assertFalse(inspector.validate_netcdf(**args)['verified'])

    def test_resample_wind_mean_and_coordinate_mismatch(self):
        import numpy as np
        import xarray as xr
        hourly, daily = self.root/'hourly.nc', self.root/'daily.nc'
        data = xr.DataArray([1., 3., 5., 7.], dims='time',
                            coords={'time': np.array(['2023-01-01T00', '2023-01-01T12',
                                                      '2023-01-02T00', '2023-01-02T12'], dtype='datetime64[h]')})
        data.to_dataset(name='t').to_netcdf(hourly)
        data.resample(time='D').mean().to_dataset(name='t').to_netcdf(daily)
        inspector = gis.ArtifactInspector([hourly, daily])
        self.assertTrue(inspector.validate_netcdf('resample', [str(hourly)], str(daily), 't')['verified'])
        u, v, wind = [self.root/f'{name}.nc' for name in ('u', 'v', 'wind')]
        for path, name, values in [(u, 'u', [3., 0.]), (v, 'v', [4., 2.]), (wind, 'speed', [5., 2.])]:
            xr.Dataset({name: ('x', values, {'units': 'm s**-1'})}, coords={'x': [1, 2]}).to_netcdf(path)
        inspector = gis.ArtifactInspector([u, v, wind])
        self.assertTrue(inspector.validate_netcdf('wind_speed', [str(u), str(v)], str(wind),
                                               'u', output_variable='speed', second_variable='v')['verified'])
        samples = []
        for index, value in enumerate([280., 284., 288.]):
            path = self.root/f'sst{index}.nc'
            xr.Dataset({'sst': (('time','x'), [[value, np.nan]], {'units':'K'})},
                       coords={'time':[np.datetime64(f'{1990+index*15}-07-01')], 'x':[1,2]}).to_netcdf(path)
            samples.append(path)
        mean = self.root/'mean.nc'
        xr.Dataset({'sst': ('x', [284., np.nan], {'units':'K'})}, coords={'x':[1,2]}).to_netcdf(mean)
        inspector = gis.ArtifactInspector(samples+[mean])
        self.assertTrue(inspector.validate_netcdf('field_mean', list(map(str,samples)), str(mean),'sst')['verified'])
        xr.Dataset({'sst': ('x', [284., np.nan], {'units':'K'})}, coords={'x':[9,10]}).to_netcdf(mean)
        with self.assertRaises(ValueError):
            inspector.validate_netcdf('field_mean', list(map(str,samples)), str(mean),'sst')

    def test_specialist_records_evidence_without_api(self):
        class Message:
            content = None
            tool_calls = [SimpleNamespace(id='one', function=SimpleNamespace(
                name='inspect_netcdf', arguments=json.dumps({'path':'forbidden.nc'})))]
            def model_dump(self, **kwargs):
                return {'role':'assistant', 'tool_calls': []}
        messages = iter([Message(), SimpleNamespace(content='Unavailable evidence.', tool_calls=[],
                         model_dump=lambda **kw: {'role':'assistant','content':'Unavailable evidence.'})])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=next(messages))]))))
        result = gis.GISJudgeAgent(client, 'mock', gis.ArtifactInspector([])).examine('Check file')
        self.assertIn('error',result['tool_calls'][0]['result'])
        self.assertEqual(result['summary'],'Unavailable evidence.')

    def test_aaj_delegation_and_serialized_audit(self):
        from unittest.mock import patch
        from types import ModuleType
        # Avoid initializing unrelated candidate pipelines through evaluation/__init__.
        package = ModuleType('agenticeoecosystem.evaluation')
        package.__path__ = [str(MODULE.parent)]
        credentials = ModuleType('agenticeoecosystem.config.credentials')
        credentials.get_openai_client = lambda **kwargs: None
        previous = {name: sys.modules.get(name) for name in (
            'agenticeoecosystem.evaluation', 'agenticeoecosystem.config.credentials')}
        sys.modules['agenticeoecosystem.evaluation'] = package
        sys.modules['agenticeoecosystem.config.credentials'] = credentials
        try:
            from agenticeoecosystem.evaluation import agent_judge
            from agenticeoecosystem.evaluation.dataclasses import Task, Solution
        finally:
            for name, original in previous.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
        import xarray as xr

        path = self.root/'candidate.nc'
        xr.Dataset({'t': ('x', [1., 2.], {'units': 'C'})}).to_netcdf(path)

        class Message:
            def __init__(self, name=None, args=None, content=None):
                self.content = content
                self.tool_calls = [] if name is None else [SimpleNamespace(
                    id=name, function=SimpleNamespace(name=name, arguments=json.dumps(args)))]
            def model_dump(self, **kwargs):
                return {'role': 'assistant', 'content': self.content}

        sequence = iter([
            Message('ask_gis_agent', {'question': 'Inspect saved temperature units.'}),
            Message('inspect_netcdf', {'path': str(path)}),
            Message(content='The saved t variable uses C and has two values.'),
            Message('submit_verdict', {'score': 2, 'reasoning': 'Verified saved units.'}),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=next(sequence))]))))
        task = Task(task_ID='test', task_text='Inspect temperature.',
                    generated_solution=Solution(final_answer=str(path)))
        with patch.object(agent_judge, 'get_openai_client', return_value=client):
            agent_judge.judge_task_agentic(task, model='mock', artifact_paths=[str(path)])
        self.assertEqual(task.evaluator_scores['agent_judge'], 2)
        evidence = task.evaluator_tool_calls['agent_judge'][0]['evidence']
        self.assertEqual(evidence['tool_calls'][0]['result']['variables']['t']['units'], 'C')
        restored = Task.model_validate_json(task.model_dump_json())
        self.assertEqual(restored.evaluator_tool_calls, task.evaluator_tool_calls)


if __name__ == '__main__':
    unittest.main()
