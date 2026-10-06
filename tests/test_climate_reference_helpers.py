"""Offline checks of actual climate tool bodies on small synthetic grids."""
import ast
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import xarray as xr

SOURCE = Path(__file__).resolve().parents[1] / 'src/agenticeoecosystem/servers/climate/server.py'


def load_tools():
    names = {'clip_to_bbox', 'convert_hourly_to_daily', 'convert_daily_to_monthly'}
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    class InlineExecutor:
        async def run_in_executor(self, executor, function):
            # Exercise the real tool body without sandbox-dependent thread wakeups.
            return function()

    namespace = {'xr': xr, 'asyncio': SimpleNamespace(get_running_loop=InlineExecutor), 'Optional': Optional,
                 'resolve_run_path': Path, '_coerce_output_path': lambda path, prefix: path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class ClimateReferenceTests(unittest.TestCase):
    def test_clip_normalizes_western_longitudes_and_rejects_empty_region(self):
        tools = load_tools()
        with tempfile.TemporaryDirectory() as directory:
            source, target = [Path(directory) / name for name in ('source.nc', 'clip.nc')]
            xr.Dataset({'tp': (('latitude', 'longitude'), np.arange(9).reshape(3, 3))},
                       coords={'latitude': [43., 42., 41.], 'longitude': [351., 352., 353.]}).to_netcdf(source)
            result = asyncio.run(tools['clip_to_bbox'](str(source), 43., 41., -8., -9., str(target)))
            self.assertNotIn('Error:', result)
            with xr.open_dataset(target) as clipped:
                np.testing.assert_equal(clipped.longitude.values, [-9., -8.])
                self.assertEqual(clipped.tp.shape, (3, 2))
            result = asyncio.run(tools['clip_to_bbox'](str(source), 10., 9., 2., 1., str(target)))
            self.assertIn('no grid-cell centres', result)

    def test_valid_time_resampling_retains_missing_cells_and_units(self):
        tools = load_tools()
        with tempfile.TemporaryDirectory() as directory:
            source, daily, monthly = [Path(directory) / name for name in ('hourly.nc', 'daily.nc', 'monthly.nc')]
            xr.Dataset({'tp': (('valid_time', 'x'), [[1., np.nan], [2., np.nan], [4., np.nan]], {'units': 'm'})},
                       coords={'valid_time': np.array(['2023-01-01T00', '2023-01-01T12', '2023-01-02T00'], dtype='datetime64[h]')}).to_netcdf(source)
            result = asyncio.run(tools['convert_hourly_to_daily'](str(source), str(daily), 'sum', 'tp'))
            self.assertNotIn('Error', result)
            result = asyncio.run(tools['convert_daily_to_monthly'](str(daily), str(monthly), 'sum', 'tp'))
            self.assertNotIn('Error', result)
            with xr.open_dataset(monthly) as data:
                self.assertEqual(data.tp.values[0, 0], 7.)
                self.assertTrue(np.isnan(data.tp.values[0, 1]))
                self.assertEqual(data.tp.attrs['units'], 'm')
