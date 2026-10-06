"""Validate reference computation on small synthetic source datasets."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reference_generation', ROOT / 'scripts/references/regenerate_numeric_ground_truths.py')
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReferenceGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.generator = MODULE.NumericalReferenceGenerator(self.root / 'gt.json')

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_extent_preserves_negative_normalized_longitudes(self) -> None:
        path = self.root / 'extent.nc'
        dataset = xr.Dataset({'tp': (('valid_time', 'latitude', 'longitude'), np.ones((1, 3, 4)))},
                             coords={'valid_time': [np.datetime64('2023-01-01')],
                                     'latitude': [-1., 0., 1.], 'longitude': [0., 10., 340., 350.]})
        dataset.tp.attrs['units'] = 'm'
        dataset.to_netcdf(path, engine='netcdf4')
        source = MODULE.ReferenceInput('test', path, 'tp', '2023-01', 1, 'extent', (-20., -1., 0., 1.))
        result = self.generator.calculate(source)
        self.assertEqual(result['extent'], {'west': -20., 'south': -1., 'east': 0., 'north': 1.})

    def test_incomplete_hourly_month_is_rejected(self) -> None:
        path = self.root / 'incomplete.nc'
        dataset = xr.Dataset({'tp': (('valid_time', 'latitude', 'longitude'), np.ones((743, 1, 1)))},
                             coords={'valid_time': np.datetime64('2022-01-01') + np.arange(743).astype('timedelta64[h]'),
                                     'latitude': [0.], 'longitude': [0.]})
        dataset.tp.attrs['units'] = 'm'
        dataset.to_netcdf(path, engine='netcdf4')
        source = MODULE.ReferenceInput('test', path, 'tp', '2022-01', 744, 'monthly_hourly_sum')
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            self.generator.calculate(source)


if __name__ == '__main__':
    unittest.main()
