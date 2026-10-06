"""Regression for mixed-area HRV/VIS cropping, without live downloads."""
import ast
from copy import deepcopy
from pathlib import Path
from typing import Any
import unittest


class SceneDouble(dict):
    """Model the mixed-area comparison failure raised by satpy Scene.crop."""
    def copy(self, datasets: list[str] | None = None) -> 'SceneDouble':
        return SceneDouble({k: deepcopy(v) for k, v in self.items()
                            if datasets is None or k in datasets})

    def crop(self, ll_bbox: tuple[float, ...]) -> 'SceneDouble':
        if len({v['area'] for v in self.values()}) > 1:
            raise ValueError("Can't compare areas of different types")
        return SceneDouble({k: dict(v, bbox=ll_bbox) for k, v in self.items()})


class MixedAreaCropTests(unittest.TestCase):
    def test_stacked_windows_preserve_pixels_and_georeferencing(self) -> None:
        import numpy as np
        import xarray as xr
        from pyresample.geometry import AreaDefinition, StackedAreaDefinition

        source = Path(__file__).resolve().parents[1] / 'src/agenticeoecosystem/servers/eumetsat/server.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_crop_stacked_band')
        namespace = {'Any': Any}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        upper = AreaDefinition('upper', 'upper', 'latlong', 'EPSG:4326', 4, 2, (0, 2, 4, 4))
        lower = AreaDefinition('lower', 'lower', 'latlong', 'EPSG:4326', 4, 2, (1, 0, 5, 2))
        data = xr.DataArray(np.arange(16, dtype=float).reshape(4, 4), dims=('y', 'x'),
                            attrs={'area': StackedAreaDefinition(upper, lower), 'units': '%'})
        crop = namespace['_crop_stacked_band']
        both = crop(data, [0, 0, 5, 4])
        self.assertEqual(both.shape, (4, 5))
        self.assertEqual(np.isfinite(both.values).sum(), 16)
        self.assertEqual(float(both.mean()), float(data.mean()))
        self.assertEqual(both.attrs['area'].shape, both.shape)
        self.assertEqual(both.attrs['units'], '%')
        single = crop(data, [0, 2.1, 4, 4])
        np.testing.assert_array_equal(single.values, data.values[:2])
        self.assertIsInstance(single.attrs['area'], AreaDefinition)
        with self.assertRaisesRegex(ValueError, 'does not overlap'):
            crop(data, [20, 20, 21, 21])

    def test_hrv_and_visible_channels_crop_independently(self) -> None:
        source = Path(__file__).resolve().parents[1] / 'src/agenticeoecosystem/servers/eumetsat/server.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_crop_selected_bands')
        namespace = {'Any': Any}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        scene = SceneDouble(VIS006={'area': 'single'}, HRV={'area': 'stacked'})
        bbox = [36.6, -1.5, 37., -1.1]
        with self.assertRaises(ValueError):
            scene.crop(tuple(bbox))
        cropped = namespace['_crop_selected_bands'](scene, ['VIS006', 'HRV', 'unavailable'], bbox)
        self.assertEqual(cropped['VIS006']['bbox'], tuple(bbox))
        self.assertEqual(cropped['HRV']['bbox'], tuple(bbox))
        self.assertNotIn('bbox', scene['HRV'])


if __name__ == '__main__':
    unittest.main()
