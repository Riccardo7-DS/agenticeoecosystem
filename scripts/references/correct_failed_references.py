"""Recompute ground truths that failed the 2026-10-12 reference validation.

Writes staged references only (``ground_truths_corrected.json``) plus every tool
call, geocoder snapshot and artifact under ``--output-dir``. Canonical
``data/tasks/ground_truths.json`` is not modified.

Imagery rule (also the rule for candidate-bound recomputation): a pixel is valid
when the Sentinel Hub dataMask is 1 (or, without a mask, not all bands are 0)
and every reflectance band used is > 0. Indices are computed in float64 and
statistics are over valid pixels only. Reprojection fill is NaN, never 0.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from dotenv import load_dotenv
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
SERVERS = ROOT / 'src/agenticeoecosystem/servers'
NOMINATIM = 'https://nominatim.openstreetmap.org/search'
NARRATIVES = 'https://esa-eodashboards.github.io/eodashboard-narratives/narratives.json'
USER_AGENT = 'agenticeoecosystem-reference-correction'
OLD_REVIEW = ROOT / 'data/reference_review_20261012'
OLD_ARTIFACTS = ROOT / 'data/reference_completion_20261012/artifacts'
OLD_RETRY_ARTIFACTS = ROOT / 'data/reference_completion_retry_20261012/artifacts'
TARGET_GSD_M = 10.0
MAX_PX = 2500
CLOUD_SCL = (3, 8, 9, 10)  # cloud shadow, medium/high cloud, cirrus


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(raw: dict[str, Any]) -> Any:
    """Return parsed tool output, refusing tool errors as reference values."""
    if raw.get('isError'):
        raise RuntimeError(f'Tool error: {raw}')
    texts = [b['text'] for b in raw.get('content', []) if b.get('type') == 'text']
    if len(texts) != 1:
        return texts
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        if texts[0].lstrip().lower().startswith('error'):
            raise RuntimeError(texts[0])
        return texts[0]


def gsd_m(bbox: list[float], width: int, height: int) -> tuple[float, float]:
    west, south, east, north = bbox
    lat = math.radians((south + north) / 2)
    return ((east - west) * 111320 * math.cos(lat) / width, (north - south) * 110574 / height)


def size_for_gsd(bbox: list[float], target: float = TARGET_GSD_M) -> tuple[int, int]:
    west, south, east, north = bbox
    lat = math.radians((south + north) / 2)
    return (math.ceil((east - west) * 111320 * math.cos(lat) / target),
            math.ceil((north - south) * 110574 / target))


def stats(values: np.ndarray) -> dict[str, float]:
    return {'min': float(values.min()), 'max': float(values.max()), 'mean': float(values.mean()),
            'std': float(values.std()), 'valid_pixels': int(values.size)}


class RunningStats:
    """Streaming min/max/mean/std for tiled rasters."""

    def __init__(self) -> None:
        self.n, self.total, self.squares = 0, 0.0, 0.0
        self.low, self.high = math.inf, -math.inf

    def add(self, values: np.ndarray) -> None:
        if values.size:
            self.n += values.size
            self.total += float(values.sum())
            self.squares += float(np.square(values).sum())
            self.low, self.high = min(self.low, float(values.min())), max(self.high, float(values.max()))

    def result(self) -> dict[str, float]:
        mean = self.total / self.n
        return {'min': self.low, 'max': self.high, 'mean': mean,
                'std': math.sqrt(max(self.squares / self.n - mean ** 2, 0.0)), 'valid_pixels': self.n}


def evalscript(bands: list[str]) -> str:
    """Raw float reflectance plus dataMask and SCL, so validity is explicit."""
    inputs = json.dumps([*bands, 'dataMask', 'SCL'])
    values = ','.join(f's.{b}' for b in [*bands, 'dataMask', 'SCL'])
    return ('//VERSION=3\nfunction setup(){return{input:[{bands:' + inputs + '}],'
            f'output:{{bands:{len(bands) + 2},sampleType:"FLOAT32"}}}}}}\n'
            f'function evaluatePixel(s){{return[{values}]}}')


@dataclass
class Config:
    output_dir: Path
    tool_timeout: int = 1800


class ReferenceCorrection:
    """Replay corrected workflows through the benchmark MCP servers and stage references."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.out = config.output_dir.resolve()
        self.artifacts = self.out / 'artifacts'
        for sub in ('sources', 'artifacts'):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        os.environ['EO_ARTIFACT_ROOT'] = str(self.artifacts)
        self.tasks = {t['task_ID']: t for t in json.loads((ROOT / 'data/tasks/benchmark_tasks.json').read_text())['tasks']}
        self.sessions: dict[str, Any] = {}
        self.journal: dict[str, list[dict[str, Any]]] = {}
        self.references: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}
        existing = self.out / 'ground_truths_corrected.json'
        if existing.exists():
            self.references = json.loads(existing.read_text())

    # ---- persistence -------------------------------------------------------
    def save(self, name: str, payload: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        tmp.replace(path)
        return path

    def task_id(self, suffix: str) -> str:
        return next(t for t in self.tasks if t.endswith('_' + suffix))

    def record(self, task_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.journal.setdefault(task_id, []).append({'kind': kind, 'recorded_at_utc': now(), **payload})
        self.save(f'sources/{task_id}_tool_calls.json', self.journal[task_id])

    def stage(self, task_id: str, reference: dict[str, Any]) -> None:
        reference = {'task_ID': task_id, 'task_text': self.tasks[task_id]['task_text'],
                     'recorded_at_utc': now(), **reference,
                     'provenance_log': str((self.out / 'sources' / f'{task_id}_tool_calls.json').relative_to(ROOT))}
        self.references[task_id] = reference
        self.save('ground_truths_corrected.json', self.references)
        print(f'[{now()}] staged {task_id}', flush=True)

    # ---- external sources -------------------------------------------------
    def server(self, name: str) -> dict[str, Any]:
        if name == 'eoss':
            return {'transport': 'streamable_http', 'url': os.environ['GRAPHRAG_MCP_URL'],
                    'headers': {'Authorization': os.environ['GRAPHRAG_AUTH_HEADER']}}
        paths = {'cdse': 'cdse_geocode/server.py', 'eumetsat': 'eumetsat/server.py', 'climate': 'climate/server.py'}
        return {'transport': 'stdio', 'command': sys.executable, 'env': dict(os.environ),
                'args': [str(SERVERS / paths[name])]}

    async def invoke(self, stack: AsyncExitStack, task_id: str, server: str, tool: str, args: dict[str, Any]) -> Any:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        if server not in self.sessions:
            client = MultiServerMCPClient({server: self.server(server)})
            self.sessions[server] = await stack.enter_async_context(client.session(server))
        result = await asyncio.wait_for(self.sessions[server].call_tool(tool, args), timeout=self.config.tool_timeout)
        raw = result.model_dump(mode='json', exclude_none=True)
        for block in raw.get('content', []):
            if block.get('type') == 'image':
                block.pop('data', None)
        self.record(task_id, 'tool_call', {'server': server, 'tool': tool, 'arguments': args, 'result': raw})
        return decode(raw)

    def nominatim(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {'format': 'jsonv2', 'limit': 1, **params}
        time.sleep(1.1)
        response = requests.get(NOMINATIM, params=query, headers={'User-Agent': USER_AGENT}, timeout=30)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            raise RuntimeError(f'No geocoder result for {params}')
        self.record(task_id, 'geocoder_snapshot', {'url': NOMINATIM, 'params': query, 'result': rows[0]})
        return rows[0]

    def town_window(self, task_id: str, query: str, half_deg: float = 0.1) -> tuple[list[float], dict[str, Any]]:
        """Agricultural surroundings: town-centre point +/- half_deg (about a 20 km box)."""
        row = self.nominatim(task_id, {'q': query, 'featureType': 'settlement'})
        lat, lon = float(row['lat']), float(row['lon'])
        bbox = [round(lon - half_deg, 6), round(lat - half_deg, 6), round(lon + half_deg, 6), round(lat + half_deg, 6)]
        return bbox, {'geocode_query': query, 'display_name': row['display_name'], 'osm_type': row.get('osm_type'),
                      'osm_id': row.get('osm_id'), 'centre': [lon, lat],
                      'area_definition': f'town-centre point +/- {half_deg} deg'}

    # ---- imagery -----------------------------------------------------------
    async def fetch_s2(self, stack, task_id: str, name: str, bbox: list[float], bands: list[str],
                       start: str, end: str, max_cc: float | None) -> Path:
        width, height = size_for_gsd(bbox)
        if width > MAX_PX or height > MAX_PX:
            raise ValueError(f'{name}: {width}x{height} px exceeds the fetch limit; tile it')
        path = self.artifacts / task_id / f'{name}.tif'
        if path.exists():  # resume: reuse a download already journalled for this task
            self.record(task_id, 'reused_artifact', {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
            return path
        args = {'bbox': bbox, 'start_date': start, 'end_date': end, 'collection': 'L2A',
                'evalscript': evalscript(bands), 'width': width, 'height': height,
                'as_geotiff': True, 'save_path': str(path)}
        if max_cc is not None:
            args['max_cloud_cover'] = max_cc
        await self.invoke(stack, task_id, 'cdse', 'fetch_sentinel2', args)
        if not path.exists():
            raise RuntimeError(f'Expected download missing: {path}')
        return path

    @staticmethod
    def read_bands(path: Path, nbands: int, used: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return reflectance bands, validity mask (positive reflectance in the 1-based `used` bands) and identity."""
        import rasterio

        with rasterio.open(path) as ds:
            data = ds.read().astype('float64')
            identity = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'crs': str(ds.crs),
                        'width': ds.width, 'height': ds.height, 'bounds': list(ds.bounds)}
        bands = data[:nbands]
        if data.shape[0] >= nbands + 2:
            data_mask, scl = data[nbands] == 1, data[nbands + 1]
            identity['cloud_fraction_scl'] = float(np.isin(scl[data_mask], CLOUD_SCL).mean()) if data_mask.any() else None
        else:
            data_mask = ~np.all(bands == 0, axis=0)
        used_bands = bands if used is None else bands[[i - 1 for i in used]]
        valid = data_mask & np.all(used_bands > 0, axis=0)
        bbox = identity['bounds']
        identity['gsd_m'] = list(gsd_m([bbox[0], bbox[1], bbox[2], bbox[3]], identity['width'], identity['height'])) \
            if 'EPSG:4326' in identity['crs'] else None
        identity['masked_pixels'] = int((~valid).sum())
        return bands, valid, identity

    def write_index(self, template: Path, index: np.ndarray, valid: np.ndarray, dest: Path) -> Path:
        import rasterio

        with rasterio.open(template) as ds:
            profile = ds.profile | {'count': 1, 'dtype': 'float32', 'nodata': np.nan}
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, 'w', **profile) as out:
            out.write(np.where(valid, index, np.nan).astype('float32'), 1)
        return dest

    @staticmethod
    def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        with np.errstate(all='ignore'):
            return (nir - red) / (nir + red)

    @staticmethod
    def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        with np.errstate(all='ignore'):
            return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

    def ndvi_reference(self, task_id: str, source: Path, nir_index: int, red_index: int, nbands: int,
                       dest_name: str, thresholds: tuple[float, ...] = ()) -> tuple[dict[str, Any], dict[str, Any], Path]:
        bands, valid, identity = self.read_bands(source, nbands, (nir_index, red_index))
        index = self.ndvi(bands[nir_index - 1], bands[red_index - 1])
        valid &= np.isfinite(index)
        dest = self.write_index(source, index, valid, self.artifacts / task_id / dest_name)
        values = index[valid]
        extra = {f'pct_pixels_ndvi_gt_{t}': float(100 * (values > t).mean()) for t in thresholds}
        return stats(values), extra, dest

    def reproject_stats(self, source: Path, target_crs: str, dest: Path,
                        zero_is_nodata: bool = True) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        """Bilinear reprojection with NaN fill; source no-data (NaN, or all-zero reflectance) excluded."""
        import rasterio
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        with rasterio.open(source) as ds:
            data = ds.read().astype('float32')
            transform, width, height = calculate_default_transform(ds.crs, target_crs, ds.width, ds.height, *ds.bounds)
            profile = ds.profile | {'crs': target_crs, 'transform': transform, 'width': width, 'height': height,
                                    'dtype': 'float32', 'nodata': np.nan}
            src_crs, src_transform = ds.crs, ds.transform
        invalid = np.any(~np.isfinite(data), axis=0)
        if zero_is_nodata:  # reflectance input: no-data zeros and non-physical reflectance
            invalid |= np.all(data == 0, axis=0) | np.any(data <= 0, axis=0)
        data[:, invalid] = np.nan
        out = np.full((data.shape[0], height, width), np.nan, dtype='float32')
        for i in range(data.shape[0]):
            reproject(data[i], out[i], src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
                      dst_transform=transform, dst_crs=target_crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest, 'w', **profile) as dst:
            dst.write(out)
        valid = np.all(np.isfinite(out), axis=0)
        per_band = {i + 1: stats(out[i][valid].astype('float64')) for i in range(out.shape[0])}
        identity = {'path': str(dest.relative_to(ROOT)), 'sha256': sha256(dest), 'crs': target_crs,
                    'width': width, 'height': height, 'pixel_size_m': [transform.a, -transform.e],
                    'fill_pixels_excluded': int((~valid).sum())}
        return per_band, identity

    def imagery_policy(self, rule_extra: str = '') -> dict[str, str]:
        return {'reference_type': 'corrected_reference_observation',
                'masking_rule': 'valid = dataMask==1 (or not all-zero bands) AND all used reflectance bands > 0; '
                                'float64 index; statistics over valid pixels only' + rule_extra,
                'evaluation_policy': 'Stored values are one valid observation. Score candidates by recomputing the '
                                     'same statistics with the same masking rule on their own task-valid downloads; '
                                     'compare stored values only for identical scene, area, pixel size and mask.'}

    async def task_200009(self, stack) -> None:
        tid = self.task_id('200009')
        row = self.nominatim(tid, {'q': 'Barcelona, Spain', 'featureType': 'city'})
        lat, lon = float(row['lat']), float(row['lon'])
        bbox = [round(lon - 0.0235, 6), round(lat - 0.0175, 6), round(lon + 0.0235, 6), round(lat + 0.0175, 6)]
        bands = ['B02', 'B03', 'B04', 'B08']
        path = await self.fetch_s2(stack, tid, 'sentinel2_barcelona', bbox, bands, '2022-07-01', '2022-07-31', 20)
        data, valid, identity = self.read_bands(path, len(bands))
        band_stats = {b: stats(data[i][valid]) for i, b in enumerate(bands)}
        self.stage(tid, {'band_statistics': band_stats, 'units': 'surface reflectance (0-1)',
                         'source': {'area': {'display_name': row['display_name'], 'centre': [lon, lat], 'bbox': bbox,
                                             'area_definition': 'central Barcelona: about 4 km x 4 km box centred on the '
                                                                'geocoded city-centre point'},
                                    'raster': identity},
                         'correction': 'Previous reference geocoded a power-station building in Sant Adria de Besos '
                                       '(170 m box upsampled to 0.3 m pixels).',
                         **self.imagery_policy()})

    async def task_200010(self, stack) -> None:
        tid = self.task_id('200010')
        source = OLD_ARTIFACTS / tid / 'sentinel2_po_valley_200010.tif'
        ndvi, extra, dest = self.ndvi_reference(tid, source, 1, 2, 2, 'ndvi_po_valley.tif', (0.6,))
        _, _, identity = self.read_bands(source, 2)
        self.stage(tid, {'ndvi_stats': ndvi, **extra, 'source': {'raster': identity, 'ndvi_raster': str(dest.relative_to(ROOT))},
                         'correction': 'Previous range (-7000 to 307) came from near-zero/negative reflectance; '
                                       'same input recomputed with the masking rule.',
                         **self.imagery_policy()})

    async def task_200011(self, stack) -> None:
        tid = self.task_id('200011')
        source = OLD_ARTIFACTS / tid / 'sentinel2_alps.tif'
        per_band, identity = self.reproject_stats(source, 'EPSG:32632', self.artifacts / tid / 'sentinel2_alps_utm32n.tif')
        names = {1: 'R(B04)', 2: 'G(B03)', 3: 'B(B02)'}
        self.stage(tid, {'band_stats': {names[i]: v for i, v in per_band.items()}, 'units': 'surface reflectance (0-1)',
                         'source': {'input_raster': str(source.relative_to(ROOT)), 'input_sha256': sha256(source),
                                    'reprojected_raster': identity},
                         'correction': 'Previous statistics counted reprojection fill and mosaic no-data zeros as data.',
                         **self.imagery_policy('; reprojection fill and source no-data excluded')})

    async def task_200012(self, stack) -> None:
        tid = self.task_id('200012')
        source = OLD_RETRY_ARTIFACTS / tid / 'sentinel2_spain_clipped.tif'
        ndvi, extra, dest = self.ndvi_reference(tid, source, 4, 3, 4, 'ndvi_spain.tif', (0.5,))
        _, _, identity = self.read_bands(source, 4)
        self.stage(tid, {'ndvi_stats': ndvi, **extra, 'source': {'raster': identity, 'ndvi_raster': str(dest.relative_to(ROOT)),
                                                               'area': 'Almeria province bbox from geocoder (includes coastal sea)'},
                         'correction': 'Previous min/max (-51.5/3200) and mean were driven by zero-denominator pixels.',
                         **self.imagery_policy()})

    async def ndvi_10m(self, stack, suffix: str, query: str, months: tuple[str, str], max_cc: float | None,
                       bands: list[str], nir: int, red: int, name: str, reproject_to: str | None = None) -> None:
        tid = self.task_id(suffix)
        bbox, area = self.town_window(tid, query)
        path = await self.fetch_s2(stack, tid, name, bbox, bands, *months, max_cc)
        ndvi, _, dest = self.ndvi_reference(tid, path, nir, red, len(bands), f'ndvi_{name}.tif')
        _, _, identity = self.read_bands(path, len(bands))
        source = {'area': {**area, 'bbox': bbox}, 'raster': identity, 'ndvi_raster': str(dest.relative_to(ROOT))}
        if reproject_to:
            per_band, reprojected = self.reproject_stats(dest, reproject_to, self.artifacts / tid / f'ndvi_{name}_utm.tif',
                                                         zero_is_nodata=False)
            ndvi, source['reprojected_raster'] = per_band[1], reprojected
        self.stage(tid, {'ndvi_stats': ndvi, 'source': source,
                         'correction': 'Re-fetched at true ~10 m pixels over the town surroundings; previous reference '
                                       'used default 512x512 output (coarser than 10 m) and unmasked NDVI.',
                         **self.imagery_policy()})

    async def task_400026(self, stack) -> None:
        """Whole Niger park polygon at ~10 m, fetched in tiles under the 2500 px limit."""
        import rasterio
        from rasterio.features import geometry_mask

        tid = self.task_id('400026')
        row = self.nominatim(tid, {'q': 'W National Park, Niger', 'polygon_geojson': 1})
        polygon = row['geojson']
        south, north, west, east = map(float, row['boundingbox'])
        width, height = size_for_gsd([west, south, east, north])
        nx, ny = math.ceil(width / MAX_PX * 1.05), math.ceil(height / MAX_PX * 1.05)
        xs, ys = np.linspace(west, east, nx + 1), np.linspace(south, north, ny + 1)
        bands = ['B02', 'B04', 'B08']
        running, tiles, cloud, park_pixels = RunningStats(), [], [], 0
        for j in range(ny):
            for i in range(nx):
                tile = [round(xs[i], 6), round(ys[j], 6), round(xs[i + 1], 6), round(ys[j + 1], 6)]
                path = await self.fetch_s2(stack, tid, f'sentinel2_w_park_tile_{j}_{i}', tile, bands,
                                           '2023-07-01', '2023-07-31', 20)
                data, valid, identity = self.read_bands(path, 3)
                with rasterio.open(path) as ds:
                    inside = ~geometry_mask([polygon], out_shape=(ds.height, ds.width), transform=ds.transform)
                index = self.evi(data[2], data[1], data[0])
                use = valid & inside & np.isfinite(index) & (np.abs(index) <= 1)
                park_pixels += int(inside.sum())
                self.write_index(path, index, use, self.artifacts / tid / f'evi_w_park_tile_{j}_{i}.tif')
                running.add(index[use])
                if identity.get('cloud_fraction_scl') is not None and inside.any():
                    cloud.append((int(inside.sum()), identity['cloud_fraction_scl']))
                tiles.append(identity)
        evi_stats = running.result()
        self.stage(tid, {'evi_stats': evi_stats, 'formula': '2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)',
                         'source': {'area': {'display_name': row['display_name'], 'osm_type': row['osm_type'],
                                             'osm_id': row['osm_id'], 'area_definition': 'OSM park boundary polygon'},
                                    'tiles': tiles, 'park_pixels': park_pixels,
                                    'valid_fraction_of_park': evi_stats['valid_pixels'] / park_pixels,
                                    'approx_cloud_fraction_scl_tile_weighted':
                                        sum(n * c for n, c in cloud) / sum(n for n, _ in cloud) if cloud else None},
                         'correction': 'Previous reference used ~150 m pixels over the bbox; now ~10 m over the park polygon.',
                         **self.imagery_policy('; restricted to the park polygon; EVI outside [-1, 1] excluded as non-physical; '
                                               'no cloud masking (not requested)')})

    # ---- EUMETSAT ----------------------------------------------------------
    async def task_700017(self, stack) -> None:
        """First match = chronologically first RSS slot in the reference 11:00-13:00 search window."""
        tid = self.task_id('700017')
        bbox = [-18.5, 27.5, -13.0, 29.5]
        search = await self.invoke(stack, tid, 'eumetsat', 'search_eumetsat_products', {
            'product': 'MSG-SEVIRI-RSS', 'start_date': '2024-07-05T11:00:00', 'end_date': '2024-07-05T13:00:00', 'bbox': bbox})
        # The listing is truncated to the latest slots, so confirm the first slot with a narrow search.
        opening = await self.invoke(stack, tid, 'eumetsat', 'search_eumetsat_products', {
            'product': 'MSG-SEVIRI-RSS', 'start_date': '2024-07-05T11:00:00', 'end_date': '2024-07-05T11:05:00', 'bbox': bbox})
        first = min((p for p in opening['products'] if p['date'] >= '2024-07-05T11:00'), key=lambda p: p['date'])
        start = first['date'].split('/')[0]
        hour = int(start[11:13])
        download = await self.invoke(stack, tid, 'eumetsat', 'download_eumetsat_product', {'args': {
            'product': 'MSG-SEVIRI-RSS', 'start_date': '2024-07-05T00:00:00', 'end_date': '2024-07-05T23:59:00',
            'bbox': bbox, 'method': 'datastore', 'start_hour': hour, 'interval_minutes': 5}})
        match = next((p for p in download['products'] if p['identifier'] == first['identifier']), None)
        if match is None or not match['paths']:
            raise RuntimeError(f'First search match {first["identifier"]} was not downloaded: {download}')
        read = await self.invoke(stack, tid, 'eumetsat', 'read_eumetsat_product', {'args': {
            'product': 'MSG-SEVIRI-RSS', 'paths': match['paths'], 'bands': ['HRV'], 'bbox': bbox}})
        self.stage(tid, {'band_statistics': {'HRV': read['bands']['HRV']}, 'units': 'percent reflectance',
                         'reference_type': 'corrected_reference_observation',
                         'selection_scope': {'search_window': '2024-07-05T11:00/13:00 UTC', 'search_count': search['count'],
                                             'search_listing_truncated': search.get('truncated'), 'selected': first,
                                             'rule': 'chronologically first slot in the search window (confirmed by an 11:00-11:05 search)',
                                             'crop': 'read_eumetsat_product bbox crop (tool-consistent)'},
                         'source': {'paths': [str(Path(p)) for p in match['paths']]},
                         'correction': 'Previous reference downloaded 13:00/13:05 slots outside the searched window '
                                       '(solar-noon default) and mixed identifiers.',
                         'evaluation_policy': 'Candidates that read the first item of the search listing (descending, '
                                              '12:55 slot) made a defensible choice; recompute on their own file.'})

    # ---- ERA5 --------------------------------------------------------------
    @staticmethod
    def regional_mean(path: Path, variable: str, reduce: str) -> tuple[float, dict[str, Any]]:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            da = ds[variable]
            time_dim = next((d for d in ('valid_time', 'time', 'date') if d in da.dims), None)
            lat, lon = ('latitude', 'longitude') if 'latitude' in da.dims else ('lat', 'lon')
            per_cell = (da.sum(time_dim, min_count=1) if reduce == 'sum' else da.mean(time_dim)) if time_dim else da
            value = float(per_cell.mean([lat, lon], skipna=True))
            support = {'cells': int(np.isfinite(per_cell).sum()), 'time_steps': int(da.sizes.get(time_dim, 1)),
                       'grid_cell_centres': {'west': float(da[lon].min()), 'east': float(da[lon].max()),
                                             'south': float(da[lat].min()), 'north': float(da[lat].max())},
                       'units': da.attrs.get('units'), 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
        return value, support

    def era5_path(self, relative: str) -> Path:
        return self.artifacts / 'era5' / relative.removeprefix('output/')

    def era5_policy(self) -> dict[str, str]:
        return {'reference_type': 'replayed_reference_workflow',
                'evaluation_policy': 'Value holds for the region the reference workflow resolves. Candidates that '
                                     'downloaded a different defensible region must be recomputed on their own file.'}

    async def named_region_mean(self, stack, suffix: str, region: str, fetch_tool: str, fetch_args: dict[str, Any],
                                variable: str, reduce: str, units: str) -> None:
        tid = self.task_id(suffix)
        resolved = await self.invoke(stack, tid, 'cdse', 'resolve_named_region', {'region_name': region})
        west, south, east, north = resolved['bbox_array']
        relative = f'output/{tid}/{variable}_{suffix}.nc'
        await self.invoke(stack, tid, 'climate', fetch_tool, {**fetch_args, 'area': [north, west, south, east],
                                                              'output_filename': relative})
        value, support = self.regional_mean(self.era5_path(relative), variable, reduce)
        self.stage(tid, {'regional_mean': value, 'units': units, 'source': {'region': resolved, **support},
                         'aggregation': f'per-cell temporal {reduce}, then unweighted mean over finite cells',
                         'correction': f'Previous reference came from an older file whose region did not match '
                                       f'resolve_named_region("{region}").', **self.era5_policy()})

    async def task_600073(self, stack) -> None:
        tid = self.task_id('600073')
        geocode = await self.invoke(stack, tid, 'cdse', 'geocode_place', {'place_name': 'Death Valley, USA'})
        west, south, east, north = geocode['top_result']['bbox_array']
        global_file, clipped = f'output/{tid}/t2m_global_2023_07.nc', f'output/{tid}/t2m_death_valley_2023_07.nc'
        await self.invoke(stack, tid, 'climate', 'fetch_era5_single_levels',
                          {'variable': '2m_temperature', 'year': '2023', 'month': '07', 'output_filename': global_file})
        await self.invoke(stack, tid, 'climate', 'clip_to_bbox', {'input_filepath': str(self.era5_path(global_file)), 'north': north, 'south': south,
                                                                  'east': east, 'west': west, 'output_filepath': clipped})
        value, support = self.regional_mean(self.era5_path(clipped), 't2m', 'mean')
        self.stage(tid, {'regional_mean': value - 273.15, 'units': 'degC',
                         'source': {'geocode': geocode['top_result'], **support},
                         'correction': 'Previous clipped region (-118..-116.5, 35.5..37.25) was not derivable from the '
                                       'geocoder bbox used by the reference workflow.', **self.era5_policy()})

    async def task_600081(self, stack) -> None:
        """Continental Africa (mainland + Madagascar) extreme points; the geocoder has no usable Africa bbox."""
        tid = self.task_id('600081')
        bbox = {'west': -17.54, 'south': -34.84, 'east': 51.42, 'north': 37.35}
        global_file, clipped = f'output/{tid}/tp_global_2023_01.nc', f'output/{tid}/tp_africa_2023_01.nc'
        await self.invoke(stack, tid, 'climate', 'fetch_era5_single_levels',
                          {'variable': 'total_precipitation', 'year': '2023', 'month': '01', 'output_filename': global_file})
        await self.invoke(stack, tid, 'climate', 'clip_to_bbox', {'input_filepath': str(self.era5_path(global_file)), **bbox,
                                                                  'output_filepath': clipped})
        _, support = self.regional_mean(self.era5_path(clipped), 'tp', 'mean')
        self.stage(tid, {'extent': support['grid_cell_centres'], 'units': 'degrees',
                         'source': {'region_definition': 'Mainland Africa + Madagascar extreme points: Cap-Vert '
                                                         '(17.54W), Cape Agulhas (34.84S), Ras Hafun (51.42E), '
                                                         'Ras ben Sakka (37.35N)', 'clip_bbox': bbox, **support},
                         'correction': 'Previous padded bbox (-20/-35/55/38) was not derivable from the workflow; '
                                       'geocode_place("Africa") returns an unusable centroid box.',
                         'reference_type': 'documented_region_definition',
                         'evaluation_policy': 'Region convention: accept extents within the configured tolerance of '
                                              'the continental extreme-point extent.'})

    # ---- sets --------------------------------------------------------------
    async def dashboard_sets(self, stack) -> None:
        response = requests.get(NARRATIVES, timeout=30)
        response.raise_for_status()
        narratives = response.json()
        for suffix, code in (('100006', 'FNF_Palsar'), ('100010', 'SIC_sea_ice_thickness_cryosat')):
            tid = self.task_id(suffix)
            self.record(tid, 'source_snapshot', {'url': NARRATIVES, 'narratives': len(narratives)})
            titles = [n['title'].strip() for n in narratives
                      if code.upper() in [str(c).upper() for c in n.get('collections') or []]]
            self.stage(tid, {'reference_set': titles, 'reference_count': len(titles), 'indicator_code': code,
                             'reference_type': 'source_snapshot',
                             'evaluation_policy': 'Compare trimmed narrative titles against this snapshot.'})

    async def ontology_sets(self, stack) -> None:
        for suffix in ('500008', '500013'):
            tid = self.task_id(suffix)
            cypher = self.tasks[tid]['reference_solutions'][0]['steps'][-1]['arguments']['cypher']
            rows = await self.invoke(stack, tid, 'eoss', 'query_ontology', {'cypher': cypher})
            if suffix == '500008':
                self.stage(tid, {'reference_set': [{'id': r['id'], 'aliases': [r['name']]} for r in rows],
                                 'reference_count': len(rows), 'reference_type': 'source_snapshot',
                                 'evaluation_policy': 'Explicit HAS_KAO edges in this snapshot.'})
            else:
                self.stage(tid, {'kaos_per_question': {r['question']: int(r['n_kaos']) for r in rows},
                                 'total_kaos': sum(int(r['n_kaos']) for r in rows), 'reference_type': 'source_snapshot',
                                 'evaluation_policy': 'Explicit HAS_KAO edges in this snapshot; every ScienceQuestion '
                                                      'node must be listed.'})

    async def task_600011(self, stack) -> None:
        tid = self.task_id('600011')
        catalogue = await self.invoke(stack, tid, 'climate', 'list_cds_datasets', {})
        text = catalogue if isinstance(catalogue, str) else json.dumps(catalogue)
        names = [n for n in ('reanalysis-era5-single-levels', 'reanalysis-era5-single-levels-monthly-means',
                             'reanalysis-era5-pressure-levels', 'reanalysis-era5-pressure-levels-monthly-means',
                             'reanalysis-era5-land', 'reanalysis-era5-land-monthly-means',
                             'derived-era5-single-levels-daily-statistics', 'derived-era5-pressure-levels-daily-statistics')
                 if n in text]
        if len(names) != 8:
            raise RuntimeError(f'Catalogue does not list all expected ERA5 datasets: {names}')
        self.stage(tid, {'reference_set': names, 'reference_count': len(names),
                         'accepted_optional': ['sis-agrometeorological-indicators'],
                         'reference_type': 'source_snapshot',
                         'correction': 'Previous set omitted the two derived ERA5 daily-statistics datasets.',
                         'evaluation_policy': 'Optional item is ERA5-derived (AgERA5); do not penalise it either way.'})

    # ---- driver ------------------------------------------------------------
    def jobs(self) -> dict[str, Callable]:
        return {
            '100006': self.dashboard_sets, '100010': self.dashboard_sets,
            '500008': self.ontology_sets, '500013': self.ontology_sets,
            '600011': self.task_600011,
            '200010': self.task_200010, '200011': self.task_200011, '200012': self.task_200012,
            '200009': self.task_200009,
            '200013': lambda s: self.ndvi_10m(s, '200013', 'Bila Tserkva, Kyiv Oblast, Ukraine', ('2023-06-01', '2023-06-30'),
                                              20, ['B04', 'B08'], 2, 1, 'sentinel2_bila_tserkva'),
            '200035': lambda s: self.ndvi_10m(s, '200035', 'Yahotyn, Kyiv Oblast, Ukraine', ('2023-06-01', '2023-06-30'),
                                              20, ['B08', 'B04'], 1, 2, 'sentinel2_yahotyn', reproject_to='EPSG:32636'),
            '200036': lambda s: self.ndvi_10m(s, '200036', 'Mopti, Mali', ('2023-08-01', '2023-08-31'),
                                              30, ['B08', 'B04'], 1, 2, 'sentinel2_mopti'),
            '400025': lambda s: self.ndvi_10m(s, '400025', 'Lodi, Lombardy, Italy', ('2023-06-01', '2023-08-31'),
                                              20, ['B08', 'B04'], 1, 2, 'sentinel2_lodi'),
            '400026': self.task_400026,
            '700017': self.task_700017,
            '600063': lambda s: self.named_region_mean(s, '600063', 'Sahel', 'fetch_era5_daily_statistics',
                                                       {'variable': 'total_precipitation', 'statistic': 'daily_sum',
                                                        'year': '2022', 'month': '07'}, 'tp', 'sum', 'm'),
            '600070': lambda s: self.named_region_mean(s, '600070', 'Mediterranean', 'fetch_era5_hourly_single_levels',
                                                       {'variable': 'total_precipitation', 'year': '2022', 'month': '01'},
                                                       'tp', 'sum', 'm'),
            '600073': self.task_600073,
            '600081': self.task_600081,
        }

    async def run(self, suffixes: list[str]) -> None:
        done_groups: set[Callable] = set()
        async with AsyncExitStack() as stack:
            for suffix, job in self.jobs().items():
                if suffixes and suffix not in suffixes:
                    continue
                if self.task_id(suffix) in self.references or job in done_groups:
                    continue
                done_groups.add(job)
                print(f'[{now()}] running {suffix}', flush=True)
                try:
                    await job(stack)
                except Exception as error:  # keep going; failures are reported, never staged
                    self.errors[self.task_id(suffix)] = f'{type(error).__name__}: {error}'
                    print(f'[{now()}] FAILED {suffix}: {self.errors[self.task_id(suffix)]}', flush=True)
                self.save('errors.json', self.errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/reference_corrections_20261012')
    parser.add_argument('--task-suffixes', nargs='+', default=[])
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    correction = ReferenceCorrection(Config(args.output_dir))
    asyncio.run(correction.run(args.task_suffixes))
    print(f'[{now()}] staged={len(correction.references)} failed={len(correction.errors)}', flush=True)
    sys.exit(1 if correction.errors else 0)


if __name__ == '__main__':
    main()
