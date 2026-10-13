"""Recompute reference observations from identified local NetCDF snapshots.

Run from the repository root. Different task-valid regions or inputs require
their own runtime recomputation; these observations are not universal keys.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class ReferenceInput:
    """An identified input, its expected time coverage and reduction."""
    task_id: str
    path: Path
    variable: str
    month: str
    expected_samples: int
    operation: str
    bbox: tuple[float, float, float, float] | None = None


class NumericalReferenceGenerator:
    """Validate sources and persist requested numeric outputs with provenance."""

    def __init__(self, inventory_path: Path, extraction_dir: Path | None = None) -> None:
        self.inventory_path = inventory_path
        self.extraction_dir = extraction_dir or Path('output/ground_truth/extracted_inputs')
        self.results: dict[str, dict] = {}
        self.failures: dict[str, str] = {}

    @staticmethod
    def source_identity(path: Path) -> dict:
        """Bind each observation to the exact inspected bytes."""
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'path': str(path), 'sha256': digest}

    def calculate(self, source: ReferenceInput) -> dict:
        """Aggregate time first, then take the unweighted valid-cell mean."""
        prepared = source.path
        member = None
        if zipfile.is_zipfile(source.path):
            with zipfile.ZipFile(source.path) as archive:
                members = [name for name in archive.namelist() if name.endswith('.nc')]
                if len(members) != 1:
                    raise ValueError('Archive needs an explicit unique NetCDF member selection')
                member = members[0]
                directory = self.extraction_dir
                directory.mkdir(parents=True, exist_ok=True)
                prepared = directory / (self.source_identity(source.path)['sha256'] + '.nc')
                if not prepared.exists():
                    with archive.open(member) as input_stream, prepared.open('wb') as output_stream:
                        shutil.copyfileobj(input_stream, output_stream)
        with xr.open_dataset(prepared, engine='netcdf4') as dataset:
            array = dataset[source.variable]
            if source.variable == 'z' and ('pressure_level' not in dataset or
                    not np.all(dataset['pressure_level'].values == 500)):
                raise ValueError('Expected geopotential at 500 hPa')
            time = next(k for k in ('valid_time', 'time') if k in array.dims)
            dates = np.asarray(dataset[time].values, dtype='datetime64[ns]')
            if len(dates) != source.expected_samples or len(np.unique(dates)) != len(dates):
                raise ValueError('Missing or duplicate expected time samples')
            if not np.all(dates.astype('datetime64[M]') == np.datetime64(source.month)):
                raise ValueError('Input time coverage differs from the requested month')
            if source.expected_samples > 1:
                step = {'daily_sum': 1, 'monthly_daily_sum': 24, 'monthly_daily_mean': 24,
                        'daily_mean': 6, 'monthly_hourly_sum': 1, 'monthly_hourly_mean': 1}[source.operation]
                if not np.all(np.diff(dates) == np.timedelta64(step, 'h')):
                    raise ValueError('Irregular or incomplete time sampling')
                if str(dates[0])[:10] != source.month + '-01' or str(dates[-1])[:10] != source.month + '-31':
                    raise ValueError('Expected complete 31-day month')
            latitude, longitude = dataset['latitude'], dataset['longitude']
            normalized_lon = ((longitude + 180) % 360) - 180
            array = array.assign_coords(longitude=normalized_lon)
            if source.bbox:
                west, south, east, north = source.bbox
                array = array.where((latitude >= south) & (latitude <= north), drop=True)
                # Use the assigned longitude index: using the pre-normalization
                # coordinate would align 0..360 against -180..180 and drop west.
                array = array.where((array.longitude >= west) & (array.longitude <= east), drop=True)
            if not array.sizes['latitude'] or not array.sizes['longitude']:
                raise ValueError('No grid-cell centres in the selected reference region')
            units = array.attrs.get('units')
            if source.variable == 'tp' and units != 'm':
                raise ValueError('Precipitation input must be metres per original accumulation')
            if source.variable == 't2m' and units != 'K':
                raise ValueError('Temperature input must be Kelvin')
            if source.variable == 'msl' and units != 'Pa':
                raise ValueError('Pressure input must be Pa')
            if source.operation == 'daily_sum':
                reduced = array.resample({time: '1D'}).sum(skipna=True, min_count=24)
            elif source.operation == 'daily_mean':
                valid = array.resample({time: '1D'}).count()
                reduced = array.resample({time: '1D'}).mean(skipna=True).where(valid == 4)
            elif source.operation in ('monthly_daily_sum', 'monthly_hourly_sum'):
                reduced = array.sum(time, skipna=True, min_count=source.expected_samples)
            elif source.operation == 'monthly_daily_mean':
                reduced = array.mean(time, skipna=True).where(array.count(time) == source.expected_samples)
            elif source.operation == 'monthly_hourly_mean':
                valid = array.resample({time: '1D'}).count()
                daily = array.resample({time: '1D'}).mean(skipna=True).where(valid == 24)
                reduced = daily.mean(time, skipna=True).where(daily.count(time) == 31)
            elif source.operation == 'monthly_celsius':
                reduced = array - 273.15
                units = 'degC'
            elif source.operation == 'extent':
                reduced = array
            else:
                raise ValueError('Unsupported reference operation')
            actual_extent = {k: float(v) for k, v in zip(('west','south','east','north'),
                (array.longitude.min(), array.latitude.min(), array.longitude.max(), array.latitude.max()))}
            if source.operation == 'extent':
                fields = {'extent': actual_extent}
                units = 'degrees'
            else:
                means = reduced.mean(('latitude', 'longitude'), skipna=True).squeeze()
                if not np.isfinite(means.values).all():
                    raise ValueError('Nonfinite regional reference')
                if source.operation.startswith('daily_'):
                    fields = {'daily_values': {str(t)[:10]: float(v) for t, v in zip(means[time].values, means.values)}}
                else:
                    fields = {'regional_mean': float(means.item())}
            return dict(fields, task_ID=source.task_id, units=units,
                reference_type='observed_valid_input_snapshot',
                provenance={'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                    'source': self.source_identity(source.path), 'variable': source.variable,
                    'archive_member': member, 'dataset_source': self.source_identity(prepared),
                    'month': source.month, 'original_samples': source.expected_samples,
                    'input_variable_attributes': dict(dataset[source.variable].attrs),
                    'operation': source.operation, 'selected_grid_cell_centres': actual_extent,
                    'aggregation': 'temporal reduction per cell, followed by unweighted mean over finite cells'},
                evaluation_policy='Use only for matching verified source, spatial support, units and mask. '
                    'Recompute from candidate-owned task-valid inputs for other permitted choices; '
                    'do not impose this reference bbox as a question requirement.')

    def run(self, sources: list[ReferenceInput]) -> None:
        """Persist successful references and explicit diagnostics for failures."""
        for source in sources:
            try:
                self.results[source.task_id] = self.calculate(source)
                print(source.task_id, 'computed', flush=True)
            except (OSError, ValueError, KeyError, StopIteration) as error:
                self.failures[source.task_id] = str(error)
                print(source.task_id, 'unavailable:', error, flush=True)
        inventory = json.loads(self.inventory_path.read_text())
        inventory.update(self.results)
        self.inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False,
            default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()) + '\n')
        report_path = Path('output/ground_truth/netcdf_reference_report.json')
        previous = json.loads(report_path.read_text()) if report_path.exists() else {'computed': [], 'failed': {}}
        failures = dict(previous['failed'], **self.failures)
        for task_id in self.results:
            failures.pop(task_id, None)
        report_path.write_text(json.dumps({'computed': sorted(set(previous['computed']) | set(self.results)),
                                          'failed': failures}, indent=2) + '\n')


def default_sources() -> list[ReferenceInput]:
    """Return the identified input inventory shared by regeneration and discovery."""
    return [
        ReferenceInput('TASK_260713_000053_600053', Path('output/era5/precip/era5_hourly_precip_sicily_2021_08.nc'), 'tp', '2021-08', 744, 'daily_sum'),
        ReferenceInput('TASK_260713_000061_600061', Path('output/ground_truth/city_climate/600061.nc'), 'tp', '2022-08', 744, 'monthly_hourly_sum'),
        ReferenceInput('TASK_260713_000062_600062', Path('output/ground_truth/city_climate/600062.nc'), 't2m', '2022-07', 744, 'monthly_hourly_mean'),
        ReferenceInput('TASK_260713_000063_600063', Path('output/era5/data/sahel_precip_daily_2022_07.nc'), 'tp', '2022-07', 31, 'monthly_daily_sum'),
        ReferenceInput('TASK_260713_000067_600067', Path('output/era5/malta_temperature/malta_2m_temp_daily_2023_07.nc'), 't2m', '2023-07', 31, 'monthly_daily_mean'),
        ReferenceInput('TASK_260713_000068_600068', Path('output/era5/mslp_6hourly_azores_2023_01.nc'), 'msl', '2023-01', 124, 'daily_mean'),
        ReferenceInput('TASK_260713_000070_600070', Path('output/era5/mediterranean_precip_2022_01.nc'), 'tp', '2022-01', 744, 'monthly_hourly_sum'),
        ReferenceInput('TASK_260713_000073_600073', Path('output/era5/data/death_valley_t2m_monthly_2023_07_clipped.nc'), 't2m', '2023-07', 1, 'monthly_celsius'),
        ReferenceInput('TASK_260713_000081_600081', Path('output/era5/precip_africa_2023_01.nc'), 'tp', '2023-01', 1, 'extent', (-20., -35., 55., 38.)),
        ReferenceInput('TASK_260713_000082_600082', Path('output/era5/global_2m_temp_2022_07.nc'), 't2m', '2022-07', 1, 'extent', (24., -5., 40., 32.)),
        ReferenceInput('TASK_260713_000083_600083', Path('output/era5/data/500hPa_geopotential_2023_01.nc'), 'z', '2023-01', 1, 'extent', (60., 5., 100., 35.)),
    ]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discover-missing', action='store_true',
                        help='Discover proposed references for excluded tasks without modifying canonical GT')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--complete-references', action='store_true',
                        help='Compute geometry/count references and replay imagery reference workflows')
    parser.add_argument('--skip-imagery', action='store_true')
    parser.add_argument('--task-suffixes', nargs='+', default=[], help='Limit reference completion to these task suffixes')
    parser.add_argument('--export-reference-review', action='store_true', help='Join staged references and candidate answer traces into task-ID packets')
    parser.add_argument('--prepare-evaluation', action='store_true', help='Stage combined corrected references and audit evaluation readiness')
    args = parser.parse_args()
    if args.prepare_evaluation:
        if args.output_dir is None:
            parser.error('--prepare-evaluation requires --output-dir')
        from reference_completion import EvaluationPreparation

        EvaluationPreparation(args.output_dir).run()
    elif args.export_reference_review:
        if args.output_dir is None:
            parser.error('--export-reference-review requires --output-dir')
        from reference_completion import ReferenceReviewExporter

        ReferenceReviewExporter(args.output_dir).run()
    elif args.complete_references:
        if args.output_dir is None:
            parser.error('--complete-references requires --output-dir')
        from reference_completion import CompletionConfig, ReferenceCompletion

        ReferenceCompletion(CompletionConfig(args.output_dir, replay_imagery=not args.skip_imagery,
                                            task_suffixes=tuple(args.task_suffixes))).run()
    elif args.discover_missing:
        if args.output_dir is None:
            parser.error('--discover-missing requires --output-dir')
        from ground_truth_discovery import DiscoveryConfig, GroundTruthDiscovery

        GroundTruthDiscovery(DiscoveryConfig(args.output_dir)).run()
    else:
        NumericalReferenceGenerator(Path('data/tasks/ground_truths.json')).run(default_sources())
