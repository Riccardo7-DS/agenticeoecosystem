"""Reproduce numerical reference workflows and prepare evidence-aware AAJ routing."""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from pyproj import Geod, Transformer
import requests
from shapely import wkt
from shapely.ops import transform

from ground_truth_discovery import DiscoveryConfig, GroundTruthDiscovery, ROOT

GEOMETRY = set('200014 200016 200017 300014 400001 400002 400003 400015 400016 400019 400020'.split())
COUNTS = set('500011 500013 500028'.split())
IMAGERY = set('200009 200010 200011 200012 200013 200035 200036 400010 400025 400026 700016 700017 700018 700019'.split())


def decode_result(result: dict[str, Any]) -> Any:
    """Decode structured MCP text without treating service errors as numerical GT."""
    if result.get('isError'):
        raise ValueError('MCP tool returned an error')
    structured = result.get('structuredContent')
    if isinstance(structured, dict):
        structured = structured.get('result', structured)
        if isinstance(structured, dict):
            if 'error' in structured:
                raise ValueError('Service returned an error')
            return structured
        if isinstance(structured, str):
            try:
                return json.loads(structured)
            except json.JSONDecodeError:
                pass
    texts = [block['text'] for block in result.get('content', []) if block.get('type') == 'text']
    if len(texts) == 1:
        try:
            value = json.loads(texts[0])
        except json.JSONDecodeError:
            if re.search(r'\b(error|failed|exception)\b', texts[0], re.I):
                raise ValueError('Tool response indicates failure')
            return texts[0]
        if isinstance(value, dict) and 'error' in value:
            raise ValueError('Service returned an error')
        return value
    return result


@dataclass(frozen=True)
class CompletionConfig:
    """Isolated output location and bounded per-tool execution."""
    output_dir: Path
    replay_imagery: bool = True
    tool_timeout: int = 600
    task_suffixes: tuple[str, ...] = ()


class ReferenceCompletion(GroundTruthDiscovery):
    """Compute independent geometry references and execute curated tool calls."""

    def __init__(self, config: CompletionConfig) -> None:
        super().__init__(DiscoveryConfig(config.output_dir))
        self.completion = config
        self.references: dict[str, dict[str, Any]] = {}
        self.geocodes: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, Any] = {}
        self.tool_schemas: dict[str, dict[str, Any]] = {}
        self.journal: dict[str, list[dict[str, Any]]] = {}

    def geocode(self, query: str) -> dict[str, Any]:
        """Capture the same Nominatim source used by geocode_place, retaining all matches."""
        if query not in self.geocodes:
            time.sleep(1.1)  # Respect the public service's request-rate limit.
            url = 'https://nominatim.openstreetmap.org/search'
            search_query = {'central Addis Ababa, Ethiopia': 'Addis Ababa, Ethiopia',
                            'Chernobyl Nuclear Power Plant, Ukraine': 'Chornobyl Nuclear Power Plant'}.get(query, query)
            response = requests.get(url, params={'q': search_query, 'format': 'json', 'limit': '5', 'addressdetails': '1'},
                headers={'User-Agent': 'PhiLiveBench-reference-verification/1.0'}, timeout=30)
            response.raise_for_status()
            results = response.json()
            if not isinstance(results, list) or not results:
                raise ValueError('No geocoding results')
            provenance = self.snapshot('geocode_' + str(len(self.geocodes)), results,
                {'url': url, 'query': search_query, 'original_query': query, 'source_type': 'live_geocoder_snapshot'})
            selected = results[0]
            self.geocodes[query] = {'lat': float(selected['lat']), 'lon': float(selected['lon']),
                'display_name': selected['display_name'], 'osm_type': selected.get('osm_type'),
                'osm_id': selected.get('osm_id'), 'matches': len(results), 'provenance': provenance}
            self.save('reference_locations.json', self.geocodes)
        return self.geocodes[query]

    @staticmethod
    def projected(point: dict[str, Any], target: str) -> dict[str, Any]:
        """Use longitude/latitude input order regardless of CRS axis metadata."""
        x, y = Transformer.from_crs('EPSG:4326', target, always_xy=True).transform(point['lon'], point['lat'])
        return {'x': x, 'y': y, 'target_crs': target}

    @staticmethod
    def utm(point: dict[str, Any]) -> str:
        """Select ordinary UTM zone for the non-polar locations in this task group."""
        zone = min(60, max(1, math.floor((point['lon'] + 180) / 6) + 1))
        return f'EPSG:{(32600 if point["lat"] >= 0 else 32700) + zone}'

    def geometry(self, task_id: str) -> dict[str, Any]:
        """Resolve reference locations and independently calculate the requested quantity."""
        task = self.tasks[task_id]
        suffix = task_id.rsplit('_', 1)[-1]
        steps = task['reference_solutions'][0]['steps']
        points = [self.geocode(step['arguments']['place_name']) for step in steps if step['function_name'] == 'geocode_place']
        fixed = suffix in {'200017', '400015', '400020'}
        reference: dict[str, Any] = {'task_ID': task_id, 'task_text': task['task_text'],
            'reference_type': 'fixed_geometry' if fixed else 'geocoded_reference_observation',
            'status': 'computed_reference', 'reference_locations': points,
            'provenance': {'computed_at_utc': datetime.now(timezone.utc).isoformat(),
                           'implementation': 'pyproj always_xy=True; WGS84 ellipsoid; Shapely buffer',
                           'reference_steps': steps},
            'evaluation_policy': 'Fixed mathematical inputs are comparable directly. Geocoded references require entity validation; accept valid alternative points and recompute dependent quantities.'}
        if suffix in {'300014', '400001', '400002', '400003'}:
            reference.update(lat=points[0]['lat'], lon=points[0]['lon'])
        elif suffix in {'200014', '400015', '400016'}:
            step = next(step for step in steps if step['function_name'] == 'transform_coordinates')
            if suffix == '400015':
                lon, lat = step['arguments']['coordinates']
                point = {'lon': lon, 'lat': lat}
            else:
                point = points[0]
            target = self.utm(point) if suffix == '400016' else step['arguments']['target_crs']
            reference['transform'] = self.projected(point, target)
            if suffix == '400016':
                reference['utm_crs'] = {'epsg': target, 'zone': int(target[-2:]), 'hemisphere': 'N'}
        elif suffix == '200016':
            distance = abs(Geod(ellps='WGS84').inv(points[0]['lon'], points[0]['lat'], points[1]['lon'], points[1]['lat'])[2])
            reference.update(distance_m=distance, distance_km=distance / 1000)
        else:
            if suffix == '400020':
                polygon = wkt.loads(steps[0]['arguments']['geometry'])
            else:
                step = next(step for step in steps if step['function_name'] == 'buffer')
                if suffix == '200017':
                    centre, crs = wkt.loads(step['arguments']['geometry']), 'EPSG:32722'
                else:
                    crs = self.utm(points[0])
                    xy = self.projected(points[0], crs)
                    centre = wkt.loads(f'POINT({xy["x"]} {xy["y"]})')
                buffered = centre.buffer(step['arguments']['distance'], quad_segs=step['arguments']['resolution'])
                polygon = transform(Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform, buffered)
            area, perimeter = Geod(ellps='WGS84').geometry_area_perimeter(polygon)
            reference.update(area_m2=abs(area), area_km2=abs(area) / 1e6, perimeter_m=perimeter)
            path = self.config.output_dir / 'sources' / f'{task_id}.wkt'
            path.write_text(polygon.wkt)
            reference['provenance'].update(geometry_path=str(path.resolve()), geometry_sha256=self.sha(path))
        return reference

    def prepare_profile(self) -> None:
        """Route artifact and retrieval outcomes through AAJ without removing other checks."""
        payload = json.loads((ROOT / 'data/tasks/task_evaluation_strategies.json').read_text())
        changes = []
        for strategy in payload['tasks']:
            if strategy['task_ID'].endswith('_200005'):
                changes.append(strategy['task_ID'])
                continue
            if strategy['evaluation_strategy'] in {'download_contract', 'retrieval_validation'} or strategy['task_ID'].endswith('_200020'):
                strategy['evaluation_methods'] = list(dict.fromkeys([*strategy['evaluation_methods'], 'agent_as_judge']))
                strategy['outcome_evaluator'] = 'tool_enabled_agent_as_judge'
                strategy['aaj_evidence_policy'] = ('Inspect candidate-owned artifacts and source responses. Verify task-specific spatial/temporal/variable constraints. '
                    'Use deterministic measurements as evidence; unavailable evidence is unverified, not success.')
                changes.append(strategy['task_ID'])
        self.save('evaluation_strategies_aaj.json', payload)
        self.save('aaj_routing.json', {'tasks': changes, 'count': len(changes),
            'policy': 'New evaluation profile only; no judge calls or published score changes'})

    def server_configuration(self, name: str) -> dict[str, Any]:
        """Configure only the servers used by these reference workflows."""
        paths = {'cdse': 'cdse_geocode/server.py', 'eumetsat': 'eumetsat/server.py', 'gee': 'gee/server.py',
                 'dashboard': 'EOdashboard/eodashboard_server.py'}
        if name == 'eoss':
            url, auth = os.getenv('GRAPHRAG_MCP_URL'), os.getenv('GRAPHRAG_AUTH_HEADER')
            if not url or not auth:
                raise ValueError('EOSS credentials unavailable')
            return {'transport': 'streamable_http', 'url': url, 'headers': {'Authorization': auth}}
        if name == 'gis':
            return {'transport': 'stdio', 'command': str(Path.home() / '.local/bin/uv'),
                    'args': ['tool', 'run', '--python', '3.12', 'gis-mcp'],
                    'env': dict(os.environ),
                    'cwd': str(self.config.output_dir.resolve() / 'artifacts')}
        return {'transport': 'stdio', 'command': sys.executable,
                'env': dict(os.environ),
                'args': [str(ROOT / 'src/agenticeoecosystem/servers' / paths[name])]}

    async def invoke(self, stack: AsyncExitStack, server: str, task_id: str, name: str, args: dict[str, Any]) -> Any:
        """Persist actual arguments/results and refuse tool errors as reference values."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        if server not in self.sessions:
            client = MultiServerMCPClient({server: self.server_configuration(server)})
            self.sessions[server] = await stack.enter_async_context(client.session(server))
            schema = await self.sessions[server].list_tools()
            self.tool_schemas.update({tool.name: tool.inputSchema for tool in schema.tools})
            self.save(f'sources/{server}_tool_schemas.json', {tool.name: tool.inputSchema for tool in schema.tools})
        schema = self.tool_schemas.get(name, {})
        if 'args' in schema.get('required', []) and 'args' not in args:
            args = {'args': args}
        result = await asyncio.wait_for(self.sessions[server].call_tool(name, args), timeout=self.completion.tool_timeout)
        raw = result.model_dump(mode='json', exclude_none=True)
        for block in raw.get('content', []):
            if block.get('type') == 'image':
                block.pop('data', None)
                block['image_data_omitted'] = True
        entry = {'server': server, 'tool': name, 'arguments': args, 'result': raw,
                 'recorded_at_utc': datetime.now(timezone.utc).isoformat()}
        self.journal.setdefault(task_id, []).append(entry)
        self.save(f'sources/{task_id}_tool_calls.json', self.journal[task_id])
        return decode_result(raw)

    async def counts(self, stack: AsyncExitStack, task_id: str) -> dict[str, Any]:
        """Replay exact count queries; persist a dated GT rather than an invariant claim."""
        steps = self.tasks[task_id]['reference_solutions'][0]['steps']
        results = []
        for step in steps:
            server = 'eoss' if step['function_name'] in {'list_ontology_nodes', 'query_ontology'} else 'eumetsat'
            if task_id.endswith('_500011'):
                results.append(await self.invoke(stack, server, task_id, 'query_ontology', {
                    'cypher': 'MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label'}))
                break
            results.append(await self.invoke(stack, server, task_id, step['function_name'], step.get('arguments', {})))
        suffix = task_id.rsplit('_', 1)[-1]
        reference: dict[str, Any] = {'task_ID': task_id, 'reference_type': 'captured_source_snapshot',
            'status': 'computed_reference', 'provenance': {'path': str((self.config.output_dir / 'sources' / f'{task_id}_tool_calls.json').resolve()),
                'captured_at_utc': datetime.now(timezone.utc).isoformat()},
            'evaluation_policy': 'Valid for this captured source state; establish comparability to historical executions.'}
        rows = results[-1]
        if suffix == '500011':
            if isinstance(rows, dict) and isinstance(rows.get('node_counts'), dict):
                reference['node_counts'] = rows['node_counts']
            elif isinstance(rows, list) and rows and all('label' in row and 'count' in row for row in rows):
                reference['node_counts'] = {row['label']: int(row['count']) for row in rows}
            else:
                reference.update(status='snapshot_requires_schema_review', raw_counts=rows)
        elif suffix == '500013':
            if not isinstance(rows, list) or not rows or not all('question' in row and 'n_objectives' in row for row in rows):
                raise ValueError('Unexpected per-question count schema')
            reference['objectives_per_question'] = {row['question']: int(row['n_objectives']) for row in rows}
            reference['schema_warning'] = 'Counts are explicit direct graph edges, not inferred conceptual relationships.'
        else:
            if not isinstance(rows, dict) or not isinstance(rows.get('count'), (int, float)):
                raise ValueError('No authoritative total count')
            reference['slot_count'] = int(rows['count'])
            reference['listing_truncated'] = rows.get('truncated')
        path = Path(reference['provenance']['path'])
        reference['provenance']['sha256'] = self.sha(path)
        return reference

    @staticmethod
    def location_bbox(result: Any) -> list[float] | None:
        """Read documented bbox structures, preserving west/south/east/north order."""
        if not isinstance(result, dict):
            return None
        result = result.get('top_result', result)
        box = result.get('bbox_array', result.get('bbox'))
        if isinstance(box, str):
            box = [float(value) for value in box.split(',')]
        return box if isinstance(box, list) and len(box) == 4 else None

    def resolve_arguments(self, args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Resolve explicit dependencies; unresolved placeholders stop the workflow."""
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith('<'):
                if key == 'bbox' and context.get('bbox') is not None:
                    value = context['bbox']
                elif key in {'paths', 'identifier'} and context.get(key) is not None:
                    value = context[key]
                elif key in {'source', 'path_or_url'} and context.get('file'):
                    value = context['file']
                else:
                    raise ValueError(f'Unresolved reference dependency: {key}')
            if isinstance(value, str) and (value.startswith('output/') or value.startswith('tool_output/')):
                value = str(self.config.output_dir.resolve() / 'artifacts' / Path(value).relative_to(value.split('/')[0]))
                Path(value).parent.mkdir(parents=True, exist_ok=True)
            resolved[key] = value
        return resolved

    async def imagery(self, stack: AsyncExitStack, task_id: str) -> dict[str, Any]:
        """Replay the reference workflow, keeping inferred choices explicit in the journal."""
        steps = self.tasks[task_id]['reference_solutions'][0]['steps']
        context: dict[str, Any] = {}
        observations = []
        suffix = task_id.rsplit('_', 1)[-1]
        if suffix == '200012':
            result = await self.invoke(stack, 'cdse', task_id, 'geocode_place', {'place_name': 'Almeria, Spain'})
            context['bbox'] = self.location_bbox(result)
        for step in steps:
            name = step['function_name']
            arguments = dict(step.get('arguments', {}))
            if name == 'download_gee_product':
                point = self.geocode('Fresno, California, USA')
                arguments.update(latitude=point['lat'], longitude=point['lon'])
                context['reference_choice'] = 'Fresno as documented central-California reference location'
            args = self.resolve_arguments(arguments, context)
            if name in {'geocode_place', 'resolve_named_region', 'fetch_sentinel2'}:
                server = 'cdse'
            elif name in {'search_eumetsat_products', 'download_eumetsat_product', 'read_eumetsat_product'}:
                server = 'eumetsat'
            elif name == 'download_gee_product':
                server = 'gee'
            elif name == 'search_eoss':
                server = 'eoss'
            elif name == 'search_narratives':
                server = 'dashboard'
            else:
                server = 'gis'
            if name in {'clip_raster_with_bbox', 'compute_evi'}:
                result = self.local_raster_operation(name, args)
                self.journal.setdefault(task_id, []).append({'server': 'independent_rasterio_verifier',
                    'tool': name, 'arguments': args, 'result': result,
                    'adaptation': 'Reference tool absent from installed GIS server; independently reproduce documented operation.'})
                self.save(f'sources/{task_id}_tool_calls.json', self.journal[task_id])
            else:
                result = await self.invoke(stack, server, task_id, name, args)
            observations.append({'tool': name, 'result': result})
            bbox = self.location_bbox(result)
            if name in {'geocode_place', 'resolve_named_region'} and bbox:
                context['bbox'] = bbox
            if isinstance(result, dict):
                files = result.get('new_files', result.get('files', []))
                files = [entry.get('path', entry.get('file_path')) if isinstance(entry, dict) else entry for entry in files]
                files = [entry for entry in files if isinstance(entry, str)]
                if files:
                    context.update(paths=files, file=files[0])
                if result.get('saved_path'):
                    context['file'] = result['saved_path']
                products = result.get('products', [])
                if products and isinstance(products[0], dict) and products[0].get('identifier'):
                    context['identifier'] = products[0]['identifier']
            if args.get('destination'):
                context['file'] = args['destination']
        provenance_path = self.config.output_dir / 'sources' / f'{task_id}_tool_calls.json'
        reference = {'task_ID': task_id, 'reference_type': 'replayed_reference_observation',
            'status': 'observed_requires_scientific_input_validation', 'observations': observations,
            'resolved_context': context, 'provenance': {'path': str(provenance_path.resolve()), 'sha256': self.sha(provenance_path)},
            'evaluation_policy': 'Compare only matching scenes, spatial support, units, masks and processing, or recompute on task-valid candidate inputs.'}
        if context.get('file') and Path(context['file']).suffix.lower() in {'.tif', '.tiff'}:
            reference.update(self.raster_reference(Path(context['file']), suffix))
        return reference

    @staticmethod
    def local_raster_operation(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Reproduce missing GIS operations with explicit masks and reflectance units."""
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds

        source = args.get('raster_path', args.get('source'))
        destination = Path(args['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(source) as dataset:
            profile = dataset.profile.copy()
            if name == 'clip_raster_with_bbox':
                bounds = transform_bounds('EPSG:4326', dataset.crs, *args['bbox'])
                window = from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
                window = window.intersection(rasterio.windows.Window(0, 0, dataset.width, dataset.height))
                data = dataset.read(window=window, masked=True)
                profile.update(width=data.shape[2], height=data.shape[1], transform=dataset.window_transform(window))
                with rasterio.open(destination, 'w', **profile) as output:
                    output.write(data.filled(dataset.nodata or 0))
                    output.write_mask((~np.any(np.ma.getmaskarray(data), axis=0)).astype('uint8') * 255)
                return {'saved_path': str(destination), 'operation': 'clip_to_wgs84_bbox'}
            data = dataset.read(masked=True).astype('float64')
            # CDSE's custom-band evalscript explicitly requests REFLECTANCE FLOAT32.
            blue, red, nir = [data[args[key] - 1] for key in ('blue_band_index', 'red_band_index', 'nir_band_index')]
            denominator = nir + 6 * red - 7.5 * blue + 1
            with np.errstate(divide='ignore', invalid='ignore'):
                evi = 2.5 * (nir - red) / denominator
            evi = np.ma.masked_invalid(np.ma.masked_where(denominator == 0, evi))
            profile.update(count=1, dtype='float32', nodata=-9999)
            with rasterio.open(destination, 'w', **profile) as output:
                output.write(evi.filled(-9999).astype('float32'), 1)
            return {'saved_path': str(destination), 'formula': '2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)',
                    'input_encoding': 'CDSE custom bands REFLECTANCE FLOAT32',
                    'validation_warning': 'Verify scene resolution independently; calculation does not establish 10 m compliance.'}

    @staticmethod
    def raster_reference(path: Path, suffix: str) -> dict[str, Any]:
        """Independently measure the produced raster using its actual valid-pixel mask."""
        import rasterio

        with rasterio.open(path) as dataset:
            statistics = {}
            for band in range(1, dataset.count + 1):
                values = dataset.read(band, masked=True).compressed().astype('float64')
                values = values[np.isfinite(values)]
                if not len(values):
                    raise ValueError('Reference raster contains no valid values')
                statistics[str(band)] = {'min': float(values.min()), 'max': float(values.max()),
                    'mean': float(values.mean()), 'std': float(values.std()), 'valid_pixels': len(values)}
            result: dict[str, Any] = {'independent_band_statistics': statistics,
                'raster_identity': {'path': str(path), 'sha256': GroundTruthDiscovery.sha(path),
                    'crs': str(dataset.crs), 'width': dataset.width, 'height': dataset.height,
                    'pixel_size': list(dataset.res), 'bounds': list(dataset.bounds),
                    'descriptions': list(dataset.descriptions), 'scales': list(dataset.scales),
                    'offsets': list(dataset.offsets), 'nodata': dataset.nodata}}
            if suffix in {'200010', '200012'}:
                threshold = 0.6 if suffix == '200010' else 0.5
                values = dataset.read(1, masked=True).compressed()
                values = values[np.isfinite(values)]
                result['percentage_above_threshold'] = {'threshold': threshold, 'strictly_greater': True,
                    'percentage': float(np.count_nonzero(values > threshold) / len(values) * 100),
                    'denominator': 'finite unmasked pixels'}
        return result

    async def remote_phase(self) -> None:
        """Run count and imagery workflows sequentially with per-task checkpoints."""
        async with AsyncExitStack() as stack:
            for suffixes, operation in ((COUNTS, self.counts), (IMAGERY if self.completion.replay_imagery else set(), self.imagery)):
                for task_id in sorted(self.tasks):
                    suffix = task_id.rsplit('_', 1)[-1]
                    if suffix not in suffixes or (self.completion.task_suffixes and suffix not in self.completion.task_suffixes):
                        continue
                    try:
                        self.references[task_id] = await operation(stack, task_id)
                        self.records[task_id] = {'status': self.references[task_id]['status']}
                    except Exception as error:
                        self.records[task_id] = {'status': 'blocked', 'error_type': type(error).__name__,
                            'reason': str(error) if isinstance(error, ValueError) else 'Tool/service failure; inspect preserved call evidence'}
                    self.checkpoint()
                    print(task_id, self.records[task_id]['status'], flush=True)

    def checkpoint(self) -> None:
        """Persist reference values, unresolved tasks, and current completion counts."""
        self.save('ground_truths_computed.json', self.references)
        self.save('task_status.json', self.records)
        self.save('summary.json', {'updated_at': datetime.now(timezone.utc).isoformat(),
            'tasks_processed': len(self.records), 'statuses': dict(Counter(row['status'] for row in self.records.values())),
            'reference_count': len(self.references), 'aaj_profile_tasks': 48})

    def run(self) -> None:
        """Stage isolated GT and AAJ profile, then execute explicitly requested workflows."""
        from dotenv import load_dotenv

        load_dotenv(ROOT / '.env')
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / 'sources').mkdir(exist_ok=True)
        (self.config.output_dir / 'artifacts').mkdir(exist_ok=True)
        os.environ['EO_ARTIFACT_ROOT'] = str(self.config.output_dir.resolve() / 'artifacts')
        os.environ['PYTHONPATH'] = str(ROOT / 'src')
        self.save('manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
            'benchmark_sha256': self.sha(ROOT / 'data/tasks/benchmark_tasks.json'),
            'canonical_gt_sha256': self.sha(ROOT / 'data/tasks/ground_truths.json'),
            'implementation_sha256': self.sha(Path(__file__)),
            'geometry_tasks': sorted(GEOMETRY), 'count_tasks': sorted(COUNTS),
            'imagery_tasks': sorted(IMAGERY) if self.completion.replay_imagery else [],
            'policy': 'Independent reference generation; original candidate results and canonical GT preserved'})
        self.prepare_profile()
        for task_id in sorted(self.tasks):
            suffix = task_id.rsplit('_', 1)[-1]
            if suffix in GEOMETRY and (not self.completion.task_suffixes or suffix in self.completion.task_suffixes):
                try:
                    self.references[task_id] = self.geometry(task_id)
                    self.records[task_id] = {'status': 'computed_reference'}
                except Exception as error:
                    self.records[task_id] = {'status': 'blocked', 'error_type': type(error).__name__,
                        'reason': 'Geocoder unavailable or calculation input not resolved'}
                self.checkpoint()
                print(task_id, self.records[task_id]['status'], flush=True)
        asyncio.run(self.remote_phase())


class ReferenceReviewExporter:
    """Join staged references and saved candidate answers by exact task ID."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    @staticmethod
    def inventory_reference(solution: dict[str, Any]) -> dict[str, Any]:
        """Recover the execution-specific paper inventory from successful tool evidence."""
        inventories = []
        for step in solution.get('steps', []):
            if step.get('function_name') != 'list_papers':
                continue
            raw = step.get('tool_result')
            if isinstance(raw, list):
                raw = {'content': raw}
            if not isinstance(raw, dict):
                continue
            try:
                result = decode_result(raw)
            except ValueError:
                continue
            papers = result.get('papers') if isinstance(result, dict) else None
            if isinstance(papers, list) and all(isinstance(paper, str) for paper in papers):
                inventories.append(sorted(set(papers)))
        if not inventories:
            return {'status': 'unverified', 'reason': 'No successful recorded paper inventory'}
        if any(inventory != inventories[0] for inventory in inventories[1:]):
            return {'status': 'unverified', 'reason': 'Inventory changed between recorded calls', 'snapshots': inventories}
        return {'status': 'captured_candidate_inventory', 'reference_set': inventories[0],
                'source': 'successful list_papers tool response within this candidate execution',
                'policy': 'Compare reported IDs to this snapshot; no universal cross-run inventory.'}

    def run(self) -> None:
        """Export one review packet per excluded task, retaining provenance and traces."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        worker = GroundTruthDiscovery(DiscoveryConfig(self.output_dir))
        review_strategies = json.loads((ROOT / 'data/tasks/task_evaluation_strategies.json').read_text())['tasks']
        review_strategies = [row for row in review_strategies
                            if row in worker.strategies or row['task_ID'].endswith('_200005')]
        packets = {row['task_ID']: {'task_ID': row['task_ID'], 'task': worker.tasks[row['task_ID']],
            'evaluation_strategy': row, 'reference_candidates': [], 'candidate_answers': []}
            for row in review_strategies}
        sources = [ROOT / 'data/ground_truth_discovery_20261012_1011/proposed_ground_truths.json',
            ROOT / 'data/reference_completion_20261012/ground_truths_computed.json',
            ROOT / 'data/reference_completion_20261012/chernobyl_recovery/ground_truths_computed.json',
            ROOT / 'data/reference_completion_retry_20261012/ground_truths_computed.json',
            ROOT / 'data/reference_completion_retry_isolated_20261012/ground_truths_computed.json',
            ROOT / 'data/reference_counts_retry_20261012/ground_truths_computed.json']
        for path in sources:
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for task_id, reference in payload.items():
                if task_id in packets:
                    packets[task_id]['reference_candidates'].append({'source_file': str(path),
                        'source_sha256': worker.sha(path), 'reference': reference})
        import hashlib

        identities: dict[tuple[str, str], dict[str, Any]] = {}
        for path in sorted((ROOT / 'data/candidate_reruns').rglob('results*.json')):
            if any(word in path.name for word in ('before_', 'checkpoint')):
                continue
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                continue
            for task in payload.get('tasks', []):
                task_id, solution = task.get('task_ID'), task.get('generated_solution')
                if task_id not in packets or not solution:
                    continue
                digest = hashlib.sha256(json.dumps(solution, sort_keys=True).encode()).hexdigest()
                key = (task_id, digest)
                if key in identities:
                    identities[key]['source_files'].append(str(path))
                    continue
                entry = {'source_files': [str(path)], 'solution_sha256': digest,
                    'generated_solution': solution}
                if task_id.endswith('_200005') and isinstance(solution, dict):
                    entry['candidate_bound_reference'] = self.inventory_reference(solution)
                identities[key] = entry
                packets[task_id]['candidate_answers'].append(entry)
        index = []
        for task_id, packet in packets.items():
            packet['validation_notes'] = [
                'References are proposed; inspect captured evidence, not only numerical agreement.',
                'For imagery compare scene, dates, region, resolution, units, masks, and processing. Different valid choices may need candidate-bound recomputation.',
                'Use later retry references to investigate corrections; earlier records are preserved for audit.',
                'Original Addis Ababa facility match is invalid as city reference; use the city retry.',
                'First GEE/EUMETSAT retry did not inherit the artifact root; use the isolated retry for preserved source artifacts. Original output JSON files were not modified.',
                'Chernobyl reference uses 256 segments; valid candidate 64-segment polygons have slightly different areas.',
                'Missing evidence is unverified; task failures and demonstrated wrong outputs are distinct.']
            filename = task_id + '.json'
            worker.save(filename, packet)
            index.append({'task_ID': task_id, 'packet': filename,
                'reference_candidates': len(packet['reference_candidates']),
                'distinct_candidate_answers': len(packet['candidate_answers']),
                'evaluation_strategy': packet['evaluation_strategy']['evaluation_strategy']})
        worker.save('index.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'tasks': index, 'reference_source_files': [str(path) for path in sources if path.exists()],
            'candidate_scope': 'data/candidate_reruns/results*.json recursively; archived before/checkpoint filenames excluded; identical solutions deduplicated with paths retained'})
        print(f'Exported {len(index)} task packets to {self.output_dir}', flush=True)


class EvaluationPreparation:
    """Stage explicit evaluation inputs and audit coverage without publishing scores."""

    REWRITTEN = {'100006', '100010', '500008', '500013'}

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.worker = GroundTruthDiscovery(DiscoveryConfig(output_dir))

    def run(self) -> None:
        """Combine reference registries, snapshot profiles, and identify missing contracts."""
        from agenticeoecosystem.evaluation.metrics import numerical_metrics as nm
        from agenticeoecosystem.evaluation.judges.deterministic_judge import DeterministicAnswerJudge

        self.output_dir.mkdir(parents=True, exist_ok=False)
        benchmark = json.loads((ROOT / 'data/tasks/benchmark_tasks.json').read_text())
        strategies = json.loads((ROOT / 'data/tasks/task_evaluation_strategies.json').read_text())
        gt = json.loads((ROOT / 'data/tasks/ground_truths.json').read_text())
        self.worker.save('ground_truths_before.json', gt)
        sources = [ROOT / 'data/ground_truth_discovery_20261012_1011/proposed_ground_truths.json',
                   ROOT / 'data/reference_corrections_20261012/ground_truths_corrected.json']
        origins = {key: 'data/tasks/ground_truths.json' for key in gt}
        rejected = []
        corrections = json.loads(sources[-1].read_text())
        for source in sources:
            entries = json.loads(source.read_text())
            for task_id, entry in entries.items():
                if task_id not in self.worker.tasks:
                    raise ValueError(f'Unknown reference task ID: {task_id}')
                if entry.get('task_text', self.worker.tasks[task_id]['task_text']) != self.worker.tasks[task_id]['task_text']:
                    rejected.append({'task_ID': task_id, 'source': str(source), 'reason': 'Question version differs'})
                    continue
                if source == sources[-1] or task_id not in gt:
                    gt[task_id] = entry
                    origins[task_id] = str(source.relative_to(ROOT))
        correction_audit = self.audit_corrections(corrections)
        for task_id, entry in correction_audit.items():
            if entry['missing_provenance_log']:
                # Point to the actual local audit rather than fabricate an MCP invocation.
                gt[task_id]['original_provenance_log'] = gt[task_id]['provenance_log']
                gt[task_id]['provenance_log'] = str((self.output_dir / 'correction_source_audit.json').resolve())
        self.worker.save('correction_source_audit.json', correction_audit)
        methods_added = []
        for strategy in strategies['tasks']:
            task_id = strategy['task_ID']
            if task_id.endswith('_200005'):
                strategy['evaluation_methods'] = ['agent_as_judge']
            elif strategy['evaluation_strategy'] in {'download_contract', 'retrieval_validation'} or task_id.endswith('_200020'):
                strategy['evaluation_methods'] = list(dict.fromkeys([*strategy['evaluation_methods'], 'agent_as_judge']))
                methods_added.append(task_id)
            reference = gt.get(task_id, {})
            if reference.get('reference_type') in {'corrected_reference_observation', 'replayed_reference_workflow'}:
                strategy['evaluation_methods'] = list(dict.fromkeys([*strategy['evaluation_methods'], 'agent_as_judge', 'gis_verification_agent']))
                strategy['reference_policy'] = 'independently_recomputed_from_candidate_valid_inputs'
                strategy['evaluation_notes'] = strategy.get('evaluation_notes', '') + ' ' + reference.get('evaluation_policy', '')
                strategy['reference_recomputation_rules'] = {'masking_rule': reference.get('masking_rule'),
                    'aggregation': reference.get('aggregation'), 'source': reference.get('source'),
                    'policy': 'Reference artifacts are reference-owned and must not be substituted for candidate-owned inputs.'}
            if task_id.endswith('_200010'):
                strategy.setdefault('numerical_field_tolerances', {})['ndvi_stats.mean'] = {
                    'abs_tol': 0, 'rel_tol': 0.05, 'floor_scale': 0.01}
        self.worker.save('ground_truths.json', gt)
        self.worker.save('evaluation_strategies.json', strategies)
        self.worker.save('benchmark_tasks.json', benchmark)
        for filename in ('rubrics_108q_v4_evidence.json', 'tool_agent_mapping.json'):
            self.worker.save(filename, json.loads((ROOT / 'data/tasks' / filename).read_text()))
        rewritten = [task for task in benchmark['tasks'] if task['task_ID'].rsplit('_', 1)[-1] in self.REWRITTEN]
        self.worker.save('tasks_rewritten4.json', {'metadata': {'purpose': 'Generate fresh candidate answers for rewritten questions'}, 'tasks': rewritten})
        cohort = json.loads((ROOT / 'data/reference_review_20261012/index.json').read_text())['tasks']
        cohort_ids = {entry['task_ID'] for entry in cohort}
        audit = []
        for strategy in strategies['tasks']:
            task_id = strategy['task_ID']
            if task_id not in cohort_ids:
                continue
            reference, problems = gt.get(task_id, {}), []
            methods = set(strategy['evaluation_methods'])
            numeric_fields = []
            if methods & {'unit_normalized_error', 'numeric_tolerance_accuracy', 'exact_count_match', 'coordinate_distance_error'}:
                configured = strategy.get('numerical_field_tolerances', reference.get('field_tolerances'))
                try:
                    if reference.get('reference_type') == 'source_dependent_runtime':
                        problems.append('Needs candidate-bound paper/centroid verification and runtime distance')
                    else:
                        specs = None if configured is None else {key: nm.ToleranceSpec(**value) for key, value in configured.items()}
                        values, _ = nm.extract_scorable_fields(reference, specs)
                        numeric_fields = list(values)
                        if not numeric_fields:
                            problems.append('No executable numerical fields')
                except (KeyError, ValueError, TypeError) as error:
                    problems.append(str(error))
            deterministic = DeterministicAnswerJudge.supported(strategy, reference)
            unsupported = sorted(methods - deterministic - {
                'agent_as_judge', 'gis_verification_agent', 'unit_normalized_error', 'numeric_tolerance_accuracy',
                'exact_count_match', 'coordinate_distance_error', 'rubrics', 'llm_judge',
                'aaj_image_inspection', 'numerical_artifact_validation', 'aligned_series_mae_rmse', 'clipped_extent_comparison'})
            audit.append({'task_ID': task_id, 'strategy': strategy['evaluation_strategy'],
                'reference_present': bool(reference), 'reference_origin': origins.get(task_id),
                'numeric_fields': numeric_fields, 'unsupported_methods': unsupported, 'problems': problems,
                'needs_candidate_rerun': task_id.rsplit('_', 1)[-1] in self.REWRITTEN,
                'source_comparability': 'Requires candidate-bound verification unless existing fixed/catalogue contract applies'})
        self.worker.save('readiness.json', {'cohort_tasks': len(audit), 'tasks': audit,
            'rejected_old_question_references': rejected, 'corrections': len(corrections),
            'reference_candidates': sum(row['reference_present'] for row in audit),
            'numerical_contract_problems': [row for row in audit if row['problems']],
            'policy': 'No validation attestations fabricated; reference-owned files are not candidate artifact manifests.'})
        self.worker.save('manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'input_sources': {str(path): self.worker.sha(path) for path in [*sources,
                ROOT / 'data/tasks/ground_truths.json', ROOT / 'data/tasks/task_evaluation_strategies.json',
                ROOT / 'data/tasks/benchmark_tasks.json', Path(__file__), ROOT / 'scripts/references/correct_failed_references.py']},
            'reference_origins': origins, 'canonical_files_modified': False})
        self.audit_candidate_versions()
        print(f'Staged {len(gt)} references; audited {len(audit)} questions at {self.output_dir}', flush=True)

    def audit_candidate_versions(self) -> None:
        """Record source identity and prevent old answers being reused for rewritten tasks."""
        published = Path('/shared/projects/philab-eve/ESAEOBench/candidate_final')
        paths = sorted(published.glob('results*.json'))
        relative = [
            'mimo_v26_flash_multi_20261008/results_MiMo-V2.6-Flash-multi.json',
            'mimo_v26_flash_multi_20261008_budget_resume_20261008_2230/results_MiMo-V2.6-Flash-multi.json',
            'deepseek_v4_flash_vision_multi_20261008/results_DeepSeek-V4-Flash-Vision-Exp-multi.json',
            'deepseek_v4_flash_vision_multi_20261008_budget_resume_20261008_2230/results_DeepSeek-V4-Flash-Vision-Exp-multi.json',
            'qwen3_8b_remaining195_20261008/results_Qwen3-8B-remaining195.json',
            'api_full265_20261007/Nemotron-3-Super-OR/results_Nemotron-3-Super-OR.json']
        paths.extend(ROOT / 'data/candidate_reruns' / name for name in relative)
        rows = []
        for path in paths:
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            mismatch = [task['task_ID'] for task in payload['tasks']
                if task['task_ID'] in self.worker.tasks and task['task_text'] != self.worker.tasks[task['task_ID']]['task_text']]
            rows.append({'source_file': str(path), 'sha256': self.worker.sha(path),
                'tasks': len(payload['tasks']), 'saved_answers': sum(bool(task.get('generated_solution')) for task in payload['tasks']),
                'question_version_mismatches': mismatch,
                'policy': 'Base and budget-resume files are alternatives/continuations, not independent models; select or merge into a new candidate file with per-task provenance.'})
        self.worker.save('candidate_sources.json', rows)

    def audit_corrections(self, entries: dict[str, Any]) -> dict[str, Any]:
        """Check preserved artifact identities without claiming scientific validation."""
        cache: dict[str, str] = {}
        results = {}

        def identities(node: Any) -> list[dict[str, Any]]:
            records = []
            if isinstance(node, dict):
                if isinstance(node.get('path'), str) and node.get('sha256'):
                    path = Path(node['path'])
                    path = path if path.is_absolute() else ROOT / path
                    if path.is_file() and str(path) not in cache:
                        cache[str(path)] = self.worker.sha(path)
                    records.append({'path': str(path), 'expected_sha256': node['sha256'],
                        'actual_sha256': cache.get(str(path)), 'matches': cache.get(str(path)) == node['sha256']})
                for value in node.values():
                    records.extend(identities(value))
            elif isinstance(node, list):
                for value in node:
                    records.extend(identities(value))
            return records

        for task_id, entry in entries.items():
            log = ROOT / entry['provenance_log']
            results[task_id] = {'task_text_matches': entry['task_text'] == self.worker.tasks[task_id]['task_text'],
                'missing_provenance_log': not log.is_file(),
                'provenance_log': str(log), 'artifact_hash_checks': identities(entry.get('source', {})),
                'scientific_validation': 'pending; successful generation and matching hashes do not establish candidate comparability'}
            if task_id.rsplit('_', 1)[-1] in {'200010', '200011', '200012'}:
                results[task_id]['local_derivation_check'] = self.verify_local_derivation(task_id, entry)
        return results

    @staticmethod
    def verify_local_derivation(task_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Recompute the three local-only corrections from preserved raster pixels."""
        import rasterio

        suffix = task_id.rsplit('_', 1)[-1]
        source = entry['source']
        if suffix == '200011':
            path = ROOT / source['reprojected_raster']['path']
            with rasterio.open(path) as dataset:
                data = dataset.read().astype('float64')
            valid = np.all(np.isfinite(data), axis=0)
            measured = {name: {stat: float(value) for stat, value in zip(
                ('min', 'max', 'mean', 'std'),
                (data[i][valid].min(), data[i][valid].max(), data[i][valid].mean(), data[i][valid].std()))}
                for i, name in enumerate(('R(B04)', 'G(B03)', 'B(B02)'))}
            expected = entry['band_stats']
            comparisons = {f'{name}.{stat}': bool(np.isclose(value, expected[name][stat], rtol=1e-10, atol=1e-12))
                           for name, statistics in measured.items() for stat, value in statistics.items()}
            scope = 'Statistics from preserved reprojected raster; projection operation itself not independently replayed here.'
        else:
            path = ROOT / source['raster']['path']
            with rasterio.open(path) as dataset:
                data = dataset.read().astype('float64')
            nir, red = (data[0], data[1]) if suffix == '200010' else (data[3], data[2])
            valid = np.isfinite(nir) & np.isfinite(red) & (nir > 0) & (red > 0)
            values = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
            measured = {stat: float(value) for stat, value in zip(('min', 'max', 'mean', 'std'),
                (values.min(), values.max(), values.mean(), values.std()))}
            comparisons = {stat: bool(np.isclose(value, entry['ndvi_stats'][stat], rtol=1e-10, atol=1e-12))
                           for stat, value in measured.items()}
            threshold = .6 if suffix == '200010' else .5
            percent_key = f'pct_pixels_ndvi_gt_{threshold}'
            measured[percent_key] = float(np.count_nonzero(values > threshold) / len(values) * 100)
            comparisons[percent_key] = bool(np.isclose(measured[percent_key], entry[percent_key], rtol=1e-10, atol=1e-12))
            scope = 'NDVI and threshold percentage independently recomputed from positive finite input reflectance bands.'
        return {'scope': scope, 'source_path': str(path), 'source_sha256': GroundTruthDiscovery.sha(path),
            'computed': measured, 'comparisons': comparisons, 'all_match': all(comparisons.values()),
            'policy': 'This is a local verification record, not a fabricated MCP tool-call log or proof of candidate-input comparability.'}
