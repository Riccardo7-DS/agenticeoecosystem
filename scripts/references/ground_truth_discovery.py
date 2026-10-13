"""Checkpoint proposed references, source snapshots, and unresolved task requirements."""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from regenerate_numeric_ground_truths import NumericalReferenceGenerator, default_sources

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"numerical_answer", "set_answer", "retrieval_validation", "download_contract"}


def literal_constant(path: Path, name: str, record_fields: tuple[str, ...] | None = None) -> Any:
    """Read a literal catalogue without importing servers or initializing credentials."""
    definitions = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node.value

    def evaluate(node: ast.AST, visiting: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, ast.Name):
            if node.id in visiting or node.id not in definitions:
                raise ValueError('Unresolved catalogue constant')
            return evaluate(definitions[node.id], visiting | {node.id})
        if isinstance(node, ast.Dict):
            return {evaluate(key, visiting): evaluate(value, visiting) for key, value in zip(node.keys, node.values)}
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [evaluate(item, visiting) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = evaluate(node.left, visiting), evaluate(node.right, visiting)
            if type(left) is type(right) and isinstance(left, (list, tuple, str)):
                return left + right
            raise ValueError('Unsupported catalogue expression')
        return ast.literal_eval(node)

    if name not in definitions:
        raise ValueError(f'Literal catalogue {name} not found')
    if record_fields is not None:
        outer = definitions[name]
        if not isinstance(outer, ast.Dict) or not all(isinstance(record, ast.Dict) for record in outer.values):
            raise ValueError('Expected a catalogue of literal records')
        for record in outer.values:
            pairs = [(key, value) for key, value in zip(record.keys, record.values)
                     if isinstance(key, ast.Constant) and key.value in record_fields]
            record.keys = [key for key, _ in pairs]
            record.values = [value for _, value in pairs]
    return evaluate(definitions[name], frozenset({name}))


@dataclass(frozen=True)
class DiscoveryConfig:
    """Independent output directory and bounded network reads."""
    output_dir: Path
    request_timeout: int = 30
    ontology_timeout: int = 60


class GroundTruthDiscovery:
    """Find references using source data, never candidate final answers or LLM guesses."""

    def __init__(self, config: DiscoveryConfig) -> None:
        self.config = config
        self.tasks = {t['task_ID']: t for t in json.loads((ROOT / 'data/tasks/benchmark_tasks.json').read_text())['tasks']}
        self.strategies = [t for t in json.loads((ROOT / 'data/tasks/task_evaluation_strategies.json').read_text())['tasks']
                           if t['evaluation_strategy'] in EXCLUDED]
        self.existing = json.loads((ROOT / 'data/tasks/ground_truths.json').read_text())
        self.records: dict[str, dict[str, Any]] = {}
        self.proposals: dict[str, dict[str, Any]] = {}
        self.cache: dict[str, Any] = {}
        self.sources = {source.task_id: source for source in default_sources()}

    @staticmethod
    def sha(path: Path) -> str:
        """Fingerprint exact source bytes."""
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def save(self, name: str, payload: Any) -> None:
        """Atomically checkpoint each completed task."""
        target = self.config.output_dir / name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')
        temporary.replace(target)

    def snapshot(self, name: str, data: Any, provenance: dict[str, Any]) -> dict[str, Any]:
        """Preserve independently obtained evidence and its source identity."""
        self.save(f'sources/{name}.json', data)
        path = self.config.output_dir / 'sources' / f'{name}.json'
        return {**provenance, 'snapshot_path': str(path.resolve()), 'snapshot_sha256': self.sha(path),
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}

    def catalogue(self, name: str) -> tuple[Any, dict[str, Any]]:
        """Extract catalogue constants rather than hash unrelated server implementation."""
        if name not in self.cache:
            if name == 'products_list':
                path = next((ROOT / '.venv/lib').glob('python*/site-packages/eumetsearch/data_collection/downloader.py'))
            else:
                path = ROOT / 'src/agenticeoecosystem/servers/climate/server.py'
            fields = ('product_id', 'satpy_reader', 'bands') if name == 'products_list' else None
            data = literal_constant(path, name, fields)
            self.cache[name] = (data, self.snapshot(name, data, {
                'source_type': 'versioned_local_catalogue', 'module': str(path),
                'module_sha256': self.sha(path), 'constant': name}))
        return self.cache[name]

    def dashboard(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve the complete narrative index, including entries hidden by tool previews."""
        if 'dashboard' not in self.cache:
            url = 'https://esa-eodashboards.github.io/eodashboard-narratives/narratives.json'
            response = requests.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not all(isinstance(row, dict) and row.get('title') for row in data):
                raise ValueError('Unexpected narrative index schema')
            self.cache['dashboard'] = (data, self.snapshot('dashboard_narratives', data, {
                'source_type': 'current_source_snapshot', 'url': url,
                'historical_comparability': 'Not established; validate against the candidate execution snapshot'}))
        return self.cache['dashboard']

    def propose(self, task_id: str, reference: dict[str, Any], provenance: dict[str, Any]) -> None:
        """Stage a reviewable contract; never promote it into canonical GT automatically."""
        self.proposals[task_id] = {'task_ID': task_id, 'task_text': self.tasks[task_id]['task_text'],
            **reference, 'provenance': provenance, 'status': 'proposed_requires_review',
            'evaluation_policy': 'Validate temporal/source and candidate-input comparability before scoring.'}
        self.records[task_id]['status'] = 'proposed_reference'

    def discover_set(self, task_id: str) -> bool:
        """Handle explicit catalogue and complete-dashboard set contracts."""
        task = self.tasks[task_id]
        steps = task['reference_solutions'][0]['steps']
        if len(steps) != 1:
            return False
        step = steps[0]
        tool, args = step['function_name'], step.get('arguments', {})
        suffix = task_id.rsplit('_', 1)[-1]
        if tool in {'list_themes', 'list_narratives', 'find_narratives_for_indicator'}:
            data, provenance = self.dashboard()
            if tool == 'list_themes':
                def theme(row: dict[str, Any]) -> str:
                    value = row.get('theme', 'unknown')
                    if isinstance(value, list):
                        value = value[0] if value else 'unknown'
                    if not isinstance(value, str):
                        raise ValueError('Unexpected theme schema')
                    return value.lower()
                reference = sorted({theme(row) for row in data})
            elif tool == 'list_narratives':
                reference = sorted({row['title'] for row in data})
            else:
                indicator = args['indicator_code'].strip()
                if not all(isinstance(row.get('collections', []), list) for row in data):
                    raise ValueError('Indicator membership is not a complete list')
                reference = sorted({row['title'] for row in data if indicator in row.get('collections', [])})
            self.propose(task_id, {'reference_set': reference, 'reference_count': len(reference)}, provenance)
            return True
        selection: list[str] | None = None
        if tool == 'list_cds_datasets':
            data, provenance = self.catalogue('_CDS_DATASETS')
            if suffix == '600014':
                labels = {'projections-cmip6': 'future climate projections',
                          'seasonal-original-single-levels': 'seasonal forecasts'}
                self.propose(task_id, {'reference_dictionary': labels, 'reference_count': 2,
                    'catalogue_snapshot': {key: data[key] for key in labels}}, provenance)
                return True
            selection = [key for key in data if key.startswith('reanalysis-era5-')]
        elif tool == 'list_era5_single_level_variables':
            data, provenance = self.catalogue('_ERA5_SINGLE_LEVEL_VARIABLES')
            selection = data['precipitation_and_hydrology']
        elif tool == 'list_era5_pressure_level_variables':
            constant = '_ERA5_PRESSURE_LEVELS' if suffix == '600007' else '_ERA5_PRESSURE_LEVEL_VARIABLES'
            data, provenance = self.catalogue(constant)
            selection = [str(value) for value in data]
        elif tool == 'list_eumetsat_products':
            data, provenance = self.catalogue('products_list')
            keys = ['MTG-FCI-L2-CLM', 'MTG-FCI-L2-AFM', 'MTG-FCI-L2-OCA'] if suffix == '500019' else list(data)
            reference = [{'id': key, 'aliases': [data[key]['product_id']]} for key in keys]
            self.propose(task_id, {'reference_set': reference, 'reference_count': len(reference),
                                  'product_metadata': {key: data[key] for key in keys}}, provenance)
            return True
        if selection is not None:
            self.propose(task_id, {'reference_set': selection, 'reference_count': len(selection)}, provenance)
            return True
        return False

    async def ontology(self, task_id: str) -> bool:
        """Read curated ontology queries and preserve raw results for schema review."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        steps = self.tasks[task_id]['reference_solutions'][0]['steps']
        if len(steps) != 1 or steps[0]['function_name'] not in {'query_ontology', 'list_ontology_nodes'}:
            return False
        step = steps[0]
        query = step.get('arguments', {}).get('cypher', '')
        if query and not query.lstrip().upper().startswith('MATCH '):
            return False
        url, auth = os.getenv('GRAPHRAG_MCP_URL'), os.getenv('GRAPHRAG_AUTH_HEADER')
        if not url or not auth:
            self.records[task_id]['reason'] = 'Ontology credentials unavailable'
            return False
        client = MultiServerMCPClient({'eoss': {'transport': 'streamable_http', 'url': url,
                                              'headers': {'Authorization': auth}}})
        tools = await asyncio.wait_for(client.get_tools(), timeout=self.config.ontology_timeout)
        tool = next(tool for tool in tools if tool.name == step['function_name'])
        result = await asyncio.wait_for(tool.ainvoke(step.get('arguments', {})), timeout=self.config.ontology_timeout)
        provenance = self.snapshot(task_id, result, {'source_type': 'current_ontology_snapshot',
            'tool': step['function_name'], 'arguments': step.get('arguments', {}),
            'historical_comparability': 'Not established'})
        self.records[task_id].update(status='source_snapshot_needs_contract', provenance=provenance,
            reason='Raw authoritative response captured; review result schema and completeness before creating a reference set')
        blocks = result if isinstance(result, list) else []
        texts = [block.get('text', '') for block in blocks if isinstance(block, dict) and block.get('type') == 'text']
        if len(texts) == 1:
            try:
                rows = json.loads(texts[0])
            except json.JSONDecodeError:
                self.records[task_id].update(status='source_error', error_type='NonJSONOntologyResponse',
                    reason='Ontology response was not a structured result; snapshot retained for diagnosis')
                return True
            if isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get('id'), str) for row in rows):
                entities: dict[str, dict[str, Any]] = {}
                for row in rows:
                    identifier = row['id']
                    record = entities.setdefault(identifier, {'id': identifier, 'aliases': []})
                    name = row.get('name', row.get('tipping_point'))
                    if isinstance(name, str) and name not in record['aliases']:
                        record['aliases'].append(name)
                self.propose(task_id, {'reference_set': list(entities.values()),
                    'reference_count': len(entities), 'source_rows': rows,
                    'completeness_status': 'Review service truncation and query scope before scoring'}, provenance)
        return True

    def run(self) -> None:
        """Checkpoint all 105 excluded tasks and attempt supported source retrievals."""
        from dotenv import load_dotenv

        load_dotenv(ROOT / '.env')
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / 'sources').mkdir(exist_ok=True)
        self.save('manifest.json', {'started_at': datetime.now(timezone.utc).isoformat(),
            'benchmark_sha256': self.sha(ROOT / 'data/tasks/benchmark_tasks.json'),
            'strategy_sha256': self.sha(ROOT / 'data/tasks/task_evaluation_strategies.json'),
            'canonical_gt_sha256': self.sha(ROOT / 'data/tasks/ground_truths.json'),
            'implementation_sha256': self.sha(Path(__file__)),
            'scope': sorted(EXCLUDED), 'tasks': len(self.strategies),
            'policy': 'No LLM calls, candidate-answer copying, canonical GT changes, or automatic score promotion'})
        generator = NumericalReferenceGenerator(self.config.output_dir / 'proposed_ground_truths.json',
                                                self.config.output_dir / 'sources/extracted_inputs')
        for strategy in self.strategies:
            task_id = strategy['task_ID']
            self.records[task_id] = {'category': strategy['evaluation_strategy'],
                'task_text': self.tasks[task_id]['task_text'], 'status': 'unresolved',
                'existing_gt': task_id in self.existing,
                'reference_policy': strategy.get('reference_policy'),
                'reason': 'Requires an explicit executable contract or independently recomputed candidate-valid inputs'}
            try:
                handled = self.discover_set(task_id) if strategy['evaluation_strategy'] == 'set_answer' else False
                if not handled and task_id in self.sources:
                    source = self.sources[task_id]
                    if source.path.is_file():
                        reference = generator.calculate(source)
                        provenance = reference.pop('provenance')
                        self.propose(task_id, reference, provenance)
                        handled = True
                    else:
                        self.records[task_id]['reason'] = f'Identified numerical input is missing: {source.path}'
                if not handled and strategy['evaluation_strategy'] == 'set_answer':
                    asyncio.run(self.ontology(task_id))
            except Exception as error:
                # Per-task boundary includes MCP ExceptionGroups; continue other tasks.
                # Do not copy exception text from authenticated clients into logs.
                self.records[task_id].update(status='source_error', error_type=type(error).__name__)
            self.save('proposed_ground_truths.json', self.proposals)
            self.save('task_status.json', self.records)
            print(task_id, self.records[task_id]['status'], flush=True)
        from collections import Counter
        self.save('summary.json', {'finished_at': datetime.now(timezone.utc).isoformat(),
            'tasks': len(self.records), 'statuses': dict(Counter(row['status'] for row in self.records.values())),
            'proposed': len(self.proposals), 'canonical_gt_unchanged': self.sha(ROOT / 'data/tasks/ground_truths.json') ==
                json.loads((self.config.output_dir / 'manifest.json').read_text())['canonical_gt_sha256']})
