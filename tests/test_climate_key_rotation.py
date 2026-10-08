"""Offline credential selection and download failover tests."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
from typing import List, Optional
import unittest
from unittest.mock import Mock

import requests

SOURCE = Path(__file__).resolve().parents[1] / 'src/agenticeoecosystem/servers/climate/server.py'


def load_helpers() -> dict:
    names = {'_CDSKeyPool', '_credential_retry_delay', '_internal_fetch_and_inspect'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    namespace = dict(List=List, Optional=Optional, threading=threading,
                     requests=requests, datetime=datetime, timezone=timezone,
                     parsedate_to_datetime=parsedate_to_datetime)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def http_error(status: int, retry_after: str = '') -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response.headers['Retry-After'] = retry_after
    return requests.HTTPError('failed secret-one', response=response)


class ClimateKeyRotationTests(unittest.TestCase):
    def test_selection_deduplicates_and_is_thread_safe(self) -> None:
        pool = load_helpers()['_CDSKeyPool']([
            ('first', 'one'), ('second', 'two'), ('duplicate', ' one '), ('empty', '')])
        with ThreadPoolExecutor(max_workers=8) as executor:
            indices = list(executor.map(lambda _: pool.select(), range(100)))
        self.assertEqual(indices.count(0), 50)
        self.assertEqual(indices.count(1), 50)
        self.assertEqual(len(pool.credentials), 2)
        self.assertEqual(pool.redact(ValueError('one two')), '[REDACTED] [REDACTED]')
        fallback = load_helpers()['_CDSKeyPool']([])
        self.assertIsNone(fallback.credentials[0][1])

    def test_retry_classification_and_headers(self) -> None:
        delay = load_helpers()['_credential_retry_delay']
        self.assertEqual(delay(http_error(429, '30'), 15), 30)
        self.assertEqual(delay(http_error(401), 15), 15)
        self.assertEqual(delay(http_error(429, 'bad'), 15), 15)
        self.assertGreater(delay(http_error(429, 'Wed, 01 Jan 2099 00:00:00 GMT'), 15), 15)
        for error in [http_error(403), http_error(500), ValueError('401')]:
            self.assertIsNone(delay(error, 15))

    def run_download(self, errors: list[Exception]) -> tuple[str, Mock, Mock, Mock]:
        namespace = load_helpers()
        client = Mock()
        client.retrieve.side_effect = [*errors, None]
        factory = Mock(return_value=client)
        sleeps = Mock()
        logger = Mock()

        class InlineExecutor:
            def run_in_executor(self, executor: object, function: object) -> asyncio.Future:
                future = asyncio.get_running_loop().create_future()
                future.set_result(function())
                return future

        async def inspect(path: str) -> str:
            return 'inspected'

        def inspect_inline(coroutine: object) -> str:
            # The production callback runs in a worker thread with its own loop.
            coroutine.close()
            return 'inspected'

        namespace.update(_CDS_KEYS=namespace['_CDSKeyPool']([
            ('CDS_KEY', 'secret-one'), ('CDS_KEY_2', 'secret-two')]),
            cdsapi=SimpleNamespace(Client=factory), URL='test-url',
            asyncio=SimpleNamespace(run=inspect_inline, get_running_loop=InlineExecutor),
            time=SimpleNamespace(time=lambda: 0, sleep=sleeps), logger=logger,
            resolve_run_path=Path, _DOWNLOAD_MAX_ATTEMPTS=3,
            _DOWNLOAD_RETRY_BACKOFF_S=15, _internal_inspect_netcdf=inspect)
        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(namespace['_internal_fetch_and_inspect'](
                'dataset', {}, str(Path(directory) / 'result.nc')))
        return result, factory, sleeps, logger

    def test_auth_and_rate_limit_switch_clients(self) -> None:
        for status in (401, 429):
            result, factory, sleeps, logger = self.run_download([http_error(status, '30')])
            self.assertIn('Successfully', result)
            self.assertEqual([call.kwargs['key'] for call in factory.call_args_list],
                             ['secret-one', 'secret-two'])
            self.assertIsNot(factory.call_args_list[0].kwargs['session'],
                             factory.call_args_list[1].kwargs['session'])
            sleeps.assert_called_once_with(30)
            self.assertNotIn('secret-one', str(logger.mock_calls))

    def test_transfer_failure_retains_client(self) -> None:
        result, factory, sleeps, _ = self.run_download([AssertionError('size mismatch')])
        self.assertIn('Successfully', result)
        self.assertEqual(factory.call_count, 1)
        sleeps.assert_called_once_with(15)

    def test_failures_are_bounded_and_redacted(self) -> None:
        result, factory, sleeps, logger = self.run_download([http_error(401)] * 3)
        self.assertIn('after 3 attempts', result)
        self.assertNotIn('secret-one', result)
        self.assertNotIn('secret-one', str(logger.mock_calls))
        self.assertEqual(factory.call_count, 3)
        self.assertEqual(sleeps.call_count, 2)
