"""Check download-to-reader attribution without live downloads or Satpy imports."""
import ast
from pathlib import Path
from typing import Callable, Optional
import unittest


class DownloadPathTests(unittest.TestCase):
    """Chunk paths belong to a single observation, never to every observation."""

    def test_single_fci_product_keeps_all_chunks(self) -> None:
        resolve = self._resolver()
        chunks = ['/data/FCI-CHK-BODY-0001.nc', '/data/FCI-CHK-BODY-0002.nc']
        self.assertEqual(resolve('FCI-parent', chunks, 1), chunks)
        self.assertEqual(resolve('FCI-parent', chunks, 2), [])

    def test_native_products_remain_separate(self) -> None:
        resolve = self._resolver()
        paths = ['/data/MSG3-obs1.nat', '/data/MSG3-obs2.nat']
        self.assertEqual(resolve('MSG3-obs1', paths, 2), paths[:1])

    def test_empty_explicit_paths_do_not_read_unrelated_files(self) -> None:
        source = Path(__file__).resolve().parents[1] / 'src/agenticeoecosystem/servers/eumetsat/server.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_read_product')
        namespace = {'Optional': Optional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        with self.assertRaisesRegex(ValueError, 'must not be empty'):
            namespace['_read_product']('MTG-FCI-L1C-FDHSI', None, None, False, paths=[])

    @staticmethod
    def _resolver() -> Callable[[Optional[str], list[str], int], list[str]]:
        source = Path(__file__).resolve().parents[1] / 'src/agenticeoecosystem/servers/eumetsat/server.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_downloaded_product_paths')
        namespace = {'Optional': Optional, 'Path': Path}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['_downloaded_product_paths']


if __name__ == '__main__':
    unittest.main()
