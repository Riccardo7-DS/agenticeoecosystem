"""Verify lazy package imports and backwards-compatible module identity."""
import subprocess
import sys
import unittest
from pathlib import Path


class EvaluationLayoutTests(unittest.TestCase):
    def test_lazy_exports_and_module_aliases(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src"
        code = f'''
import sys
import importlib
from types import ModuleType
sys.path.insert(0, {str(source)!r})
import agenticeoecosystem.evaluation as evaluation
assert "agenticeoecosystem.evaluation.legacy.evaluator" not in sys.modules
assert "agenticeoecosystem.evaluation.judges.agent_judge" not in sys.modules
credentials = ModuleType("agenticeoecosystem.config.credentials")
credentials.get_openai_client = lambda *args, **kwargs: None
sys.modules[credentials.__name__] = credentials
for old, canonical in evaluation.MODULE_ALIASES.items():
    previous = importlib.import_module(f"agenticeoecosystem.evaluation.{{old}}")
    current = importlib.import_module(f"agenticeoecosystem.evaluation.{{canonical}}")
    assert previous is current, old
assert evaluation.Task is importlib.import_module("agenticeoecosystem.evaluation.dataclasses").Task
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
