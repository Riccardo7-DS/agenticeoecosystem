"""Test portable rerun inputs without calling model or MCP services."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "generation"))
from prepare_missing_candidate_reruns import MissingCandidateReruns, RerunConfig


class PortableInputTests(unittest.TestCase):
    """Portable inputs must be independent of results and validated before execution."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.runner = MissingCandidateReruns(RerunConfig(self.root / "absent-results", self.root / "output"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def snapshot(self, solution: dict | None = None, filename: str = "tasks.json") -> Path:
        """Create a minimal portable snapshot and its hash manifest."""
        path = self.inputs / "tasks.json"
        path.write_text(json.dumps({"metadata": {}, "tasks": [{
            "task_ID": "task-1", "task_text": "List themes.", "generated_solution": solution,
        }]}))
        manifest = {"candidates": {"candidate": {
            "input": filename, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_ids": ["task-1"],
        }}}
        (self.inputs / "manifest.json").write_text(json.dumps(manifest))
        return path

    def test_load_without_historical_results(self) -> None:
        path = self.snapshot()
        manifest = self.runner.load_prepared(self.inputs)
        self.assertEqual(manifest["candidates"]["candidate"]["input"], str(path.resolve()))

    def test_changed_input_rejected(self) -> None:
        path = self.snapshot()
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.runner.load_prepared(self.inputs)

    def test_candidate_answer_rejected(self) -> None:
        self.snapshot({"steps": [], "final_answer": "Previously generated answer"})
        with self.assertRaisesRegex(ValueError, "contain candidate results"):
            self.runner.load_prepared(self.inputs)

    def test_outside_input_rejected(self) -> None:
        self.snapshot(filename="../tasks.json")
        with self.assertRaisesRegex(ValueError, "inside its task directory"):
            self.runner.load_prepared(self.inputs)


if __name__ == "__main__":
    unittest.main()
