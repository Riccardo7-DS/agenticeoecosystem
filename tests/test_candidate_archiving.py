"""Offline regressions for archiving across legacy and current output roots."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class CandidateArchivingTests(unittest.TestCase):
    """Verify artifact retention and confinement without importing candidate agents."""

    @classmethod
    def setUpClass(cls) -> None:
        source = Path(__file__).resolve().parents[1] / "src/agenticeoecosystem/evaluation/evidence/candidate_evidence.py"
        package = ModuleType("_archive_test")
        package.__path__ = [str(source.parent.parent)]
        schema = ModuleType("_archive_test.dataclasses")
        schema.Task = object
        spec = importlib.util.spec_from_file_location("_archive_test.evidence.candidate_evidence", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"_archive_test": package, "_archive_test.dataclasses": schema}):
            spec.loader.exec_module(module)
        cls.evidence = module.CandidateEvidence

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.legacy = self.root / "output"
        self.current = self.root / "tool_output"
        self.legacy.mkdir()
        self.current.mkdir()

    def task(self, paths: list[str]) -> SimpleNamespace:
        """Create a candidate with exact recorded artifact paths."""
        solution = SimpleNamespace(
            steps=[SimpleNamespace(arguments={}, tool_result={"paths": paths})],
            model_dump=lambda: {"paths": paths},
        )
        return SimpleNamespace(task_ID="archive-test", task_text="Produce files.",
                               generated_solution=solution, candidate_artifacts=[])

    def test_both_roots_are_archived_with_distinct_files_and_valid_hashes(self) -> None:
        paths = [self.legacy / "map.txt", self.current / "map.txt"]
        for index, path in enumerate(paths):
            path.write_text(f"candidate artifact {index}")
        task = self.task([str(path) for path in paths])
        self.evidence.archive(task, (self.legacy, self.current), self.root / "evidence")
        self.assertEqual(len(task.candidate_artifacts), 2)
        for artifact in task.candidate_artifacts:
            archived = Path(artifact["archived_path"])
            self.assertEqual(archived.read_bytes(), Path(artifact["original_path"]).read_bytes())
            self.assertEqual(self.evidence.file_digest(archived), artifact["sha256"])

    def test_outside_paths_traversal_and_symlinks_are_excluded(self) -> None:
        outside = self.root / "secret.txt"
        outside.write_text("outside candidate outputs")
        (self.current / "link.txt").symlink_to(outside)
        task = self.task([str(outside), "../secret.txt", str(self.current / "link.txt")])
        self.assertEqual(self.evidence.artifact_paths(task, (self.legacy, self.current)), [])

    def test_single_root_and_relative_paths_remain_supported(self) -> None:
        path = self.legacy / "map.txt"
        path.write_text("legacy artifact")
        task = self.task(["map.txt", str(path)])
        self.assertEqual(self.evidence.artifact_paths(task, self.legacy), [path.resolve()])


if __name__ == "__main__":
    unittest.main()
