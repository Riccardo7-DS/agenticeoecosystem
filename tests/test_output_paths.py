"""Artifact paths remain consistent across package roots and subprocesses."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agenticeoecosystem.utils.output_paths import artifact_root, resolve_run_path, server_output_dir


class OutputPathTests(unittest.TestCase):
    def test_run_root_overrides_server_defaults_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "EO_ARTIFACT_ROOT": directory, "CDSE_OUTPUT_DIR": "/old/location",
        }):
            root = Path(directory).resolve()
            self.assertEqual(artifact_root(), root)
            self.assertEqual(server_output_dir("cdse_geocode", "CDSE_OUTPUT_DIR"), root / "cdse_geocode")
            self.assertEqual(resolve_run_path("output/task/image.tif"), root / "task/image.tif")
            self.assertEqual(resolve_run_path("tool_output/gee/image.tif"), root / "gee/image.tif")
            self.assertEqual(resolve_run_path("image.tif"), root / "misc/image.tif")
            with self.assertRaises(ValueError):
                resolve_run_path("output/../../outside.tif")

    def test_absolute_inputs_remain_readable_without_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.tif"
            self.assertEqual(resolve_run_path(str(path)), path)


if __name__ == "__main__":
    unittest.main()
