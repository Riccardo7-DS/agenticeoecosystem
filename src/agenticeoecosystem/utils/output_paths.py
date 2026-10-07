"""Shared artifact roots for candidate tools, archiving and evaluator discovery.

EO_ARTIFACT_ROOT selects an absolute run-specific root, inherited by MCP
subprocesses. Relative output/tool_output paths resolve under that root;
absolute paths remain explicit overrides for existing files. Evaluators prefer
hash-checked candidate_artifacts inventories over mutable trace paths.
"""

from __future__ import annotations

from pathlib import Path
import os

from agenticeoecosystem.definitions import ROOT_DIR

_MISC_SUBDIR = "misc"


def artifact_root() -> Path:
    """Return the explicit run artifact root, independent of installed package location."""
    return Path(os.environ.get("EO_ARTIFACT_ROOT", str(ROOT_DIR / "tool_output"))).resolve()


def server_output_dir(server: str, legacy_env: str | None = None) -> Path:
    """Use the common root; retain server overrides for standalone legacy launches."""
    if "EO_ARTIFACT_ROOT" not in os.environ and legacy_env and os.environ.get(legacy_env):
        return Path(os.environ[legacy_env]).resolve()
    return artifact_root() / server


def candidate_output_roots() -> tuple[Path, ...]:
    """Roots for exact trace-path discovery; archived inventories take precedence."""
    return (artifact_root(), ROOT_DIR / "output", ROOT_DIR / "tool_output")


def resolve_run_path(path_str: str) -> Path:
    """Resolve relative artifact paths under the shared run root.

    Strip legacy output/tool_output prefixes; put bare filenames under misc.
    Preserve absolute input or explicit output paths and create parents.
    """
    p = Path(path_str)
    if not p.is_absolute():
        if p.parts and p.parts[0] in {"output", "tool_output"}:
            p = artifact_root().joinpath(*p.parts[1:])
        else:
            p = artifact_root() / _MISC_SUBDIR / p
        if not p.resolve().is_relative_to(artifact_root()):
            raise ValueError("Relative artifact path escapes the run artifact root")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
