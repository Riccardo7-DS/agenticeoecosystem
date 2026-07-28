"""Shared output-path resolution for MCP servers.

Every MCP server used in the benchmark (cdse_geocode, eumetsat, gee, climate)
used to decide where to write/read files its own way — some read a
server-specific env var with a hardcoded fallback, one was fully hardcoded to
a folder inside its own package, and others just resolved whatever path
string the caller passed against the subprocess's current working directory.
Since caller-supplied paths are the only thing that varies per benchmark
task (all four servers are spawned once as long-lived subprocesses and
reused for an entire run, not restarted per task), standardizing where data
lands means standardizing how those path strings are resolved, not adding a
new per-task env var.

Convention: reference solutions in data/tasks/benchmark_tasks.json use
relative paths of the form "output/<task_ID>/<filename>". This helper
resolves any relative path against the project root (ROOT_DIR), so that
string always lands at the same place regardless of the subprocess's CWD.
Absolute paths pass through unchanged, as an explicit override.

Agents don't always follow that convention — a model will sometimes pass a
bare, human-friendly filename (e.g. "milano_no2_march2020.jpg") with no
"output/" prefix at all. Resolved naively against ROOT_DIR, that lands
directly in the repo root instead of under output/. To close that gap,
any relative path whose first segment isn't "output" is nested under
"output/misc/" instead, so nothing a caller passes can ever escape the
output tree.
"""

from __future__ import annotations

from pathlib import Path

from agenticeoecosystem.definitions import ROOT_DIR

_MISC_SUBDIR = "misc"


def resolve_run_path(path_str: str) -> Path:
    """Resolve a caller-supplied output/input path consistently.

    Relative paths are resolved against ROOT_DIR (so "output/<task_ID>/x.tif"
    always lands at "<repo_root>/output/<task_ID>/x.tif", regardless of the
    server subprocess's working directory). A relative path that doesn't
    already start with "output" is nested under "output/misc/" instead of
    landing at the repo root. Absolute paths are returned as-is. Parent
    directories are created if missing.
    """
    p = Path(path_str)
    if not p.is_absolute():
        if p.parts and p.parts[0] != "output":
            p = Path("output") / _MISC_SUBDIR / p
        p = ROOT_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
