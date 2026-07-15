#!/usr/bin/env python3
"""Analyze benchmark task coverage and MCP tool completeness.

Usage:
    python analyze_coverage.py <project_root>

Output (stdout): JSON with two top-level keys:

  "task_coverage": {
    "function_name": {
      "count": N,          # total tasks referencing this fn (across all files)
      "active_count": N,   # tasks NOT labeled Placeholder
      "categories": [...],
      "has_placeholder": bool,
      "task_ids": [...],
      "placeholder_task_ids": [...]
    },
    ...
    "_files_scanned": [...],
    "_total_tasks": N
  }

  "mcp_tool_coverage": {
    "server_key": {
      "source": "local" | "external",
      "tools": {
        "tool_name": {
          "active_task_count": N,
          "total_task_count": N,
          "categories": [...],
          "covered": bool      # True if active_task_count > 0
        },
        ...
      },
      "all_covered": bool,
      "uncovered_tools": [...]
    },
    ...
  }
"""

import ast
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Known MCP server → Python file mapping (relative to project root)
# Third-party servers where we can't read the source are marked "external".
# ---------------------------------------------------------------------------
_SERVER_FILES = {
    "eodashboard":   "src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py",
    "gee-mcp":       "src/agenticeoecosystem/servers/gee/server.py",
    "moocs":         "src/agenticeoecosystem/servers/moocs/server.py",
    "eoss_graphrag": "src/eoss_graphrag/05_mcp_server.py",
    "geos_ops":      "src/agenticeoecosystem/servers/eumetsat/server.py",
    "climate":       "src/agenticeoecosystem/servers/climate/server.py",
    "cdse":          "src/agenticeoecosystem/servers/cdse_geocode/server.py",
}
_EXTERNAL_SERVERS = ["arxiv-mcp-server", "gis-mcp", "eve_retrieval"]


def _extract_tools_from_file(path: Path) -> list[str]:
    """Return names of all @mcp.tool()-decorated functions in a Python file."""
    try:
        tree = ast.parse(path.read_text())
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
        return []

    tools = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            # matches @mcp.tool(), @mcp.tool, @app.tool(), @app.tool
            is_mcp_tool = (
                (isinstance(deco, ast.Attribute) and deco.attr == "tool")
                or (isinstance(deco, ast.Call)
                    and isinstance(deco.func, ast.Attribute)
                    and deco.func.attr == "tool")
            )
            if is_mcp_tool:
                # Use name= kwarg when present (MCP registered name may differ from fn name)
                mcp_name = node.name
                if isinstance(deco, ast.Call):
                    for kw in deco.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            mcp_name = kw.value.value
                            break
                tools.append(mcp_name)
                break
    return tools


_MANIFEST_FILE = Path(__file__).parent / "external_tools.json"


def _load_external_manifest() -> dict[str, list[str]]:
    """Load external server → tool names from the JSON manifest."""
    if _MANIFEST_FILE.exists():
        return json.loads(_MANIFEST_FILE.read_text())
    return {}


def build_mcp_inventory(project_root: Path) -> dict:
    """Return {server_key: {"source": ..., "tool_names": [...]}}."""
    inventory = {}
    for server, rel_path in _SERVER_FILES.items():
        fpath = project_root / rel_path
        if fpath.exists():
            inventory[server] = {
                "source": "local",
                "tool_names": _extract_tools_from_file(fpath),
            }
        else:
            inventory[server] = {"source": "local", "tool_names": [], "error": "file not found"}

    manifest = _load_external_manifest()
    for server in _EXTERNAL_SERVERS:
        inventory[server] = {
            "source": "external",
            "tool_names": manifest.get(server, []),
        }
    return inventory


_LEGACY_FILES = {"eo_benchmark_tasks.json", "arxiv_gis_benchmark_tasks.json"}


def build_task_coverage(tasks_dir: Path) -> dict:
    """Scan benchmark task files and return per-function-name coverage stats.

    Reads benchmark_tasks.json when it exists (the merged canonical file) and
    skips the legacy source files to avoid double-counting. Falls back to all
    JSON files in the directory when no merged file is present.
    """
    coverage: dict = {}
    total_tasks = 0
    files_scanned = []

    merged = tasks_dir / "benchmark_tasks.json"
    if merged.exists():
        task_files = [merged]
    else:
        task_files = [f for f in sorted(tasks_dir.glob("*.json"))
                      if f.name not in _LEGACY_FILES]
    for task_file in task_files:
        files_scanned.append(str(task_file))
        try:
            data = json.loads(task_file.read_text())
        except Exception as e:
            print(f"Warning: could not parse {task_file}: {e}", file=sys.stderr)
            continue

        tasks = data.get("tasks", [])
        total_tasks += len(tasks)

        for task in tasks:
            task_id = task.get("task_ID", "unknown")
            labels = task.get("task_labels") or []
            is_placeholder = "Placeholder" in labels

            ref_solutions = task.get("reference_solutions") or []
            seen_in_task: set[str] = set()
            for solution in ref_solutions:
                for step in solution.get("steps", []):
                    fn = step.get("function_name")
                    if not fn or fn in seen_in_task:
                        continue
                    seen_in_task.add(fn)

                    if fn not in coverage:
                        coverage[fn] = {
                            "count": 0,
                            "active_count": 0,
                            "categories": [],
                            "has_placeholder": False,
                            "task_ids": [],
                            "placeholder_task_ids": [],
                        }
                    entry = coverage[fn]
                    entry["count"] += 1
                    entry["task_ids"].append(task_id)
                    for lbl in labels:
                        if lbl not in entry["categories"]:
                            entry["categories"].append(lbl)
                    if is_placeholder:
                        entry["has_placeholder"] = True
                        entry["placeholder_task_ids"].append(task_id)
                    else:
                        entry["active_count"] += 1

    coverage["_files_scanned"] = files_scanned
    coverage["_total_tasks"] = total_tasks
    return coverage


def build_mcp_tool_coverage(inventory: dict, task_coverage: dict) -> dict:
    """Cross-reference MCP tool inventory with task coverage."""
    result = {}
    for server, info in inventory.items():
        tool_names = info["tool_names"]
        source = info["source"]
        tools_detail = {}

        for tool in tool_names:
            tc = task_coverage.get(tool, {})
            active = tc.get("active_count", 0)
            total = tc.get("count", 0)
            tools_detail[tool] = {
                "active_task_count": active,
                "total_task_count": total,
                "categories": tc.get("categories", []),
                "covered": active > 0,
            }

        uncovered = [t for t, d in tools_detail.items() if not d["covered"]]
        result[server] = {
            "source": source,
            "tools": tools_detail,
            "all_covered": len(uncovered) == 0 and len(tool_names) > 0,
            "uncovered_tools": uncovered,
        }
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_coverage.py <project_root>", file=sys.stderr)
        sys.exit(1)

    project_root = Path(sys.argv[1])
    tasks_dir = project_root / "data" / "tasks"

    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    inventory = build_mcp_inventory(project_root)
    task_coverage = build_task_coverage(tasks_dir)
    mcp_tool_coverage = build_mcp_tool_coverage(inventory, task_coverage)

    print(json.dumps({
        "task_coverage": task_coverage,
        "mcp_tool_coverage": mcp_tool_coverage,
    }, indent=2))


if __name__ == "__main__":
    main()
