#!/usr/bin/env python
"""
build_table.py — Aggregate all results_*.json into a comparison CSV/HTML table.

Usage:
    python scripts/build_table.py --output-dir output/geobenchX
    python scripts/build_table.py --output-dir output/geobenchX --table-filename my_comparison
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build comparison table from results_*.json files")
    parser.add_argument("--output-dir",     required=True,            help="Directory containing results_*.json files")
    parser.add_argument("--table-filename", default="model_comparison", help="Base name for .csv and .html outputs")
    args = parser.parse_args()

    output_dir   = Path(args.output_dir)
    result_files = sorted(output_dir.glob("results_*.json"))

    if not result_files:
        print(f"No results_*.json files found in {output_dir}", file=sys.stderr)
        return 1

    from agenticeoecosystem.evaluation.dataclasses import TaskSet

    rows = []
    for f in result_files:
        label = f.stem[len("results_"):]  # strip "results_" prefix → model label
        tasks = TaskSet.read_from_file(f.name, folder=str(output_dir))

        solved       = sum(1 for t in tasks if t.generated_solution is not None)
        n_steps_list = [len(t.generated_solution) for t in tasks if t.generated_solution]
        avg_steps    = sum(n_steps_list) / len(n_steps_list) if n_steps_list else 0.0

        row: dict = {
            "model":               label,
            "tasks_attempted":     solved,
            "avg_steps":           round(avg_steps, 1),
            "total_input_tokens":  tasks.metadata.get("total_input_tokens", 0),
            "total_output_tokens": tasks.metadata.get("total_output_tokens", 0),
        }

        # Collect evaluator labels used across all tasks in this file
        evaluator_labels: set[str] = set()
        for t in tasks:
            evaluator_labels.update(t.evaluator_scores.keys())

        for elabel in sorted(evaluator_labels):
            ev = tasks.score_summary_for_evaluator(elabel)
            row[f"evaluated [{elabel}]"]             = ev.get("n", 0)
            row[f"match (2) [{elabel}]"]             = ev.get("match", 0)
            row[f"partial (1) [{elabel}]"]           = ev.get("partial", 0)
            row[f"no_match (0) [{elabel}]"]          = ev.get("no_match", 0)
            row[f"accuracy_full [{elabel}]"]         = round(ev.get("accuracy_full", 0.0), 3)
            row[f"accuracy_partial+full [{elabel}]"] = round(ev.get("accuracy_partial_or_full", 0.0), 3)

        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    csv_path  = output_dir / f"{args.table_filename}.csv"
    html_path = output_dir / f"{args.table_filename}.html"
    df.to_csv(csv_path,  index=False)
    df.to_html(html_path, index=False)

    print(f"Table saved → {csv_path} (CSV + HTML)\n")
    print(df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
