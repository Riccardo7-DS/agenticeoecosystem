#!/usr/bin/env python
"""Run one question through the research orchestrator. Hypotheses stay internal;
the ledger and review reports are written under <output-dir>/helper/<run_id>/."""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from agenticeoecosystem.utils.logging import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="Run one research question")
    p.add_argument("--question", required=True)
    p.add_argument("--model", default=None, help="Overrides MULTI_AGENT_MODEL")
    p.add_argument("--provider", default=None, help="Overrides MULTI_AGENT_PROVIDER")
    p.add_argument("--output-dir", default=f"output/research/run_{time.strftime('%Y%m%d_%H%M%S')}")
    args = p.parse_args()

    # All env before importing any agent module: llm_init and MCP subprocesses read it at import.
    if args.model:
        os.environ["MULTI_AGENT_MODEL"] = args.model
    if args.provider:
        os.environ["MULTI_AGENT_PROVIDER"] = args.provider
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["EO_ARTIFACT_ROOT"] = str(out / "artifacts")

    setup_logging(log_file="research")
    import nest_asyncio
    nest_asyncio.apply()
    import agenticeoecosystem.agents.multi_agent_research as research
    from agenticeoecosystem.agents import helper

    loop = asyncio.get_event_loop()
    try:
        raw = loop.run_until_complete(research.get_ans_tool_path(args.question))
    finally:
        helper.finish_task(loop)

    data = json.loads(raw)
    (out / "result.json").write_text(json.dumps(data, indent=1, default=str))
    print(data["final_answer"])
    print(f"\nResult: {out / 'result.json'}\nReports and hypotheses: {out / 'helper'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())