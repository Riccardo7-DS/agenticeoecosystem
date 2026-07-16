"""
Regression test for the MTG FCI FDHSI download+read pipeline.

Drives the full orchestrator with the request that originally surfaced a
chain of bugs (silent zip-extraction failure for chunked products, wrong
path/file detection for a downloaded product, and a dask-threaded-scheduler
segfault in satpy when reading ~40 FCI chunk files) and asserts the run
completes within a bounded time with a sane, non-fabricated answer.

Uses 1 October 2024 rather than the originally-requested 10 March 2024:
MTG-I1 FCI data genuinely does not exist yet on 10 March 2024 (confirmed via
search_eumetsat_products returning count: 0 for that date, full disk, no
bbox — first data appears 2024-10-01), so a test pinned to that date would
only ever exercise the "no data" branch and never touch the actual read path
this test is meant to cover.

Run:
    uv run python tests/test_eumetsat_fci_query.py
    uv run pytest tests/test_eumetsat_fci_query.py
"""

import asyncio
import json
import re

QUERY = (
    "Download the first available MTG FCI FDHSI slot over the Alps region on "
    "1 October 2024, then read that specific downloaded file and report "
    "reflectance statistics for its visible/near-infrared channels."
)

TIMEOUT_SECONDS = 600
# A healthy run needs ~6-10 calls (geocode, list_products, search, download,
# read). Anything well past that signals a retry/routing loop rather than
# genuine multi-step progress.
MAX_TOOL_CALLS = 20

_FAILURE_SIGNATURES = (
    "traceback",
    "error calling tool",
    "segmentation fault",
    "i'm sorry, but i don't have the capability",
    "i don't have the capability to access",
)


def test_fci_alps_query_completes_without_looping():
    from agenticeoecosystem.agents.multi_agent import get_ans_tool_path

    async def _run():
        return await asyncio.wait_for(get_ans_tool_path(QUERY), timeout=TIMEOUT_SECONDS)

    try:
        raw = asyncio.run(_run())
    except asyncio.TimeoutError:
        raise AssertionError(
            f"Query did not complete within {TIMEOUT_SECONDS}s — looks like a hang/loop."
        )

    result = json.loads(raw)
    final_answer = result["final_answer"]
    tools_called = result["tools_called"]

    assert len(tools_called) <= MAX_TOOL_CALLS, (
        f"{len(tools_called)} tool calls (> {MAX_TOOL_CALLS}) — looks like a "
        f"retry/routing loop: {tools_called}"
    )

    lowered = final_answer.lower()
    for bad in _FAILURE_SIGNATURES:
        assert bad not in lowered, f"Final answer contains a failure signature: {final_answer!r}"

    # Expect actual reflectance numbers, not just a narrative — this is the
    # exact failure mode we saw earlier (silent give-up, or a hallucinated
    # answer when the read pipeline was actually broken).
    assert re.search(r"\d+(\.\d+)?", final_answer), (
        f"No numeric reflectance values found in final answer: {final_answer!r}"
    )

    print(f"PASS — {len(tools_called)} tool calls: {tools_called}")
    print("Final answer:", final_answer)


if __name__ == "__main__":
    test_fci_alps_query_completes_without_looping()
