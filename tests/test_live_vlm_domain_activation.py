"""
Live integration test for the VLM_DOMAIN activation convention added in
sub_agent_vlm.py / vlm_domains. Drives the real vlm_agent end to end — real
Qwen3-VL vision endpoint (see vlm_init.py), real MCP vlm_tools server
(resolve_images/load_images), real sample images — no mocks.

This is the integration-level counterpart to tests/test_vlm_domains.py
(which only tests prompt-composition logic, no LLM). It checks the property
unit tests cannot: does the real model actually (a) apply wildfire-specific
guidance when the handoff message it inherits contains `VLM_DOMAIN:
wildfire`, (b) stay generic when no marker is present, even when asked
about an out-of-domain condition (flood — no such domain is registered),
and (c) stay generic on the SAME wildfire-relevant image when the question
asked doesn't request domain analysis and no marker is present — i.e.
activation is per-request, not per-image.

Talks to vlm_agent directly (its own Runner/session) rather than going
through the full multi_agent.py orchestrator. That's a deliberate choice:
this orchestrator's tool-calling reliability depends on whatever model is
currently deployed on the shared EVE gateway, which is infrastructure this
test doesn't own or control (at the time this was written, the gateway had
only a reasoning-distill model loaded, not the intended EVE-Instruct, and
it did not reliably emit the `transfer_to_agent` call at all — a real but
separate finding from what this test checks). Talking to vlm_agent directly
isolates the actual thing being verified: whether the VLM_DOMAIN convention
this session's code composes and reads is respected by the model, given
the same kind of handoff message an orchestrator would send.

Assertions are keyword-based sanity checks on real model output — real LLM
text isn't byte-stable, so this checks for the presence of expected
vocabulary rather than an exact match. At temperature=0 (see vlm_init.py)
this has been stable across repeated runs, but a flaky failure here should
be read as "check the printed response," not necessarily as a regression.

Requires a reachable Qwen3-VL endpoint (VLM_BASE_URL in vlm_init.py — by
default an SSH-tunneled localhost:18001) and the two untracked sample
images below to exist in this checkout's working tree (gitignored, not
version-controlled — see .gitignore's `data/visual_tasks/images/` /
`output/` entries for why sample imagery in this repo is never committed).

Run:
    AGENTICEO_SUBAGENTS=vlm uv run python tests/test_live_vlm_domain_activation.py
"""

import asyncio
import logging
import os
import uuid

os.environ.setdefault("AGENTICEO_SUBAGENTS", "vlm")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agenticeoecosystem.utils.logging import setup_logging

setup_logging()
logging.getLogger("agenticeoecosystem.agents.subagents.sub_agent_vlm").setLevel(logging.INFO)

from agenticeoecosystem.agents.subagents.sub_agent_vlm import vlm_agent

# NOTE: untracked (gitignored) sample images already present in this repo's
# working tree — see the module docstring. Overridable for a checkout where
# they live elsewhere (e.g. a worktree without its own copy of `output/`).
WILDFIRE_IMAGE = os.environ.get(
    "TEST_WILDFIRE_IMAGE",
    "/shared/home/rdercole/Riccardo/projects/agenticeoecosystem/output/misc/lnu_lightning_complex_postfire.jpg",
)
NON_FIRE_IMAGE = os.environ.get(
    "TEST_NON_FIRE_IMAGE",
    "/shared/home/rdercole/Riccardo/projects/agenticeoecosystem/output/misc/milano_no2_march2020.jpg",
)

APP_NAME = "vlm-domain-live-test"
USER_ID = "tester"

session_service = InMemorySessionService()
runner = Runner(agent=vlm_agent, app_name=APP_NAME, session_service=session_service)


async def _ask(label: str, message: str) -> str:
    session_id = f"s_{uuid.uuid4().hex}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final_text = None
    print(f"\n{'=' * 80}\nCASE: {label}\nMESSAGE:\n{message}\n{'=' * 80}")
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts)
    print(f"FINAL VLM RESPONSE:\n{final_text}\n")
    return final_text or ""


async def main():
    for path in (WILDFIRE_IMAGE, NON_FIRE_IMAGE):
        assert os.path.isfile(path), f"missing test image: {path}"

    in_domain = await _ask(
        "in_domain_wildfire",
        f"VLM_DOMAIN: wildfire\n"
        f"Analyze the image at {WILDFIRE_IMAGE} and report whether there is "
        f"a wildfire (active fire, smoke, or burn scar) visible in it.",
    )
    out_of_domain = await _ask(
        "out_of_domain_flood_no_marker",
        f"Analyze the image at {NON_FIRE_IMAGE} and report whether there is "
        f"a flood visible in it.",
    )
    generic_on_fire_image = await _ask(
        "wildfire_image_generic_question_no_marker",
        f"Analyze the image at {WILDFIRE_IMAGE} and describe, in general "
        f"terms, what kind of land cover is visible (vegetation, urban, "
        f"water, bare soil, etc.) — no need to comment on fire.",
    )

    checks = {
        "in_domain response applies wildfire guidance": any(
            w in in_domain.lower() for w in ("burn", "scar", "fire", "smoke")
        ),
        "out_of_domain response stays free of wildfire vocabulary": not any(
            w in out_of_domain.lower() for w in ("burn scar", "wildfire", "flame front")
        ),
        "generic question on the SAME wildfire image doesn't self-trigger "
        "wildfire framing (activation is per-request, not per-image)": not any(
            w in generic_on_fire_image.lower()
            for w in ("burn scar", "wildfire", "flame front", "active fire")
        ),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    failures = 0
    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")
        failures += not passed
    if failures:
        raise SystemExit(f"{failures}/{len(checks)} check(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
