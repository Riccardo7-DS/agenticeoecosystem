"""
Unit tests for the VLM sub-agent's pluggable domain-knowledge mechanism
(`agenticeoecosystem.agents.subagents.vlm_domains` and the prompt built in
`sub_agent_vlm.py`).

These test the prompt-composition logic in isolation — no LLM calls, no MCP
servers. What they deliberately do NOT and CANNOT verify is whether a real
orchestrator run actually emits the `VLM_DOMAIN: <key>` marker at the right
time, or whether the VLM model correctly scopes a domain to only the turn
that requested it — that's a property of the model's behavior over inherited
conversation history (Option A activation), not of this code, and needs a
live integration run (e.g. driving `multi_agent_wildfires` end to end, the
way tests/test_eumetsat_fci_query.py drives the main orchestrator) to check.

Run:
    AGENTICEO_SUBAGENTS=vlm uv run python tests/test_vlm_domains.py
"""

import os

os.environ.setdefault("AGENTICEO_SUBAGENTS", "vlm")

from google.genai import types

from agenticeoecosystem.agents.subagents import vlm_domains
from agenticeoecosystem.agents.subagents import sub_agent_vlm


def _fake_request(text: str):
    class _FakeRequest:
        contents = [types.Content(role="user", parts=[types.Part(text=text)])]

    return _FakeRequest()


def test_wildfire_domain_is_registered():
    assert "wildfire" in vlm_domains.DOMAINS
    domain = vlm_domains.get_domain("wildfire")
    assert domain is not None
    assert domain.key == "wildfire"
    assert domain.defer_to_orchestrator is True


def test_get_domain_is_case_insensitive():
    assert vlm_domains.get_domain("WILDFIRE") is vlm_domains.get_domain("wildfire")


def test_unknown_domain_falls_back_safely():
    # Must not raise, must return None so callers fall back to generic-only.
    assert vlm_domains.get_domain("flood") is None
    assert vlm_domains.get_domain("") is None


def test_extract_domain_marker():
    assert vlm_domains.extract_domain_marker("VLM_DOMAIN: wildfire") == "wildfire"
    assert (
        vlm_domains.extract_domain_marker(
            "some preamble\nVLM_DOMAIN:wildfire\nrest of the request"
        )
        == "wildfire"
    )
    assert vlm_domains.extract_domain_marker("just a generic request") is None
    assert vlm_domains.extract_domain_marker("") is None


def test_base_prompt_has_no_domain_specific_content():
    # The generic base prompt (what a request gets when no domain is
    # registered/activated) must stand on its own and stay domain-agnostic.
    base = sub_agent_vlm._VLM_BASE_PROMPT
    assert "load_images" in base
    assert "resolve_images" in base
    for leaked_term in ("wildfire", "smoke", "burn scar", "FEATURES ANALYSIS"):
        assert leaked_term.lower() not in base.lower(), (
            f"generic base prompt should not mention '{leaked_term}'"
        )


def test_composed_instruction_includes_registered_domains():
    composed = sub_agent_vlm._VLM_SYSTEM_PROMPT
    assert "load_images" in composed  # base rules still present
    assert "VLM_DOMAIN" in composed  # activation convention explained
    assert "wildfire" in composed.lower()  # wildfire domain content present
    assert "burn scar" in composed.lower()


def test_render_domain_catalog_empty_when_no_domains():
    assert vlm_domains.render_domain_catalog() != ""
    # sanity: rendering with an empty registry produces an empty catalog,
    # which is what makes `_build_vlm_instruction` fall back to the bare
    # base prompt when no domains are registered at all.
    assert vlm_domains.DOMAINS  # (guard: this repo always has >=1 domain)


def test_domain_marker_callback_does_not_raise_on_known_and_unknown_and_absent():
    # Pure smoke test of the before_model_callback wiring: it must never
    # raise or mutate the request, regardless of what the last turn says.
    for text in ("VLM_DOMAIN: wildfire\ncheck this image", "VLM_DOMAIN: flood\n...", "plain request", ""):
        result = sub_agent_vlm._log_domain_marker(None, _fake_request(text))
        assert result is None  # never short-circuits the model call


if __name__ == "__main__":
    tests = [obj for name, obj in globals().items() if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failures:
        raise SystemExit(f"{failures}/{len(tests)} test(s) failed")
    print(f"All {len(tests)} tests passed.")
