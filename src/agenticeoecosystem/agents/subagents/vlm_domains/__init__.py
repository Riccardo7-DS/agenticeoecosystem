"""Pluggable domain-knowledge packs for the shared VLM sub-agent.

Each module in this package (other than this one) declares a module-level
`DOMAIN: VLMDomain` describing one analysis domain (wildfire, flood, ...).
Modules are auto-discovered the same way `subagents/__init__.py` discovers
`sub_agent_*.py` files — adding a new domain is a matter of dropping a new
file here, no other file needs to change.

Activation is request-level and prompt-driven (no session state): an
orchestrator that wants domain-specific guidance applied includes a
`VLM_DOMAIN: <key>` marker line in the handoff message it sends immediately
before transferring control to the VLM sub-agent. The VLM's own instruction
(built in `sub_agent_vlm.py` from `render_domain_catalog`) tells it to look
for that marker in the message that triggered the handoff and apply the
matching domain's guidance, or fall back to generic analysis if the marker
is absent or unrecognized.

Caveat: because the instruction string is static (assembled once at import
time), every registered domain's guidance is always present in the VLM's
context — activation is a behavioral convention the model follows, not a
structural guarantee that off-domain requests never see domain text. See
the module docstring in `sub_agent_vlm.py` for the follow-up (session-state
driven) design that would give a hard guarantee instead.
"""

import importlib
import logging
import pkgutil
import re

from agenticeoecosystem.agents.subagents.vlm_domains._types import VLMDomain

logger = logging.getLogger(__name__)

__all__ = ["VLMDomain", "DOMAINS", "get_domain", "extract_domain_marker", "render_domain_catalog"]

_PACKAGE = __name__
_PKG_DIR = __path__[0]

DOMAINS: dict[str, VLMDomain] = {}

for _, _module_name, _ in pkgutil.iter_modules([_PKG_DIR]):
    if _module_name.startswith("_"):
        continue
    _module = importlib.import_module(f"{_PACKAGE}.{_module_name}")
    _domain = getattr(_module, "DOMAIN", None)
    if isinstance(_domain, VLMDomain):
        if _domain.key in DOMAINS:
            logger.warning(
                "Duplicate VLM domain key '%s' from module %s — keeping the "
                "first one registered.", _domain.key, _module_name,
            )
            continue
        DOMAINS[_domain.key] = _domain

_DOMAIN_MARKER_RE = re.compile(r"VLM_DOMAIN:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)


def get_domain(key: str) -> VLMDomain | None:
    """Look up a registered domain by key, case-insensitively.

    Returns None (and logs a warning) for an unknown key so callers can fall
    back to generic-only behavior instead of crashing.
    """
    domain = DOMAINS.get(key.lower())
    if domain is None:
        logger.warning(
            "Unknown VLM_DOMAIN '%s' requested — known domains: %s. "
            "Falling back to generic analysis.", key, sorted(DOMAINS),
        )
    return domain


def extract_domain_marker(text: str) -> str | None:
    """Pull a `VLM_DOMAIN: <key>` marker out of a block of text, if present."""
    if not text:
        return None
    match = _DOMAIN_MARKER_RE.search(text)
    return match.group(1) if match else None


def render_domain_catalog() -> str:
    """Render the 'available domains' section injected into the VLM prompt."""
    if not DOMAINS:
        return ""
    blocks = []
    for domain in DOMAINS.values():
        blocks.append(
            f"=== {domain.key} ===\n"
            f"Trigger: {domain.trigger_hint}\n\n"
            f"{domain.guidance.strip()}"
        )
    return "\n\n".join(blocks)
