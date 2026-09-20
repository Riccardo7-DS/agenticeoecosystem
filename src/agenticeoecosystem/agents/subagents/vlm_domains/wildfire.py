from agenticeoecosystem.agents.subagents.vlm_domains._types import VLMDomain

_GUIDANCE = """\
Apply this when asked about wildfires, smoke, or burned areas — only
visually observable cues apply, not thermal/SWIR/radar signals read
directly. This image set is pulled from live fire-alert satellite passes
and is NOT visually uniform from image to image — first work out which of
these three regimes the image is in, since what smoke and burn scar look
like differs between them:
- Clear daytime: ordinary natural colors — green/brown vegetation and soil,
  blue water, white puffy clouds. The majority case.
- Smoke-hazed daytime: thick smoke aerosols scatter and redden sunlight, so
  the whole scene gets an unnatural reddish-brown/maroon cast and both
  clouds AND smoke plumes render as pale lavender-white blooms that can look
  identical in color. Recognizable by that overall reddish/purplish tint
  across the whole image (not just one feature in it).
- Nighttime: land is almost pure black (no reflected sunlight) except for
  (a) small stippled clusters of bright dots tracing roads/rivers/towns —
  this is city/town lighting, not fire, cloud, or smoke — and (b) soft
  diffuse white/lavender blooms, which are clouds lit by moonlight/airglow.
  Burn scars are invisible at night; don't claim to see one in this regime.

A wildfire scene can show three independent visual features. When asked to
assess wildfire presence, explicitly consider each rather than giving a
single undifferentiated "yes, there's a wildfire" verdict:
- Active flame: a bright orange-red glow or visible flame front. This is
  rarely visible directly — most "fire-looking" content in these images is
  actually smoke or burn scar, not flame itself.
- Smoke / plume: in clear daytime imagery, a greyish-white/brownish hazy
  streak elongated downwind from a source. In smoke-hazed or nighttime
  imagery, smoke and ordinary clouds can be indistinguishable by color —
  fall back on shape: a plume trails from one or a few point sources in a
  wind-consistent comet-tail/fan shape, while ordinary clouds form broader,
  sourceless decks or scattered puffy cells with no common origin. If shape
  doesn't clearly settle it either, say the evidence is ambiguous rather
  than confidently calling it smoke or confidently dismissing it as "just
  clouds."
- Burn scar / burned area: only assess this in the clear-daytime regime. It
  looks like a distinct ash-grey or purplish-brown patch that clearly
  contrasts with the normally-colored (green/vegetated or differently-toned)
  land immediately around it in the same image, often following
  ridgelines/terrain rather than wind direction. The signal is *local
  contrast against nearby unburned land* — ordinary dark soil, water,
  shadow, or forest that covers the whole image uniformly is NOT a burn
  scar by itself. Don't call the generally darker/desaturated land under
  smoke haze, or black nighttime terrain, a burn scar — neither is one.

These features can co-occur (smoke over an already-burned area) or appear
alone (a burn scar with no remaining smoke, or smoke with no burn scar yet
visible). Do not default to describing everything as "smoke" — if what you
actually see is a scorched, locally-contrasting patch with no visible
plume, that is a burn scar, not smoke.

When you conclude that a burn scar is present, that IS a positive wildfire
finding by itself — a burn scar means a fire happened here, even if nothing
is burning right now. Don't require active flame or smoke on top of it
before you're willing to say a wildfire is evident, and don't let "the fire
isn't currently active" soften a clear burn-scar identification into a "no."

If this domain is active, state your findings in your own response text,
then transfer control back to the orchestrator instead of giving the user a
final answer yourself — the orchestrator owns the final output format for
this domain (e.g. a structured features/acquisition summary) and you have
no visibility into it, so don't attempt to reproduce it yourself.
"""

DOMAIN = VLMDomain(
    key="wildfire",
    trigger_hint=(
        "the handoff instruction asks about a wildfire, active fire, smoke "
        "plume, or burned/burn-scarred area in the image(s)"
    ),
    guidance=_GUIDANCE,
    defer_to_orchestrator=True,
)
