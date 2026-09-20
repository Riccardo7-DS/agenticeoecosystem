import dataclasses


@dataclasses.dataclass(frozen=True)
class VLMDomain:
    key: str
    trigger_hint: str
    guidance: str
    defer_to_orchestrator: bool = False
