"""Lazy aliases for evaluation module paths used before the folder reorganization."""
import sys
from collections.abc import Sequence
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType

MODULE_ALIASES: dict[str, str] = {
    "agent_judge": "judges.agent_judge",
    "rubric_evaluator": "judges.rubric_evaluator",
    "numerical_judge": "judges.numerical_judge",
    "deterministic_judge": "judges.deterministic_judge",
    "numerical_metrics": "metrics.numerical_metrics",
    "candidate_evidence": "evidence.candidate_evidence",
    "gis_judge_agent": "evidence.gis_judge_agent",
    "full_evaluation": "runners.full_evaluation",
    "pipeline_runner": "runners.pipeline_runner",
    "save_results": "reporting.save_results",
    "evaluator": "legacy.evaluator",
    "model_comparison": "legacy.model_comparison",
}


class EvaluationModuleAliases(MetaPathFinder, Loader):
    """Resolve old module names to the same canonical module object."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Handle only explicitly registered historical evaluation modules."""
        prefix = __package__ + "."
        if fullname.startswith(prefix) and fullname[len(prefix):] in MODULE_ALIASES:
            return spec_from_loader(fullname, self)
        return None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        """Reuse the canonical module so historical patch paths still work."""
        name = spec.name[len(__package__) + 1:]
        return import_module(f"{__package__}.{MODULE_ALIASES[name]}")

    def exec_module(self, module: ModuleType) -> None:
        """The canonical module has already been executed by import_module."""


if not any(isinstance(finder, EvaluationModuleAliases) for finder in sys.meta_path):
    sys.meta_path.insert(0, EvaluationModuleAliases())
