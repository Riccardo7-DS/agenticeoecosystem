"""Deterministic prefix/subsequence metrics against accepted reference workflows."""

from __future__ import annotations

import json
from typing import Any, Sequence

from ..dataclasses import Solution

TRAJECTORY_METHODS = {
    "tools_in_order", "tool_exact_match", "agents_in_order",
    "agent_exact_match", "parameter_accuracy", "tool_efficiency",
}


class AgentTrajectoryMapper:
    """Map tool calls without inventing handoffs or silently dropping unknown tools."""

    def __init__(self, mapping: dict[str, Any]) -> None:
        self.tools = mapping.get("tools", {})

    def derive(self, solution: Solution) -> dict[str, Any]:
        """Return per-call owners, collapsed segments, and unresolved calls."""
        per_tool: list[str | None] = []
        unresolved = []
        for index, step in enumerate(solution.steps):
            owners = self.tools.get(step.function_name, {}).get("agents", [])
            owner = owners[0] if len(owners) == 1 else None
            per_tool.append(owner)
            if owner is None:
                unresolved.append({"step_index": index, "tool": step.function_name,
                                   "possible_agents": owners})
        segments: list[str | None] = []
        for owner in per_tool:
            if not segments or owner != segments[-1]:
                segments.append(owner)
        return {"status": "mapped" if not unresolved else "unverified",
                "basis": "tool_ownership", "tool_sequence": [s.function_name for s in solution.steps],
                "per_tool_agents": per_tool, "agent_sequence": segments,
                "unresolved_tools": unresolved, "recorded_agents_called": solution.agents_called}


def prefix_length(predicted: Sequence[Any], reference: Sequence[Any]) -> int:
    """Count consecutive matching items from the beginning."""
    length = 0
    for actual, expected in zip(predicted, reference):
        if actual != expected:
            break
        length += 1
    return length


def ordered_prefix_length(predicted: Sequence[Any], reference: Sequence[Any]) -> int:
    """Match a reference prefix as a subsequence, never skipping reference items."""
    length = 0
    for actual in predicted:
        if length < len(reference) and actual == reference[length]:
            length += 1
    return length


class TrajectoryJudge:
    """Score each alignment against its best accepted nonempty reference."""

    def __init__(self, mapping: dict[str, Any] | None = None) -> None:
        self.mapper = AgentTrajectoryMapper(mapping) if mapping is not None else None

    def evaluate(self, predicted: Solution, references: list[Solution]) -> dict[str, dict[str, Any]]:
        """Return values, denominators and reference indices, or explicit coverage gaps."""
        results = {name: {"status": "unverified", "reason": "No nonempty reference trajectory"}
                   for name in TRAJECTORY_METHODS}
        tools = [step.function_name for step in predicted.steps]
        parameters = [(step.function_name, json.dumps(step.arguments, sort_keys=True,
                       ensure_ascii=False, allow_nan=False)) for step in predicted.steps]
        tool_references = [(index, ref) for index, ref in enumerate(references) if ref.steps]
        for index, reference in tool_references:
            expected = [step.function_name for step in reference.steps]
            expected_params = [(step.function_name, json.dumps(step.arguments, sort_keys=True,
                                ensure_ascii=False, allow_nan=False)) for step in reference.steps]
            counts = {
                "tools_in_order": ordered_prefix_length(tools, expected),
                "tool_exact_match": prefix_length(tools, expected),
                "parameter_accuracy": prefix_length(parameters, expected_params),
            }
            for name, count in counts.items():
                value = count / len(expected)
                if results[name]["status"] != "scored" or value > results[name]["value"]:
                    results[name] = {"status": "scored", "value": value, "matched": count,
                                     "reference_length": len(expected), "reference_index": index}
        if tool_references:
            index, reference = min(tool_references, key=lambda entry: len(entry[1].steps))
            results["tool_efficiency"] = {
                "status": "scored", "value": len(tools) / len(reference.steps),
                "predicted_length": len(tools), "reference_length": len(reference.steps),
                "reference_index": index, "direction": "lower_is_better",
                "interpretation": "Call-count ratio; interpret conditional on task success",
            }
        for name in ("agents_in_order", "agent_exact_match"):
            results[name] = {"status": "unverified", "reason": "No nonempty reference agent sequence"}
        if self.mapper is not None:
            prediction = self.mapper.derive(predicted)
            for name in ("agents_in_order", "agent_exact_match"):
                results[name] = {"status": "unverified", "basis": "tool_ownership",
                                 "reason": "No fully mapped nonempty reference agent sequence"}
            if prediction["status"] != "mapped":
                for name in ("agents_in_order", "agent_exact_match"):
                    results[name].update(reason="Candidate has unresolved tool ownership",
                                         unresolved_tools=prediction["unresolved_tools"])
                return results
            for index, reference in enumerate(references):
                mapped = self.mapper.derive(reference)
                agents = mapped["agent_sequence"]
                if mapped["status"] != "mapped" or not agents:
                    continue
                for name, matcher in (("agents_in_order", ordered_prefix_length),
                                      ("agent_exact_match", prefix_length)):
                    count = matcher(prediction["agent_sequence"], agents)
                    value = count / len(agents)
                    if results[name]["status"] != "scored" or value > results[name]["value"]:
                        results[name] = {"status": "scored", "value": value, "matched": count,
                                         "basis": "tool_ownership", "reference_length": len(agents),
                                         "reference_index": index,
                                         "predicted_sequence": prediction["agent_sequence"],
                                         "reference_sequence": agents}
            return results
        if predicted.agents_called is None:
            for name in ("agents_in_order", "agent_exact_match"):
                results[name] = {"status": "unverified", "reason": "Candidate agent activations were not recorded"}
            return results
        for index, reference in enumerate(references):
            agents = reference.agents_called
            if not agents:
                continue
            for name, matcher in (("agents_in_order", ordered_prefix_length),
                                  ("agent_exact_match", prefix_length)):
                count = matcher(predicted.agents_called, agents)
                value = count / len(agents)
                if results[name]["status"] != "scored" or value > results[name]["value"]:
                    results[name] = {"status": "scored", "value": value, "matched": count,
                                     "reference_length": len(agents), "reference_index": index}
        return results
