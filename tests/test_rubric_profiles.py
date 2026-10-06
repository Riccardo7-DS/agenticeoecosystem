"""Offline checks for rubric profiles without loading candidate agents or models."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RubricProfileTests(unittest.TestCase):
    """Exercise real scoring/evidence code with stubbed external dependencies."""

    @classmethod
    def setUpClass(cls) -> None:
        package = ModuleType("rubric_profile_test")
        package.__path__ = [str(ROOT / "src/agenticeoecosystem/evaluation")]
        schema = ModuleType("rubric_profile_test.dataclasses")
        schema.Task = SimpleNamespace
        schema.TaskSet = list
        constants = ModuleType("rubric_profile_test.constants")
        constants.MODEL_CLAUDE_SONNET = "offline"
        tqdm = ModuleType("tqdm")
        tqdm.tqdm = lambda tasks, **kwargs: tasks
        cls.modules = patch.dict(sys.modules, {
            "rubric_profile_test": package,
            "rubric_profile_test.dataclasses": schema,
            "rubric_profile_test.constants": constants,
            "tqdm": tqdm,
        })
        cls.modules.start()
        spec = importlib.util.spec_from_file_location(
            "rubric_profile_test.rubric_evaluator",
            ROOT / "src/agenticeoecosystem/evaluation/rubric_evaluator.py",
        )
        cls.evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.evaluator)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.modules.stop()

    def setUp(self) -> None:
        entry = json.loads((ROOT / "data/tasks/rubrics_108q_v4_trace.json").read_text())[1]
        self.criteria = entry["rubric"]["criteria"]
        self.protocol = {key: entry["rubric"][key] for key in (
            "protocol_version", "evidence_mode", "scoring_instructions")}
        step = SimpleNamespace(function_name="fetch_sentinel2", arguments={"bbox": [12, 41, 13, 42]},
                               tool_result={"hidden_response": "response-only-secret"})
        solution = SimpleNamespace(steps=[step], final_answer="The river is visible.")
        solution.model_dump = lambda: {"answer": solution.final_answer, "steps": [vars(step)]}
        self.task = SimpleNamespace(
            task_ID="rome", task_text="Assess the Rome scene.", generated_solution=solution,
            candidate_artifacts=[], evaluator_scores={}, evaluator_scores_raw={},
            evaluator_criteria={}, evaluator_evidence={}, evaluator_reasoning={},
        )

    def score(self, scores: tuple[str, str, str] = ("2", "2", "1")) -> str:
        """Run the scorer against a deterministic mock judge response."""
        prompts = []
        response = "".join(
            f"<C{i}_SCORE>{score}</C{i}_SCORE><C{i}_REASONING>Reason.</C{i}_REASONING>"
            for i, score in enumerate(scores, 1)
        )

        def create(**kwargs: object) -> object:
            prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

        credentials = ModuleType("agenticeoecosystem.config.credentials")
        credentials.get_openai_client = lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        langchain = ModuleType("langchain.chat_models")
        langchain.init_chat_model = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {
            "agenticeoecosystem.config.credentials": credentials,
            "langchain.chat_models": langchain,
        }):
            self.evaluator.rubric_score_task_solution(
                self.task, self.criteria, provider="eve", rubric_protocol=self.protocol)
        return prompts[0]

    def test_trace_scores_missing_responses_and_only_supplies_calls(self) -> None:
        prompt = self.score()
        self.assertIn("fetch_sentinel2", prompt)
        self.assertIn("bbox", prompt)
        self.assertNotIn("response-only-secret", prompt)
        self.assertNotIn("<CANDIDATE_EVIDENCE>", prompt)
        self.assertAlmostEqual(self.task.evaluator_scores_raw["rubric_judge"], 1.8)
        context = self.task.evaluator_evidence["rubric_judge"]
        self.assertEqual(context["verification_status"], "not_performed")
        self.assertEqual(context["scoring_protocol"]["evidence_mode"], "trace")

    def test_evidence_profile_keeps_missing_grounding_unverified(self) -> None:
        self.protocol["evidence_mode"] = "evidence"
        self.task.generated_solution.steps[0].tool_result = None
        self.score()
        self.assertIsNone(self.task.evaluator_scores_raw["rubric_judge"])
        self.assertIsNone(self.task.evaluator_criteria["rubric_judge"][1]["score"])

    def test_v4_explicit_failure_zero_survives_missing_evidence(self) -> None:
        self.protocol["evidence_mode"] = "evidence"
        self.task.generated_solution.steps[0].tool_result = None
        self.task.generated_solution.final_answer = "Download failed; no image or result was produced."
        self.score(("0", "0", "0"))
        self.assertEqual(self.task.evaluator_scores_raw["rubric_judge"], 0)

    def test_legacy_missing_evidence_behavior_is_preserved(self) -> None:
        self.protocol = None
        self.task.generated_solution.steps[0].tool_result = None
        self.score()
        self.assertIsNone(self.task.evaluator_scores_raw["rubric_judge"])
        self.assertNotIn("scoring_protocol", self.task.evaluator_evidence["rubric_judge"])

    def test_v4_missing_solution_receives_zero_without_judge_call(self) -> None:
        self.task.generated_solution = None
        self.evaluator.rubric_evaluate_solutions(
            [self.task], {self.task.task_ID: self.criteria},
            rubric_protocols={self.task.task_ID: self.protocol},
        )
        self.assertEqual(self.task.evaluator_scores_raw["rubric_judge"], 0)

    def test_rubric_changes_invalidate_score_context(self) -> None:
        before = self.evaluator._rubric_context(self.task, None, self.criteria, self.protocol)
        self.criteria[0]["description"] += " Changed expectation."
        after = self.evaluator._rubric_context(self.task, None, self.criteria, self.protocol)
        self.assertNotEqual(before["scoring_protocol"], after["scoring_protocol"])

    def test_profiles_preserve_questions_weights_and_evidence_descriptions(self) -> None:
        source = json.loads((ROOT / "data/tasks/rubrics_108q_v3.json").read_text())
        for mode in ("trace", "evidence"):
            versioned = json.loads((ROOT / f"data/tasks/rubrics_108q_v4_{mode}.json").read_text())
            self.assertEqual(len(versioned), 108)
            for before, after in zip(source, versioned):
                self.assertEqual(before["task_ID"], after["task_ID"])
                self.assertEqual(before["task_text"], after["task_text"])
                self.assertEqual(after["rubric"]["evidence_mode"], mode)
                for a, b in zip(before["rubric"]["criteria"], after["rubric"]["criteria"]):
                    self.assertEqual(a["weight"], b["weight"])
                    self.assertEqual(a["type"], b["type"])
                    if mode == "evidence":
                        self.assertEqual(a["description"], b["description"])


if __name__ == "__main__":
    unittest.main()
