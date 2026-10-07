"""Offline regressions for source grounding, candidate isolation and GIS obligations."""

import importlib
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class JudgeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Avoid starting candidate agents through evaluation/__init__.py.
        package = ModuleType("agenticeoecosystem.evaluation")
        package.__all__ = []
        package.__path__ = [str(ROOT / "src/agenticeoecosystem/evaluation")]
        credentials = ModuleType("agenticeoecosystem.config.credentials")
        credentials.get_openai_client = lambda **kwargs: None
        cls.previous = {key: sys.modules.get(key) for key in (
            "agenticeoecosystem.evaluation", "agenticeoecosystem.config.credentials")}
        sys.modules["agenticeoecosystem.evaluation"] = package
        sys.modules["agenticeoecosystem.config.credentials"] = credentials
        cls.credentials = credentials
        cls.schema = importlib.import_module("agenticeoecosystem.evaluation.dataclasses")
        cls.evidence = importlib.import_module("agenticeoecosystem.evaluation.evidence.candidate_evidence")
        cls.rubric = importlib.import_module("agenticeoecosystem.evaluation.judges.rubric_evaluator")
        cls.aaj = importlib.import_module("agenticeoecosystem.evaluation.judges.agent_judge")
        cls.gis = importlib.import_module("agenticeoecosystem.evaluation.evidence.gis_judge_agent")

    @classmethod
    def tearDownClass(cls) -> None:
        for key, original in cls.previous.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.criteria = [{"type": name, "weight": weight, "description": name}
                         for name, weight in zip(
                             ("scientific_correctness", "grounded_and_complete", "internal_consistency"),
                             (.5, .3, .2))]

    def tearDown(self) -> None:
        self.directory.cleanup()

    def task(self, task_id: str = "test", result: object = None) -> object:
        return self.schema.Task(task_ID=task_id, task_text="Interpret the temperature series.",
                                generated_solution=self.schema.Solution(
                                    steps=[self.schema.Step(function_name="inspect", tool_result=result)],
                                    final_answer="January is the coldest month."))

    def score(self, task: object, c2: str = "2") -> list[dict]:
        prompts = []
        text = f"<C1_SCORE>2</C1_SCORE><C1_REASONING>Sound.</C1_REASONING><C2_SCORE>{c2}</C2_SCORE><C2_REASONING>Evidence.</C2_REASONING><C3_SCORE>1</C3_SCORE><C3_REASONING>Consistent.</C3_REASONING>"

        def create(**kwargs: object) -> object:
            prompts.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        # Other tests may have restored a different credentials module.
        with patch.dict(sys.modules, {"agenticeoecosystem.config.credentials": self.credentials}), \
                patch.object(self.credentials, "get_openai_client", return_value=client):
            self.rubric.rubric_score_task_solution(task, self.criteria, provider="eve")
        return prompts

    def test_missing_evidence_preserves_other_scores_without_aggregate(self) -> None:
        task = self.task()
        self.score(task)
        scores = task.evaluator_criteria["rubric_judge"]
        self.assertEqual([s["score"] for s in scores], [2, None, 1])
        self.assertEqual(scores[1]["status"], "unverified")
        self.assertIsNone(task.evaluator_scores_raw["rubric_judge"])
        restored = self.schema.Task.model_validate_json(task.model_dump_json())
        self.assertEqual(restored.evaluator_criteria, task.evaluator_criteria)

    def test_source_evidence_reaches_prompt_and_na_does_not_become_zero(self) -> None:
        task = self.task(result={"months": ["December", "January", "February"], "temperature": [8, 2, 5]})
        prompts = self.score(task)
        self.assertIn('temperature', prompts[0]["messages"][0]["content"])
        self.assertAlmostEqual(task.evaluator_scores_raw["rubric_judge"], 1.8)
        self.score(task, c2="N/A")
        self.assertIsNone(task.evaluator_scores_raw["rubric_judge"])
        self.assertIsNone(task.evaluator_criteria["rubric_judge"][1]["score"])

    def test_stale_inspection_and_failed_gis_do_not_supply_grounding(self) -> None:
        task = self.task()
        package = self.evidence.CandidateEvidence
        package.record_inspection(task, "aaj", [{"kind": "gis_inspection", "result": {
            "tool_calls": [{"result": {"error": "Missing archived file"}}]}}])
        self.assertFalse(package.context(task, "aaj")["available"])
        package.record_inspection(task, "aaj", [{"kind": "artifact_inspection", "result": "Read actual file."}])
        task.generated_solution.final_answer = "February is coldest."
        self.assertFalse(package.context(task, "aaj")["available"])

    def test_archived_files_survive_overwrite_and_tampering_has_no_fallback(self) -> None:
        output = self.root / "output"
        output.mkdir()
        path = output / "source.txt"
        path.write_text("Original observation")
        task = self.task(result={"saved_path": str(path)})
        self.evidence.CandidateEvidence.archive(task, output, self.root / "archive")
        path.write_text("Different candidate")
        archive = Path(task.candidate_artifacts[0]["archived_path"])
        self.assertEqual(archive.read_text(), "Original observation")
        self.assertEqual(self.aaj._candidate_artifacts(task), [archive])
        archive.write_text("Tampered")
        self.assertEqual(self.aaj._candidate_artifacts(task), [])

    def test_monthly_values_from_saved_daily_data_and_missing_month(self) -> None:
        import numpy as np
        import xarray as xr

        path = self.root / "daily.nc"
        dates = np.array(["2022-12-01", "2022-12-02", "2023-01-01", "2023-02-01"], dtype="datetime64[D]")
        xr.Dataset({"t": (("time", "latitude"), [[8, 10], [10, 12], [2, 4], [np.nan, np.nan]], {"units": "K"})},
                   coords={"time": dates, "latitude": [40, 41]}).to_netcdf(path)
        inspector = self.gis.ArtifactInspector([path])
        series = inspector.inspect_timeseries(str(path), "t", "unweighted_mean", "MS")
        self.assertEqual(series["values"], [10.0, 3.0, None])
        self.assertEqual(series["units"], "K")
        with self.assertRaises(ValueError):
            inspector.inspect_timeseries(str(path), "t")

    def test_gis_requirement_uses_full_task_mapping(self) -> None:
        mapping = self.evidence.EvaluationRequirements(ROOT / "data/tasks/task_evaluation_strategies.json")
        self.assertTrue(mapping.requires_gis("TASK_260713_000065_600065"))
        self.assertFalse(mapping.requires_gis("TASK_260713_000014_600014"))

        class Message:
            def __init__(self, name: str, arguments: dict) -> None:
                self.content = None
                self.tool_calls = [SimpleNamespace(id=name, function=SimpleNamespace(
                    name=name, arguments=json.dumps(arguments)))]

            def model_dump(self, **kwargs: object) -> dict:
                return {"role": "assistant", "tool_calls": []}

        sequence = iter([Message("submit_verdict", {"score": 2, "reasoning": "Premature"}),
                         Message("ask_gis_agent", {"question": "Read saved data"}),
                         Message("submit_verdict", {"score": 1, "reasoning": "Evidence unavailable"})])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=next(sequence))]))))
        task = self.task("TASK_260713_000065_600065")
        with patch.object(self.aaj, "get_openai_client", return_value=client), \
                patch.object(self.aaj.GISJudgeAgent, "examine", return_value={"tool_calls": [], "summary": "Unavailable"}):
            self.aaj.judge_task_agentic(task, model="mock", max_turns=3)
        self.assertEqual(task.evaluator_scores["agent_judge"], 1)
        self.assertEqual(task.evaluator_tool_calls["agent_judge"][0]["tool"], "ask_gis_agent")

    def test_runtime_distance_rejects_changed_source_and_candidate(self):
        path = self.root / 'paper.txt'
        quote = 'The study-area centroid is at longitude 0, latitude 0.'
        path.write_text(quote)
        args = dict(point1_lon_lat=[0., 0.], point2_lon_lat=[1., 0.], source_path=str(path),
                    source_quote=quote, centroid_derivation='Paper explicitly supplies the centroid.',
                    source_location_verified=True, destination_verified=True)
        result = self.gis.ArtifactInspector([path]).measure_geodetic_distance(**args)
        task = self.task()
        self.evidence.CandidateEvidence.record_inspection(task, 'aaj', [
            {'kind': 'gis_inspection', 'result': {'tool_calls': [
                {'tool': 'measure_geodetic_distance', 'arguments': json.dumps(args), 'result': result}]}}])
        reference = self.evidence.CandidateEvidence.runtime_distance_reference(task, 'aaj')
        self.assertEqual(reference['distance_m'], result['distance_m'])
        path.write_text(quote + ' modified')
        self.assertIsNone(self.evidence.CandidateEvidence.runtime_distance_reference(task, 'aaj'))
        path.write_text(quote)
        task.generated_solution.final_answer = 'Different candidate answer.'
        self.assertIsNone(self.evidence.CandidateEvidence.runtime_distance_reference(task, 'aaj'))

    def test_gis_can_be_skipped_only_for_zero_without_execution_or_artifacts(self) -> None:
        class Message:
            content = None

            def __init__(self, score: object) -> None:
                self.tool_calls = [SimpleNamespace(id="verdict", function=SimpleNamespace(
                    name="submit_verdict", arguments=json.dumps({
                        "score": score, "reasoning": "No task execution was recorded."})))]

            def model_dump(self, **kwargs: object) -> dict:
                return {"role": "assistant", "tool_calls": []}

        cases = [
            ("refusal", 0, False, False, False, 0),
            ("string_zero", "0", False, False, False, 0),
            ("partial_without_evidence", 1, False, False, False, None),
            ("success_without_evidence", 2, False, False, False, None),
            ("invalid_score", "invalid", False, False, False, None),
            ("execution_without_artifact", 0, True, False, False, None),
            ("missing_archived_artifact", 0, False, True, False, None),
            ("missing_manifest_artifact", 0, False, False, True, None),
        ]
        for name, score, steps, archive, manifest, expected in cases:
            with self.subTest(name=name):
                task = self.task("TASK_260713_000069_600069")
                if not steps:
                    task.generated_solution.steps = []
                task.generated_solution.final_answer = "I cannot download or process this data."
                if archive:
                    task.candidate_artifacts = [{
                        "archived_path": str(self.root / "missing.nc"), "sha256": "missing"}]
                paths = [str(self.root / "missing.nc")] if manifest else None
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=Message(score))]))))
                with patch.object(self.aaj, "get_openai_client", return_value=client), \
                        patch.object(self.aaj.GISJudgeAgent, "examine") as inspect:
                    self.aaj.judge_task_agentic(task, model="mock", max_turns=1,
                                              artifact_paths=paths)
                self.assertEqual(task.evaluator_scores["agent_judge"], expected)
                inspect.assert_not_called()
                self.assertFalse(self.evidence.CandidateEvidence.context(
                    task, "agent_judge")["available"])

    def test_image_previews_follow_all_tool_responses_in_batch(self) -> None:
        class Message:
            content = None

            def __init__(self, calls: list[tuple[str, dict]]) -> None:
                self.tool_calls = [SimpleNamespace(id=f"inspect0{i}", type="function",
                    function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                    for i, (name, args) in enumerate(calls)]

            def model_dump(self, **kwargs: object) -> dict:
                return {"role": "assistant", "tool_calls": [
                    {"id": call.id, "type": call.type, "function": {
                        "name": call.function.name, "arguments": call.function.arguments}}
                    for call in self.tool_calls]}

        for suffixes in ((".jpg",), (".jpg", ".jpg"), (".jpg", ".txt"), (".txt", ".jpg")):
            with self.subTest(suffixes=suffixes):
                paths = [self.root / f"artifact{i}{suffix}" for i, suffix in enumerate(suffixes)]
                for path in paths:
                    path.touch()
                calls = [("inspect_file", {"path": str(path)}) for path in paths]
                sequence = iter([Message(calls), Message([
                    ("submit_verdict", {"score": 2, "reasoning": "Inspected archived artifacts."})])])
                requests = []

                def create(**kwargs: object) -> object:
                    requests.append(copy.deepcopy(kwargs["messages"]))
                    return SimpleNamespace(choices=[SimpleNamespace(message=next(sequence))])

                def inspect(path: str, inspector: object) -> tuple[str, str | None]:
                    uri = f"data:image/jpeg;base64,preview{Path(path).stem}" if Path(path).suffix == ".jpg" else None
                    return "Read archived artifact", uri

                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                task = self.task()
                with patch.object(self.aaj, "get_openai_client", return_value=client), \
                        patch.object(self.aaj, "_tool_inspect_file", side_effect=inspect):
                    self.aaj.judge_task_agentic(task, model="mock", max_turns=2,
                                              artifact_paths=[str(path) for path in paths])
                self.assertEqual(task.evaluator_scores["agent_judge"], 2)
                messages = requests[1][-len(paths) - 2:]
                self.assertEqual([message["role"] for message in messages],
                                 ["assistant"] + ["tool"] * len(paths) + ["user"])
                self.assertEqual([message["tool_call_id"] for message in messages[1:-1]],
                                 [call["id"] for call in messages[0]["tool_calls"]])
                previews = messages[-1]["content"]
                self.assertEqual(len(previews), 2 * suffixes.count(".jpg"))
                for index, path in enumerate(path for path in paths if path.suffix == ".jpg"):
                    self.assertEqual(previews[2 * index], {"type": "text", "text": f"Preview of {path}:"})
                    self.assertEqual(previews[2 * index + 1]["image_url"]["url"],
                                     f"data:image/jpeg;base64,preview{path.stem}")

    def test_empty_allowlist_forces_one_gis_consultation_then_verdict(self) -> None:
        requests = []
        names = iter(["list_output_files", "inspect_file", "ask_gis_agent", "submit_verdict"])

        def create(**kwargs: object) -> object:
            requests.append(copy.deepcopy(kwargs))
            name = next(names)
            arguments = {"path": "invented.tif"} if name == "inspect_file" else (
                {"score": 0, "reasoning": "Download never executed; imagery unavailable."}
                if name == "submit_verdict" else {"question": "Assess missing evidence"})
            call = {"id": str(len(requests)), "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                id=call["id"], type="function", function=SimpleNamespace(**call["function"]))],
                model_dump=lambda **unused: {"role": "assistant", "tool_calls": [call]})
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        task = self.task("TASK_260619_000006_300006")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(self.aaj, "get_openai_client", return_value=client), \
                patch.object(self.aaj, "_candidate_artifacts", return_value=[]), \
                patch.object(self.aaj.GISJudgeAgent, "examine", return_value={
                    "summary": "No inspectable artifacts", "tool_calls": []}) as gis:
            self.aaj.judge_task_agentic(task, model="mock", max_turns=4,
                                      transcript_dir=str(self.root),
                                      evaluation_requirements=SimpleNamespace(requires_gis=lambda task_id: True, tasks={}))
        self.assertEqual(requests[2]["tool_choice"]["function"]["name"], "ask_gis_agent")
        self.assertEqual(requests[3]["tool_choice"]["function"]["name"], "submit_verdict")
        gis.assert_called_once()
        self.assertEqual(task.evaluator_scores["agent_judge"], 0)
        self.assertIn("Do not guess paths", requests[2]["messages"][-1]["content"])

    def test_malformed_tool_responses_retry_without_polluting_history_or_execution(self) -> None:
        class Message:
            content = "Discarded assistant text"

            def __init__(self, calls: list[tuple[str, object, object]]) -> None:
                self.tool_calls = [SimpleNamespace(id=identifier, type="function",
                    function=SimpleNamespace(name=name, arguments=arguments))
                    for name, arguments, identifier in calls]

            def model_dump(self, **kwargs: object) -> dict:
                return {"role": "assistant", "content": self.content, "tool_calls": [
                    {"id": call.id, "type": "function", "function": {
                        "name": call.function.name, "arguments": call.function.arguments}}
                    for call in self.tool_calls]}

        cases = [
            [("submit_verdict", '{}{}', "bad")],
            [("submit_verdict", '[]', "bad")],
            [("submit_verdict", 'null', "bad")],
            [("submit_verdict", '', "bad")],
            [("submit_verdict", {}, "bad")],
            [("", '{}', "bad")],
            [("unknown_function", '{}', "bad")],
            [("submit_verdict", '{}', "")],
            [("ask_gis_agent", '{"question":"Check file"}', "valid"),
             ("submit_verdict", '{}{}', "bad")],
            [("ask_gis_agent", '{}', "duplicate"),
             ("submit_verdict", '{}', "duplicate")],
        ]
        for calls in cases:
            with self.subTest(calls=calls):
                rejected = Message(calls)
                valid = Message([("submit_verdict", '{"score":0,"reasoning":"No execution"}', "valid")])
                sequence = iter([rejected, valid])
                requests = []

                def create(**kwargs: object) -> object:
                    requests.append(copy.deepcopy(kwargs["messages"]))
                    return SimpleNamespace(choices=[SimpleNamespace(message=next(sequence))])

                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                task = self.task("TASK_260713_000069_600069")
                task.generated_solution.steps = []
                with patch.object(self.aaj, "get_openai_client", return_value=client), \
                        patch.object(self.aaj.GISJudgeAgent, "examine") as inspect:
                    self.aaj.judge_task_agentic(task, model="mock", max_turns=2,
                                              transcript_dir=str(self.root))
                self.assertEqual(task.evaluator_scores["agent_judge"], 0)
                self.assertEqual(len(requests), 2)
                self.assertFalse(any(m["role"] in {"assistant", "tool"} for m in requests[1]))
                self.assertIn("discarded", requests[1][-1]["content"])
                self.assertEqual([call["tool"] for call in task.evaluator_tool_calls["agent_judge"]],
                                 ["submit_verdict"])
                inspect.assert_not_called()
                transcript = (self.root / f"{task.task_ID}_agent_judge.md").read_text()
                self.assertIn("Rejected malformed response", transcript)


if __name__ == "__main__":
    unittest.main()
