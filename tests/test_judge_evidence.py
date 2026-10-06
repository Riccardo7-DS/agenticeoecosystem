"""Offline regressions for source grounding, candidate isolation and GIS obligations."""

import importlib
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
        package.__path__ = [str(ROOT / "src/agenticeoecosystem/evaluation")]
        credentials = ModuleType("agenticeoecosystem.config.credentials")
        credentials.get_openai_client = lambda **kwargs: None
        cls.previous = {key: sys.modules.get(key) for key in (
            "agenticeoecosystem.evaluation", "agenticeoecosystem.config.credentials")}
        sys.modules["agenticeoecosystem.evaluation"] = package
        sys.modules["agenticeoecosystem.config.credentials"] = credentials
        cls.credentials = credentials
        cls.schema = importlib.import_module("agenticeoecosystem.evaluation.dataclasses")
        cls.evidence = importlib.import_module("agenticeoecosystem.evaluation.candidate_evidence")
        cls.rubric = importlib.import_module("agenticeoecosystem.evaluation.rubric_evaluator")
        cls.aaj = importlib.import_module("agenticeoecosystem.evaluation.agent_judge")
        cls.gis = importlib.import_module("agenticeoecosystem.evaluation.gis_judge_agent")

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


if __name__ == "__main__":
    unittest.main()
