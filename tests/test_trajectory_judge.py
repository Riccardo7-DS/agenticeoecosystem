"""Offline contracts for prefix matching, extra calls and missing agent traces."""
import importlib
import unittest

from test_full_evaluation import PACKAGE, schema

judge = importlib.import_module(f"{PACKAGE.__name__}.judges.trajectory_judge")


class TrajectoryTests(unittest.TestCase):
    """Check the paper's definitions rather than LCS or binary exact match."""

    def solution(self, names: list[str], agents: list[str] | None = None) -> object:
        return schema.Solution(steps=[schema.Step(function_name=name) for name in names],
                               agents_called=agents)

    def test_reference_prefix_cannot_skip_missing_tool(self) -> None:
        result = judge.TrajectoryJudge().evaluate(self.solution(['a', 'c']),
                                                  [self.solution(['a', 'b', 'c'])])
        self.assertEqual(result['tools_in_order']['value'], 1 / 3)
        self.assertEqual(result['tool_exact_match']['value'], 1 / 3)

    def test_interleaved_extra_call_and_trailing_calls(self) -> None:
        result = judge.TrajectoryJudge().evaluate(self.solution(['a', 'x', 'b', 'c']),
                                                  [self.solution(['a', 'b'])])
        self.assertEqual(result['tools_in_order']['value'], 1)
        self.assertEqual(result['tool_exact_match']['value'], .5)
        self.assertEqual(result['tool_efficiency']['value'], 2)
        trailing = judge.TrajectoryJudge().evaluate(self.solution(['a', 'b', 'x']),
                                                    [self.solution(['a', 'b'])])
        self.assertEqual(trailing['tool_exact_match']['value'], 1)

    def test_arguments_ignore_key_order_but_preserve_values(self) -> None:
        ref = self.solution(['a', 'b'])
        ref.steps[0].arguments = {'x': 1, 'y': [2, 3]}
        candidate = self.solution(['a', 'b'])
        candidate.steps[0].arguments = {'y': [2, 3], 'x': 1}
        candidate.steps[1].arguments = {'extra': True}
        result = judge.TrajectoryJudge().evaluate(candidate, [ref])
        self.assertEqual(result['parameter_accuracy']['value'], .5)

    def test_agent_sequences_and_missing_recordings(self) -> None:
        ref = self.solution(['a'], ['supervisor', 'climate', 'supervisor'])
        candidate = self.solution(['a'], ['supervisor', 'gis', 'climate', 'supervisor'])
        result = judge.TrajectoryJudge().evaluate(candidate, [ref])
        self.assertEqual(result['agents_in_order']['value'], 1)
        self.assertEqual(result['agent_exact_match']['value'], 1 / 3)
        missing = judge.TrajectoryJudge().evaluate(self.solution(['a']), [ref])
        self.assertEqual(missing['agents_in_order']['status'], 'unverified')
        empty = judge.TrajectoryJudge().evaluate(self.solution(['a'], []), [ref])
        self.assertEqual(empty['agents_in_order']['value'], 0)

    def test_empty_and_multiple_references(self) -> None:
        result = judge.TrajectoryJudge().evaluate(self.solution(['b']),
                    [self.solution([]), self.solution(['a', 'b']), self.solution(['b'])])
        self.assertEqual(result['tool_exact_match']['value'], 1)
        self.assertEqual(result['tool_exact_match']['reference_index'], 2)
        self.assertEqual(result['tool_efficiency']['value'], 1)
        missing = judge.TrajectoryJudge().evaluate(self.solution([]), [self.solution([])])
        self.assertTrue(all(value['status'] == 'unverified' for value in missing.values()))

    def test_tool_owned_segments_and_unknown_tools(self) -> None:
        mapping = {'tools': {'a': {'agents': ['climate']}, 'b': {'agents': ['climate']},
                             'c': {'agents': ['gis']}, 'shared': {'agents': ['gis', 'climate']}}}
        evaluator = judge.TrajectoryJudge(mapping)
        result = evaluator.evaluate(self.solution(['a', 'b', 'c', 'a']),
                                    [self.solution(['a', 'c', 'b'])])
        self.assertEqual(result['agent_exact_match']['value'], 1)
        self.assertEqual(result['agents_in_order']['predicted_sequence'], ['climate','gis','climate'])
        self.assertEqual(result['agent_exact_match']['basis'], 'tool_ownership')
        for name in ['unknown', 'shared']:
            result = evaluator.evaluate(self.solution(['a', name, 'c']), [self.solution(['a', 'c'])])
            self.assertEqual(result['agents_in_order']['status'], 'unverified')
            self.assertEqual(result['agents_in_order']['unresolved_tools'][0]['step_index'], 1)


if __name__ == '__main__':
    unittest.main()
