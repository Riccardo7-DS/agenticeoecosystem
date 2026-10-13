# Trajectory metrics

The full evaluation runner (`TrajectoryJudge` in
`src/agenticeoecosystem/evaluation/judges/trajectory_judge.py`) computes six
deterministic workflow metrics for each in-scope task against the reference
solutions in the benchmark file.

| Metric | Definition |
| --- | --- |
| `tools_in_order` | Longest reference prefix matched as a subsequence of predicted tools, divided by reference length. Reference items cannot be skipped. |
| `tool_exact_match` | Longest common tool prefix divided by reference length. Extra trailing calls do not reduce this score. |
| `parameter_accuracy` | Longest prefix matching both tool names and structured arguments, divided by reference length. Object key order is ignored; values and array order are preserved. |
| `agents_in_order` | The same ordered-prefix subsequence metric on agent sequences. |
| `agent_exact_match` | Longest common agent prefix divided by reference agent length. |
| `tool_efficiency` | Predicted tool-call count divided by the shortest nonempty reference's tool-call count. |

When a task has several accepted references, each alignment metric uses the
best-scoring nonempty one. The metrics measure adherence to curated workflows,
not scientific correctness, and do not enter the meta score.

Agent sequences are derived from `data/tasks/tool_agent_mapping.json`: each tool
call is assigned to its owning agent and consecutive calls by the same agent form
one segment. They therefore measure capability routing rather than observed
handoffs, which makes single- and multi-agent candidates comparable. Tools with
unknown or ambiguous owners leave the agent metrics unverified. Empty reference
trajectories are excluded rather than scored.
