# Current task configuration

Use the files directly in this directory for the current benchmark, ground
truths, evaluation strategies, tool ownership mapping and rubric profiles.
`rubrics_108q_v4_evidence.json` is the current evidence-based rubric;
`rubrics_108q_v4_trace.json` is the alternate trace-only profile, not an older
revision of the evidence profile. The rubric manifests and change log document
their origins. `example_tasks.json` remains available for small examples.

`previous/` contains one frozen previous configuration: the October 12 prepared
snapshot used by the fixed-judge evaluations. It is retained for reproducibility,
not automatically selected by the current runners. Select all configuration
paths explicitly when reproducing that snapshot.

`tasks_rewritten4.json` remains because the four revised questions still require
fresh candidate answers; moving old rerun lists does not establish completion.

Retired rerun inputs, historical rubrics and audit material were moved to
[`../archive/tasks_20261013/`](../archive/tasks_20261013/). The ignored `4096`
compatibility symlink preserves paths mentioned in historical candidate traces;
its approximately 810 MB of contents now live in the archive.

Candidate answers, judge outputs and task-specific candidate reference files
remain outside this directory. This cleanup did not change current configuration
contents or existing evaluation snapshots.
