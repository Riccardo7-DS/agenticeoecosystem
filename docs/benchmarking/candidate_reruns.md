# Run a candidate benchmark from another checkout

Use the `rerun` branch. The checked-in task files contain task definitions and
reference workflows, with candidate answers, judge scores and artifacts cleared.
They reproduce the missing-answer plus rerun-required selections used for this
rerun batch. Historical results are not needed and are not included.

## Install and configure

Use Linux and Python 3.12. The current project installs the research stack,
including vLLM and its compute dependencies, even when the candidate is reached
through an API. Candidate API calls do not launch a local LLM server.

```bash
git clone --branch rerun https://github.com/Riccardo7-DS/agenticeoecosystem.git
cd agenticeoecosystem
uv sync --frozen
source .venv/bin/activate
cp .env.example .env
```

Activation is required: MCP subprocesses invoke `python`, and GIS/arXiv tools
invoke `uv`. Fill `.env` with credentials supplied separately; do not commit it.

| Capability | Required configuration or data |
| --- | --- |
| Mistral through EVE | `EVE_ENDPOINT` (including `/v1`), `EVE_USERNAME`, `EVE_PASSWORD`; model ID `Mistral-Small-3.2-24B-Instruct-2506` must be served |
| OpenRouter candidates | `OPENROUTER_API_KEY`; use the full vendor/model ID |
| Satellite imagery and geocoding | `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` |
| ERA5 and other CDS downloads | `CDS_URL`, `CDS_KEY`, and accepted dataset licences |
| EUMETSAT retrieval | `EUMETSAT_CONSUMER_KEY`, `EUMETSAT_CONSUMER_SECRET` |
| ESA strategy retrieval | Reachable `GRAPHRAG_MCP_URL` and full `GRAPHRAG_AUTH_HEADER` |
| EVE knowledge retrieval | `EVE_RETRIEVAL_TOKEN_URL`, `EVE_RETRIEVAL_CLIENT_ID`, `EVE_RETRIEVAL_CLIENT_SECRET` |
| Google Earth Engine | Authorised account/project; provision `src/agenticeoecosystem/servers/gee/gee-key.json` separately and set the GEE variables; `GOOGLE_MAPS_API_KEY` for GEE's geocoder |
| Multi-agent image interpretation | Reachable `VLM_ENDPOINT` (default `http://localhost:18001/v1`) serving `VLM_MODEL` (default `Qwen3-VL`); arrange the team's tunnel or use another approved vision endpoint |
| MOOC retrieval | The bundled transcript files/Qdrant stores, plus embedding/reranking model downloads from Hugging Face |
| GIS and arXiv | `uv` on PATH and network access to their packages/data services |

Provide disk space for downloads and archived evidence. Optional servers may be
skipped when credentials are missing; that makes their benchmark tasks incomplete.
Review the startup tool-loading messages and multi-agent healthcheck before
interpreting results. The single-agent pipeline does not currently expose image
loading through the multi-agent vision sub-agent.

## Run one model

Mistral Small 3.2 multi-agent: 80 selected questions.

```bash
python scripts/generation/run_generate.py \
  --model Mistral-Small-3.2-24B-Instruct-2506 --provider eve \
  --pipeline multi --label Mistral-Small-multi-rerun \
  --tasks-folder data/tasks/candidate_rerun_inputs_20261003 \
  --tasks-file tasks_Mistral-Small-2506.json \
  --output-dir data/candidate_reruns/colleague_mistral_multi \
  --archive-evidence --task-timeout 1800
```

For Mistral single-agent, use `--pipeline single`,
`tasks_Mistral-Small-3.2-24B-single.json` (69 questions), and a distinct label and
output directory. For GLM Flash, use `--provider openrouter`,
`--model z-ai/glm-5.3-flash`, `tasks_GLM-5.3-Flash-OR.json` (86 questions), and
another label/output directory. Nemotron Super uses
`nvidia/nemotron-3-super-120b-a12b` and `tasks_Nemotron-3-Super-OR.json` (88 questions).

Other model-specific task selections are listed in the portable `manifest.json`.
Revised questions use their revised task versions; other missing questions retain
the historical task version. Evaluate the snapshots actually used in the run,
rather than assuming they match a later benchmark revision.

## Resume and queue

To resume `run_generate.py`, point `--tasks-folder` and `--tasks-file` at the
saved results JSON in your output directory and add `--skip-solved`. Keep the
same model, pipeline, label and output directory. Passing the original clean
task input with `--skip-solved` does not load the checkpoint automatically.

The queue helper handles that checkpoint loading for you:

```bash
python scripts/generation/prepare_missing_candidate_reruns.py \
  --prepared-inputs data/tasks/candidate_rerun_inputs_20261003 \
  --output-dir data/candidate_reruns/colleague_queue \
  --execute-mistral --model Mistral-Small-3.2-24B-Instruct-2506 \
  --execute-openrouter
```

Order: Mistral multi-agent, Mistral single-agent, GLM Flash, Nemotron Super.
Maverick is excluded. Add `--mistral-pipelines single` to run only the single
Mistral phase before OpenRouter. Omit `--execute-mistral` to run just OpenRouter.
Use a terminal multiplexer or your site's job manager for long-running jobs.

Run candidates sequentially within one checkout: MOOC stores have exclusive
locks and tool downloads share output paths. A separate result directory alone
does not isolate all tool resources.

Results are saved after each task. `--archive-evidence` preserves generated
artifact files under the run's `evidence/` directory; it does not archive logs.
Logs are written automatically under `logs/`. Generation exceptions may leave
null solutions even when the CLI exits successfully; inspect `completion.json`
for queue runs and logs for individual runs. Saved responses still need judging.

Generated runs, result JSON files, logs and downloaded artifacts are ignored by
Git. Task inputs and source code remain versioned. Never use `git add -f` on run
directories or credentials when preparing a benchmark handoff.
