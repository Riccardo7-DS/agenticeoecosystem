# Virtual Lab — iteration 2 plan

Status: **steps 1-4 implemented on `feat/virtual-lab`, 45 offline tests pass; step 5 live
run in progress** (updated 2026-11-01). Nothing is committed yet.
Package: `src/agenticeoecosystem/agents/virtual_lab/`.

### Deviations from the plan below

- **Step 1 fell back to option A.** The spike showed option B can't work on ADK 2.2 with our
  coordinator: `_SingleTurnAgentTool` runs the sub-agent through the node runtime
  (`tool_context.run_node`), which needs `InvocationContext._event_queue`. The Runner only
  sets that up when the root is an `LlmAgent` or a non-agent Workflow node; a custom
  `BaseAgent` root (our coordinator) takes the legacy path (`runners.py:1038`, "remove …
  after all agents are refactored to use the node runtime path"). Error:
  `_enqueue_event called but _event_queue is not set`. Option B becomes viable if the
  coordinator is rewritten as an ADK Workflow. Implemented instead: `AgentTool` around a
  chat-mode clone of each sub-agent (originals untouched). Ledger change for string errors
  (P8) was not needed and was reverted. Test fakes are now `single_turn` like the real
  sub-agents, and the suite fails with the live error if the clone is removed.
- **Duplicate-claim threshold is 0.4, not 0.6**: on the 2026-11-01 run the MODIS rewordings
  scored 0.43-0.65 and all distinct claims <= 0.37.

## Where we are

Iteration 1 is implemented (coordinator phases A–D, schemas, validator, ledger,
plan-only guard, report renderer) plus, since the last run:

- tightened plan checks (`validation.py`: reference independence, hypothesis ↔ step
  links, step ↔ sub-agent access, sample-size warning) and a critic plan review with
  redesign on major issues (`coordinator.py`, `prompts.py`);
- literature check on contested claims: arXiv search via the arxiv MCP server,
  `rank_papers_by_citations` (Semantic Scholar, OpenAlex fallback), `literature_reviewer`
  verdicts, results in state and the record (`literature.py`, `coordinator.py`);
- per-call LLM timing in the ledger; `scripts/render_virtual_lab_report.py`.

39 offline tests pass:
`env -u EVE_ENDPOINT uv run --with pytest pytest -q tests/test_virtual_lab_*.py`

Latest live run (Qwen3-14B on local vLLM :8011, 17m25s):
`logs/virtual_lab_emsr859_qwen3-14b_20261101_093306.{json,md}`.

## Problems found in that run (what this plan fixes)

| # | Problem | Evidence |
|---|---|---|
| P1 | Every specialist call to an existing sub-agent failed: `LlmAgent as root agent must have mode='chat', but got mode='single_turn'`. `AgentTool` runs the wrapped agent as root of a new Runner; our sub-agents are `single_turn` (needed by `multi`). Same in ADK 2.13, so upgrading doesn't help. | tool calls tc-0001..tc-0018 |
| P2 | 9/17 arXiv searches returned 0 papers: queries AND-chain quoted phrases (the reviewer prompt's own example does this). | tc-0022, tc-0025..0032 |
| P3 | Same MODIS claim checked 3× (reworded each round), identical queries re-sent. | r1-c3, r2-c1, r2-c3 |
| P4 | A specialist called `rank_papers_by_citations` with the docstring's example ids. | tc-0006 |
| P5 | No specialist cited a literature paper, so linked evidence is still 0. | final proposals |
| P6 | Critic plan review is noisy: echoes checklist text as the "claim", calls the EMS delineation non-independent, invents a feasibility issue ("thermal_specialist" as executor). | final.plan_review |
| P7 | The plan still schedules MTG FCI / MSG SEVIRI (step 2) although the validator rates them unusable (VZA ~85°) — `validate_plan` only checks `plan.datasets`, not sensors named in workflow steps. | plan step 2 |
| P8 | `_SingleTurnAgentTool` (used by fix B) returns errors as strings (`"Error running sub-agent: ..."`) instead of raising, so the ledger would mark failed calls `ok`. | ADK `tools/agent_tool.py` |

## Step 1 — Option B: sub-agents as single-turn `sub_agents` (fixes P1, P8)

ADK 2.2 turns each `single_turn` `LlmAgent` in `sub_agents` into an inline
`_SingleTurnAgentTool` (`agents/llm_agent.py:1005-1021`), run via
`tool_context.run_node(..., use_sub_branch=True)` in the caller's invocation —
so runner plugins (our `LabGuardPlugin`) see it without `include_plugins`.

1. **Spike first** (scratch test, fake LLM): a `mode="chat"` LlmAgent with a cloned
   `single_turn` sub-agent, run by our coordinator via `agent.run_async(sub_ctx)`
   (not by a Workflow). Confirm: the tool is exposed, `run_node` works in this
   (non-workflow) context, the plugin's before/after tool callbacks fire with
   `current_step` set, branch isolation holds. If `run_node` needs a workflow
   context, stop and fall back to option A (chat-mode clone in `AgentTool`).
2. `participants.py`: replace `AgentTool(agent=existing_agents[n])` with per-participant
   clones: `existing_agents[n].clone()` (keeps `mode="single_turn"`), passed as
   `sub_agents=` to the specialist/critic/PI. One clone per (participant, sub-agent)
   because an agent can have only one parent; originals stay untouched so `multi`
   keeps working. Keep the original names (`cdse_agent`, ...): prompts, ledger `via`
   and evidence linking use them. ADK only warns on duplicate names among siblings.
3. `_lab_agent`: set `mode="chat"` explicitly on lab participants (required for
   sub-agent tools; currently `None`).
4. `ledger.py` `after_tool_callback`: treat string results starting with
   `"Error running sub-agent"` / `"Error validating input"` as `status="error"` (P8).
   `isinstance(tool, AgentTool)` still holds for `_SingleTurnAgentTool`, so the
   plan_only guard is unchanged — verify.
5. Check clones share MCP toolsets/sessions sensibly (no N× MCP server spawns).
6. Tests: update `tests/_virtual_lab_fakes.py` fake sub-agents to `mode="single_turn"`;
   existing tests `test_evidence_is_linked_to_tool_calls` and
   `test_plan_only_blocks_downloading_tools` must pass unchanged; add a test that a
   failing sub-agent is logged as `error`; keep `test_existing_agents_are_not_reparented`.

## Step 2 — Literature search quality (fixes P2, P3, P4)

1. `prompts.literature_reviewer` (lit_query): 2–4 key terms per query, `OR` between
   synonyms, at most one `AND`, no quoted multi-word chains; replace the bad example.
2. `coordinator._literature_check`: if a query returns 0 papers, retry once with a
   broadened query derived in code (drop quotes, turn `AND` into `OR`) — logged as
   its own ledger call.
3. Per-run query cache (`query -> paper list`) on the coordinator: identical queries
   are not re-sent (record the reuse in the check's `tool_call_ids`).
4. Near-duplicate claims: before checking, skip a claim whose word-set Jaccard
   similarity with an already-checked claim ≥ 0.6 (pure function in
   `literature.py`, unit-tested on the three MODIS claims from the run).
5. `rank_papers_by_citations` docstring: remove example ids; say "ids returned by
   search_papers".

## Step 3 — Make literature citable and cited (fixes P5)

1. Specialist revise prompt: when LITERATURE CHECKS concern your proposal, cite the
   relevant paper(s) in `evidence` (source `arxiv:<id>`) or explain in `limitations`
   why you disagree.
2. Validator: for a proposal targeted by a literature check with verdict
   supports/contradicts, a revision that neither cites one of its papers nor
   mentions it in limitations gets a `minor` evidence_sufficiency issue.

## Step 4 — Plan review precision (fixes P6, P7)

1. `validate_plan`: also run `check_dataset` coverage/availability on every sensor
   named in workflow step descriptions (catches FCI/SEVIRI in step 2) (P7).
2. `prompts.critic_plan_review`: "claim" must quote the plan text being criticised,
   not the checklist; the EMS delineation, VHR imagery and agency reports count as
   independent references; executors are the sub-agents in the EXECUTING SUB-AGENTS
   list (give the critic that list — it currently doesn't have it).
3. Coordinator: drop critic plan issues whose `claim` is a verbatim checklist line
   (cheap deterministic filter) before deciding on a redesign.

## Step 5 — Verify

1. Offline suite green (`tests/test_virtual_lab_*.py`).
2. Re-run on local Qwen3-14B (`scratchpad/run_lab_local.py` — copy into
   `scripts/` if the scratchpad is gone; it sets MULTI_AGENT_PROVIDER=vllm,
   VLLM_PORT=8011, MULTI_AGENT_MODEL=qwen3-14b and renders the report).
3. Success criteria for the run:
   - ≥ 1 successful (status `ok`) specialist call to an existing sub-agent;
   - ≥ 1 retrieved claim linked to a tool call in the final proposals;
   - no plan step uses a sensor the validator rates blocking;
   - fewer empty arXiv searches than 9/17, no duplicate queries;
   - report renders; compare time and tokens with the 17m25s run.

## Files touched

`participants.py`, `ledger.py`, `coordinator.py`, `prompts.py`, `literature.py`,
`validation.py`, `tests/_virtual_lab_fakes.py`, `tests/test_virtual_lab_meeting.py`,
`tests/test_virtual_lab_validation.py`. No changes outside the package and tests;
no dependency changes (no `uv lock`).

## Housekeeping (before any commit — only when asked)

- HEAD is on `feat/climate-risk`; the Virtual Lab belongs on `feat/virtual-lab`
  (exists). Switch with the untracked files carried over; leave the climate-risk
  changes out of Virtual Lab commits.
- `scripts/*` is gitignored: `run_virtual_lab_example.py` and
  `render_virtual_lab_report.py` need `git add -f` (or moving) to be committed.
- Check the `uv.lock` diff before committing.
- vLLM (Qwen3-14B) is still running on :8011 (pid in the session scratchpad
  `vllm.pid`); stop it when done.

## Step 5 result — live run 2026-11-01 10:33 (`logs/virtual_lab_emsr859_qwen3-14b_20261101_103312.{json,md}`)

17m10s, 27 lab LLM calls (+12 inside sub-agents), 31 tool calls (3 blocked), 0 repairs.
Decision `insufficient_evidence` (PI said needs_experiment; capped: plan has blocking issues).

| Criterion | Result |
|---|---|
| ≥1 ok specialist → sub-agent call | **Met**: 5 ok of 6 (gee, cdse, eo_stac, arxiv_rag, mooc_rag; graph_rag overflowed Qwen's 32k context). Only optical_specialist called tools. Plan-only blocked 3 cdse `fetch_*`. |
| ≥1 linked retrieved claim in final proposals | **Not met**: optical r0 had one (mooc_rag, tc-0006/7) but revisions dropped it. |
| No plan step uses a blocking sensor | **Not met, but caught**: step 10 uses FCI/SEVIRI → new step-sensor check flags blocking → status capped. Designer also assigned steps to `thermal_specialist` etc. |
| Fewer empty arXiv searches, no duplicates | **Met**: 1/11 empty (was 9/17), 11/11 unique; MODIS rewordings no longer re-checked. |

New problems for iteration 3:
- N1 Broad queries now return off-topic papers (cond-mat/0411570, 0706.3074) and citation
  ranking promotes them (old + highly cited); the reviewer still returned "contradicts".
  Fix: arXiv categories filter (physics.ao-ph, physics.geo-ph, eess.IV, eess.SP, cs.CV) and/or a
  claim-relevance filter before ranking; reviewer must say "inconclusive" for off-topic papers.
- N2 Revisions drop earlier linked evidence. Fix: carry linked retrieved evidence forward
  unless the revision explicitly removes the claim.
- N3 Designer assigns steps to lab participants. Fix: prompt says lab participants are not
  executors; redesign hint lists valid executors.
- N4 Only one specialist grounded. Consider a required first tool call per specialist.

## Iteration 3 — N1-N3 implemented (2026-11-01), 48 offline tests pass

- N1 `literature.py`/`config.py`: `search_papers` now gets `categories`
  (`literature_categories`, default physics.ao-ph, physics.geo-ph, eess.IV, eess.SP, cs.CV;
  filtered server-side by the arxiv MCP server). Before citation ranking, papers whose
  title+abstract mention fewer than `literature_min_terms` (2) query terms are dropped
  (`query_terms`, `is_relevant`). Reviewer assess prompt: off-topic papers are ignored and
  never cited; "contradicts" needs an on-topic abstract.
- N2 `coordinator._carry_linked_evidence`: a revision keeps the previous version's
  tool-linked retrieved evidence unless it restates the claim (duplicate-claim similarity).
- N3 designer prompt names the lab members and says they execute nothing; the redesign hint
  lists the valid executors.
- N4 not done. Thermal and SAR specialists did have tools in run 3 (eumetsat/gee/tle and
  cdse/tle, plus the literature agents); they simply didn't call them.

### Run 4 (`logs/virtual_lab_emsr859_qwen3-14b_20261101_165644.{json,md}`, 23m)

- Thermal specialist now grounded too (eumetsat, gee, tle, arxiv_rag, mooc_rag calls ok).
- 40-call tool budget spent in round 1 (inner sub-agent calls count). Bug: `_budget_allows`
  treated any spent budget as fatal, so revisions, literature assessment, plan review and
  redesign were all skipped (termination `budget`, 1 round). Fixed: only a spent LLM budget
  stops LLM steps; test `test_spent_tool_budget_still_lets_the_meeting_finish`.
- N1 partly: the dNBR/smoke claim now gets on-topic papers (BFAST burned area, Sentinel-2
  fire risk); the thermal claim still got off-topic ones (methane leaks, digital twins) that
  match generic terms ("sensors", "detection"). The reviewer's queries still chain 3 ANDs;
  the SAR claim found nothing within the categories.
- N2 not observable: every evidence item was written as "assumption" with no source, even
  by specialists that called tools, so there was nothing linked to carry forward.
- N3 not observable: the designer still put lab members on steps, and the redesign that
  would correct it was skipped by the budget bug.

### Run 5 (`logs/virtual_lab_emsr859_qwen3-14b_20261101_172416.{json,md}`, 25m), after the budget fix

- Budget fix works: tool budget still spent (40), but 2 rounds ran and literature assess,
  plan review and redesign all ran. Optical and thermal both grounded (5 ok calls each).
- Reviewer now rejects off-topic papers: all 6 checks "inconclusive", summaries name the
  off-topic papers (methane, ambient backscatter, fire-spread reviews). No false
  "contradicts". But 3/6 checks found nothing: 3-AND queries inside 5 categories are too
  narrow; the reviewer ignores "at most one AND".
- Still open:
  - E1 Specialists write every evidence item as "assumption" with no source even after
    successful tool calls, so linking and carry-forward never trigger.
  - E2 Designer and redesign both assign 6/10 steps to lab members, despite the prompt and the
    explicit executor list. Likely because analysis steps (raster maths, comparison) have no
    natural executor among the loaded sub-agents (no gis_agent in this run).
  - E3 Queries: enforce the AND limit in code (keep the first AND clause, OR the rest) instead
    of the prompt.

## Iteration 4 (2026-11-02): AND limit in code, tool-call evidence, gis_agent — 53 offline tests pass

- `literature.limit_and`: queries keep one top-level AND ('A AND B AND C' -> 'A AND (B OR C)').
- `validation.tool_call_evidence`: each successful sub-agent call a specialist made becomes a
  RETRIEVED item ("<tool> returned: <result>", linked to the call and its leaf calls) unless
  already cited; the norms ask for at least one claim per tool used.
- gis_agent loaded in the local run script (`run_virtual_lab_example.py` loads all); designer
  routes analysis of downloaded data to gis_agent when it is an executor.
- Run 6 stalled on the shared filesystem (MCP 300 s timeouts, processes in D state); killed.

### Run 7 (`logs/virtual_lab_emsr859_qwen3-14b_20261102_142554.{json,md}`, 17m18s)

- Plan: 5 steps, all on real executors (cdse, eumetsat ×2, gis ×2). Lab members gone (E2 fixed).
  Still schedules FCI/SEVIRI on eumetsat steps -> blocking -> status capped insufficient_evidence.
- Queries now have one AND each (E3 fixed); 0/4 searches empty. Papers still mostly off-topic
  (generic "sensors"/"detection" matches); reviewer rejected them all (inconclusive).
- No specialist called any tool this run (one LLM call each at propose), so tool-call
  evidence could not trigger. Grounding with Qwen3-14B varies run to run: 1, 1, 2, 0
  specialists in runs 3/4/5/7. -> N4 (require a tool call in the first proposal) is now the
  main blocker for linked evidence.

## Iteration 5 (2026-11-02): role cards, linking fixes, generic checks — 57 offline tests pass

- Linking: revisions link restated claims to the specialist's earlier calls (when the tool
  wasn't called again); lab-made literature calls (`lab_call`) link only by arXiv id.
- Role cards `virtual_lab/roles/<specialist>.md` (methods, data facts, pitfalls, validation,
  references; 15 arXiv ids checked against the arXiv API) injected into specialist prompts.
  New `foundation_model_specialist`; SAR role covers InSAR deformation; all 5 roles offered,
  PI picks 2-4.
- Wildfire wording removed from generic checks/prompts; dataset role synonyms
  (validation/benchmark -> reference, training -> primary) normalised in `DatasetChoice`.

### Deformation question ("Which Foundation model is the best for ground deformation detection?")

- Before cards (`..._153629`): optical + SAR picked, no foundation model named, 37 min.
- After cards, run 1 (`..._155902`): FM + SAR picked; both proposals rejected on dataset
  role "validation"/"benchmark" -> no_valid_proposals. Fixed by role synonyms.
- After cards, run 2 (`..._161545`, 12m13s, 0 invalid outputs): FM + SAR; CROMA, DOFA,
  TerraMind compared on PANGAEA/GEO-Bench, EGMS/LiCSAR data, GNSS reference, U-Net baseline.
  Real disagreement surfaced (models pretrained on GRD amplitude vs InSAR phase). Decision
  needs_experiment. Open: no specialist tool calls (all evidence assumptions); off-topic
  literature hits; plan asks cdse_agent for LiCSAR/GNSS/EGMS and gis_agent to train models,
  which neither can do, and the validator only says "not in catalog" (minor).

## EVE-Instruct runs on EMSR859 (2026-11-03)

- `logs/virtual_lab_emsr859_result_20261103_090733` (defaults 40/40): EVE uses tools eagerly
  (37 grounding calls in Phase B, 21 linked claims), but the tool budget ran out in Phase B,
  specialists answered blocked calls in prose (13 invalid outputs), and the LLM budget ran out
  after round 1 → no decision. Fixes: budgets 100 LLM / 80 tool; budget-exhausted message and
  OUTPUT_RULE demand JSON anyway; `_budget_allows` reserves worst case (1 + repairs) per call.
- `logs/virtual_lab_emsr859_result_20261103_093705` (46 min, 2 rounds, 7 invalid, decision
  `insufficient_evidence`, 11-step plan): fixes worked (no prose outputs, Phase D ran).
  New problems:
  - cdse_agent has no read-only catalog tool in plan_only (only geocoding; all fetch_* blocked),
    so EVE answered from its own knowledge: "no 2026 data, future dates". That false claim became a
    disagreement and drove the decision. EUMETSAT meanwhile returned 288 real FCI products.
    → add a CDSE catalog search (OData/STAC, metadata only), give sub-agents today's date,
    and don't record "I don't have access/cannot" replies as retrieved evidence.
  - Tool budget (80) spent by round-1 critique (58 calls in that phase) → later phases blind.
    → per-step tool cap for critique steps.
  - MTG FCI viewing-zenith blocking finding still open; specialists disagree on 60° vs 85°.

## EVE run with catalog tool, date, no_answer, critique limit (2026-11-04)

`logs/virtual_lab_emsr859_result_20261104_200524` (71 min, 2 rounds, `needs_experiment`):
- "future dates" is gone from the conclusions, but `search_cdse_catalog` failed on 22 of 25 calls
  (exact collection names only) and a third returned 0 (`S2MSI2A` vs names with `MSIL2A`). The
  ledger recorded the failures as ok. Result: the false claim "VIIRS/SLSTR unavailable"; the plan
  drops H-timing. Live catalogue: 6 SLSTR FRP passes in the window; FCI 288 and SEVIRI 192 found.
- Fixed afterwards, not re-run: collection and product normalisation, sensors held elsewhere point
  to their agent, 0-result note, and `reports_error` for errors inside MCP replies.
- Critique limit never fired: Phase B used 62 calls, budget gone in round-1 critique. Blocked
  plan-only calls also count against the budget. Needs a proposal-phase limit.
- Plan dates: no Sentinel-1 on 01-18 (nearest SLC 01-17T23:35, 01-22); S2 01-12 is 52-72% cloud
  (01-07 is 9-17%). The plan validator does not check dates against catalog results.

## EVE run with proposal limit, future-date filter, plan date check (2026-11-09)

`logs/virtual_lab_emsr859_result_20261109_135413` (50 min, 2 rounds, `insufficient_evidence`):
- Catalog tool now works (17 of 20 searches ok, 3 bad names such as "S1_GRD"). Plan dates are all
  real acquisitions (S1 01-10/01-17/01-22, S2 01-12/01-17); the date check found nothing to flag.
- Phase B used only 25 tool calls (proposal limit 6 never reached), but round-1 critique used 40,
  and the budget ran out in round-1 revision.
- New failure: after the budget ran out, cdse_agent's leaf searches were blocked and it replied
  "the CDSE catalogue does not contain any Sentinel-2 L2A data" (tc-0083..0087). These replies were
  recorded as ok and cited as retrieved evidence. Critics then used them to overrule correct claims
  (the SAR specialist's S2 01-17 at 0.3% cloud), and the decision says S2, SLSTR and geostationary
  data do not exist. All three exist.
- Fix candidates: a sub-agent reply whose leaf calls were all blocked is no_answer; a deterministic
  check of "X unavailable" claims against catalog results; phase-level tool budgets.

## Kypseli subsidence run and federated data discovery (2026-11-13)

`logs/virtual_lab_app_result_20261113_134514.json` (EVETADP_Orchestrator, 1 round, plan only):
"Athens Metro Line 4 under Kypseli: how can InSAR detect construction-induced settlement?"
- **Study area never located.** The PI skipped `geocode_place`; `study_area.bbox` was null, the
  inventory skipped every scene search, and agents invented bboxes (`[22.98, 38.35, 23.02, 38.40]`,
  ~70 km NW of Athens; `[22.95, 39.95, ...]`, near Olympus). The decision cites 20 of these
  searches as "S1 SLC available". Kypseli is `[23.72, 37.98, 23.76, 38.02]`.
- **No NASA data considered.** Earthdata was never called: no bbox, the inventory's NASA
  products are fire/LST only (`NASA_PRODUCTS`), and the Earthdata tool docstrings give fire examples.
- 15 of 21 CDSE searches used `SENTINEL1_GRD` (amplitude, useless for InSAR); the critic sent
  the same query 7 times. The PI picked only SAR + foundation-model specialists (no atmosphere,
  so no tropospheric correction). Hypotheses restate the question; InSAR steps are assigned to
  `gis_agent`, which cannot do them; EGMS is called "independent" validation.

### Implemented (2026-11-13)
- `coordinator._locate`: when the problem names a study area without a bbox, the lab geocodes it
  in code (`r0.geocode`, free, citable). A PI bbox is kept (it may come from the question; an
  ambiguous name can geocode elsewhere); a mismatch is logged on the call.
- `validation.off_area_calls`: a successful tool call whose bbox misses the study area is a
  major `coverage` issue on that proposal ("not evidence about the study area").

### Problem: the inventory is vocabulary-driven
`inventory.py` selects data by a fixed core list, hard-coded NASA fire/LST products and 4-letter
stem overlap between the question and collection titles (`wanted_stems`, `_FIRE`). It was tuned
on EMSR859 and returns nothing relevant for problems outside that vocabulary (subsidence here).
The provider clients are already federated (CDSE, EUMETSAT, NASA CMR, CDS: metadata only, no
login, bbox + period); the selection on top of them is not.

### Design: discover, rank, verify (no vocabulary in code)
1. **Locate** (done): deterministic geocoding.
2. **Observables** (1 LLM call): the problem becomes a short needs spec: what must be observed
   (e.g. "mm-scale surface deformation, building scale, weekly-monthly, 2023-now"), constraints
   (resolution, revisit, latency, phase/amplitude, record length) and 2-3 search phrases per
   provider. This is where domain knowledge enters, from the model, not from a word list.
3. **Federated collection search**: run every provider's free-text collection search
   concurrently, filtered by bbox and period: CMR `collections.umm_json?keyword=` (exists);
   CDSE STAC / openEO collections; EUMETSAT Data Store collection browse; CDS catalogue.
   (CDSE, EUMETSAT and CDS text-search endpoints: to verify; where a provider has no text
   search, list its collections once per day, cache, and rank locally.) Normalise to one record:
   provider, id, title, abstract, platform/instrument, level, resolution, temporal/spatial extent.
4. **Rank by meaning**: rerank the union (~50-150 candidates) against the needs spec. First
   version: one EVE call over titles + abstract snippets (no new dependency); embeddings (small
   local model) if latency or cost demands it. Keep a per-provider quota so one provider with rich
   metadata cannot crowd out the others.
5. **Verify the shortlist**: granule/scene counts, dates and revisit in bbox + period for the top
   K collections only. This is the availability evidence specialists cite.
6. **One tool for everyone**: `search_catalogues(query, bbox, start, end, providers="all")` runs
   steps 3-5 on demand and replaces the six provider-specific tools, so no specialist-to-catalogue
   routing (`SpecialistSpec.catalogs`) is needed.
7. The fixed core list remains only as a fallback when discovery fails.

### Evaluation (needed to know it generalises)
A small discovery benchmark: ~20 questions across problem types the inventory was not tuned on
(subsidence, landslides, floods, crop monitoring, urban heat, air quality, glaciers, coastal
erosion, methane, drought), each with expert-expected collections per provider. Metric: recall@K
of expected collections in the shortlist, per provider and overall; report separately for fire
(seen) and non-fire (unseen) questions. Run offline against cached catalogue responses so it is
deterministic.

### Files (for approval before implementation)
| File | Change |
|---|---|
| `agents/virtual_lab/discovery.py` (new) | needs spec, federated search, normalise, rank, verify |
| `servers/{cdse_geocode,eumetsat,climate}/catalog.py` | collection text search where missing |
| `agents/virtual_lab/inventory.py` | use the discovery shortlist; drop stems/hard-coded products |
| `agents/virtual_lab/participants.py` | single `search_catalogues` tool; remove per-spec catalogs |
| `agents/virtual_lab/prompts.py` | inventory/tool wording |
| `agents/virtual_lab/coordinator.py` | discovery after `_locate` |
| `tests/test_virtual_lab_discovery.py` (new) + fakes | offline tests per provider |
| `data/virtual_lab/discovery_benchmark.json`, `scripts/eval_discovery.py` (new) | benchmark |

Open questions: does the EVE gateway serve an embedding model (it lists only
`EVETADP_Orchestrator`)? Should GEE and Planetary Computer STAC be providers too?
