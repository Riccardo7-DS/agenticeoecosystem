# Expert evaluation

Human evaluation of the benchmark's final answers — relevance, answer coverage / focus / correctness, and a
ranking of the candidate models per question. This is separate from (and
answers a different question than) the LLM-judge pipeline elsewhere in this
repo, which only checks tool-call sequences, not whether an answer is
actually correct.

It's a small pipeline: a Python script builds the study and a one-page web
app (published as a Claude Artifact) runs it. No servers, no accounts for
the people filling it in — just a link.

## Quickstart

1. **Build the study**
   ```
   python scripts/build_expert_eval.py --experts eoss=4 dashboard=4 mooc=4 general=20
   ```
   Each question goes to one expert group: `eoss`, `dashboard` or `mooc` if
   its reference solution uses the EOSS GraphRAG, EO Dashboard or MOOC
   sub-agent (priority MOOC → EOSS → Dashboard when it uses several),
   otherwise `general`. `N` = how many colleagues in that group each fill in
   one form (the values above are the defaults). More colleagues → shorter
   forms. Writes everything under `expert_eval/out/`. Run `--help` for the
   other flags.

2. **Publish the web app** — ask Claude: *"republish the expert eval app"*.
   (It's `webapp/index.html` plus the `out/forms/*.json` files, published
   together as one Claude Artifact.)

3. **Share it** — in claude.ai, open the artifact's Share menu →
   *"Anyone with the link can view"*. One-time step; skip if already set.

4. **Send the links** — `out/DISTRIBUTION.md` has one link per expert; send
   each group's forms (the "Group" column) to that group's experts. Also
   tell each person their form code in plain text (e.g. "you're F07") —
   sometimes the code gets stripped from the link in transit, and the page
   then just asks for it directly. Track who got which link in that file's
   "Sent to" column.

5. **Collect responses** — each expert downloads a small results file at the
   end and is prompted to email it to you (there's no auto-collection by
   design — see the code comments in `build_expert_eval.py` if curious why).
   Save whatever you receive into `expert_eval/responses/`.

6. **Analyse**
   ```
   python scripts/analyze_expert_eval.py
   ```
   Unblinds the responses, prints a per-model ranking and agreement stats,
   and writes the underlying CSVs to `expert_eval/out/analysis/`.

## Expert groups

Questions are routed by the sub-agent their reference solution uses, so
specialists only see questions in their area. Forms are numbered
consecutively by group. Current build (defaults, `--seed 42`):

| Forms | Group | Routed on | Experts | Questions | Per form | Rated twice |
|---|---|---|---|---|---|---|
| F01–F04 | `mooc` | `mooc_rag_agent` | 4 | 16 | 8 | all 16 |
| F05–F08 | `eoss` | `graph_rag_agent` | 4 | 29 | 8 | 3 |
| F09–F12 | `dashboard` | `eo_stac_agent` | 4 | 38 | 10 | 2 |
| F13–F32 | `general` | none of the above | 20 | 156 | 8 | 4 |

A question that also uses generic agents (CDSE, climate, GIS, …) still goes
to its specialist group. Changing any group size renumbers the forms, so
re-send links after such a rebuild.

## Files

| File | What |
|---|---|
| `scripts/build_expert_eval.py` | Builds the study. All the design logic and reasoning lives in this file's comments — read it before changing behaviour |
| `webapp/index.html` | The survey app (plain HTML/CSS/JS, no build step) |
| `scripts/analyze_expert_eval.py` | Collects + analyses responses |
| `out/DISTRIBUTION.md` | Links to send (with each form's group) + the "Sent to" tracker |
| `out/STUDY_OVERVIEW.md` | Every question/answer in one readable file, for offline reading |
| `out/selected_questions.csv` | One row per question: group, label, agents, which forms carry it |
| `PRIVATE_key.csv` | Blinding key (form, group, letter → model). **Never share with experts** |
| `responses/`, `out/analysis/` | Not in git — your local working files |

## Tweak it

- **Study size / form length**: `--experts GROUP=N ...` and `--items-per-form` (minimum per form, default 8; a group's forms get longer if it has more questions than `N × 8`).
- **Expert groups**: `EXPERT_GROUPS` / `DEFAULT_GROUP_EXPERTS` near the top of the build script.
- **Wording, fields, scoring scales**: the constants near the top of `build_expert_eval.py` (`INTRO_TEXT`, `PROFILE_FIELDS`, `RELEVANCE_*`, `COVERAGE_*`, `FOCUS_*`, `CORRECTNESS_*`, `RANK_*`, `FAMILIARITY_*`) — then rebuild and republish.
- **Which models / tasks are eligible**: `MODEL_RUNS` and `MIN_ANSWERS_TO_INCLUDE` near the top of the same file.
- **Look and feel**: `webapp/index.html` directly.

Any change to the build script needs a rebuild (step 1) **and** a republish
(step 2) — the build script only writes local files, it doesn't touch the
live app.

## Good to know

- Mostly 1 rater per question — every colleague sees a different slice. Spare slots in a group give some questions a second rater (all MOOC questions get two with the defaults). Raise `--items-per-form` for more overlap (needed for the agreement stats in step 6 to mean anything).
- Group and question set coincide: compare models within a group freely, but a difference in mean scores *between* groups mixes rater expertise with question difficulty.
- ~76 of 239 questions include an image; the rest are text-only because the original file no longer exists. Not fixable by rebuilding — see `build_expert_eval.py` for the recovery logic if you want to extend it.
- Names sit next to scores in collected responses. Fine to keep internally; strip that column before sharing results outside the team.
