# Expert evaluation

Human evaluation of the benchmark's final answers — relevance, accuracy, and a
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
   python scripts/build_expert_eval.py --n-experts <N>
   ```
   `<N>` = how many colleagues will each fill in one form. More colleagues →
   shorter forms (questions are split across everyone). Writes everything
   under `expert_eval/out/`. Run `--help` for the other flags.

2. **Publish the web app** — ask Claude: *"republish the expert eval app"*.
   (It's `webapp/index.html` plus the `out/forms/*.json` files, published
   together as one Claude Artifact.)

3. **Share it** — in claude.ai, open the artifact's Share menu →
   *"Anyone with the link can view"*. One-time step; skip if already set.

4. **Send the links** — `out/DISTRIBUTION.md` has one link per expert. Also
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

## Files

| File | What |
|---|---|
| `scripts/build_expert_eval.py` | Builds the study. All the design logic and reasoning lives in this file's comments — read it before changing behaviour |
| `webapp/index.html` | The survey app (plain HTML/CSS/JS, no build step) |
| `scripts/analyze_expert_eval.py` | Collects + analyses responses |
| `out/DISTRIBUTION.md` | Links to send + the "Sent to" tracker |
| `out/STUDY_OVERVIEW.md` | Every question/answer in one readable file, for offline reading |
| `PRIVATE_key.csv` | Blinding key (letter → model). **Never share with experts** |
| `responses/`, `out/analysis/` | Not in git — your local working files |

## Tweak it

- **Study size / form length**: `--n-experts`, `--items-per-form` on the build script.
- **Wording, fields, scoring scales**: the constants near the top of `build_expert_eval.py` (`INTRO_TEXT`, `PROFILE_FIELDS`, `RELEVANCE_*`, `ACCURACY_*`, `RANK_*`, `FAMILIARITY_*`) — then rebuild and republish.
- **Which models / tasks are eligible**: `MODEL_RUNS` and `MIN_ANSWERS_TO_INCLUDE` near the top of the same file.
- **Look and feel**: `webapp/index.html` directly.

Any change to the build script needs a rebuild (step 1) **and** a republish
(step 2) — the build script only writes local files, it doesn't touch the
live app.

## Good to know

- Default is 1 rater per question — every colleague sees a different slice, so there's no built-in second opinion. Raise `--items-per-form` if you want overlap (needed for the agreement stats in step 6 to mean anything).
- ~76 of 239 questions include an image; the rest are text-only because the original file no longer exists. Not fixable by rebuilding — see `build_expert_eval.py` for the recovery logic if you want to extend it.
- Names sit next to scores in collected responses. Fine to keep internally; strip that column before sharing results outside the team.
