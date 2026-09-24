# Sending out the forms

Live app: https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo — already shared as "Anyone with the link can view".

Send each expert their own row's link below (one per expert — each opens a different, unique set of questions). Match the expert to the "Group" column: `mooc` = mooc_rag_agent, `eoss` = graph_rag_agent, `dashboard` = eo_stac_agent specialists; `general` = everyone else (questions using none of those agents). **Also tell them their code in plain text** (the "Form" column) — claude.ai's own routing has been observed to strip the `?form=` part of the link before the page loads, in which case the page falls back to a plain "enter your code" box instead of failing outright. Every expert downloads their own results file at the end and emails it back to you (the page prompts for this — nothing is submitted automatically). Drop all the files you receive into `expert_eval/responses/` and run `python scripts/analyze_expert_eval.py` to collect and analyse them — see expert_eval/README.md.

Fill in "Sent to" as you send each link out, to keep track of who has which form. Rebuilding the study preserves whatever you've filled in here, matched by form code -- but only if the form codes still mean the same thing (same --experts/--items-per-form/--seed); a rebuild with different settings reshuffles which questions each code holds, so treat old assignments as stale and re-send in that case.

If you rebuild the study or publish the web app again, re-run this script and re-copy the links below — the JSON specs may have changed.

| Form | Group | Questions | Estimated minutes | Link | Sent to |
|---|---|---|---|---|---|
| F01 | mooc | 8 | 48.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F01 |  |
| F02 | mooc | 8 | 38.1 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F02 |  |
| F03 | mooc | 8 | 48.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F03 |  |
| F04 | mooc | 8 | 41.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F04 |  |
| F05 | eoss | 8 | 30.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F05 |  |
| F06 | eoss | 8 | 31.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F06 |  |
| F07 | eoss | 8 | 40.3 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F07 |  |
| F08 | eoss | 8 | 44.7 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F08 |  |
| F09 | dashboard | 10 | 59.6 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F09 |  |
| F10 | dashboard | 10 | 58.7 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F10 |  |
| F11 | dashboard | 10 | 60.6 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F11 |  |
| F12 | dashboard | 10 | 72.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F12 |  |
| F13 | general | 8 | 48.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F13 |  |
| F14 | general | 8 | 39.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F14 |  |
| F15 | general | 8 | 41.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F15 |  |
| F16 | general | 8 | 40.7 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F16 |  |
| F17 | general | 8 | 32.1 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F17 |  |
| F18 | general | 8 | 40.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F18 |  |
| F19 | general | 8 | 44.4 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F19 |  |
| F20 | general | 8 | 27.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F20 |  |
| F21 | general | 8 | 41.7 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F21 |  |
| F22 | general | 8 | 37.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F22 |  |
| F23 | general | 8 | 53.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F23 |  |
| F24 | general | 8 | 37.3 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F24 |  |
| F25 | general | 8 | 36.6 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F25 |  |
| F26 | general | 8 | 44.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F26 |  |
| F27 | general | 8 | 31.4 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F27 |  |
| F28 | general | 8 | 46.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F28 |  |
| F29 | general | 8 | 39.1 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F29 |  |
| F30 | general | 8 | 45.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F30 |  |
| F31 | general | 8 | 38.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F31 |  |
| F32 | general | 8 | 33.6 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F32 |  |
