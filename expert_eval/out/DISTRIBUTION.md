# Sending out the forms

Live app: https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo — already shared as
"Anyone with the link can view".

Send each expert their own row's link below (one per expert — each opens a different, unique set of questions). **Also tell them their code in plain text** (the "Form" column) — claude.ai's own routing has been observed to strip the `?form=` part of the link before the page loads, in which case the page falls back to a plain "enter your code" box instead of failing outright. Every expert downloads their own results file at the end and emails it back to you (the page prompts for this — nothing is submitted automatically). Drop all the files you receive into `expert_eval/responses/` and run `python scripts/analyze_expert_eval.py` to collect and analyse them — see expert_eval/README.md.

Fill in "Sent to" as you send each link out, to keep track of who has which form. Rebuilding the study preserves whatever you've filled in here, matched by form code -- but only if the form codes still mean the same thing (same --n-experts/--seed); a rebuild with different settings reshuffles which questions each code holds, so treat old assignments as stale and re-send in that case.

If you rebuild the study or publish the web app again, re-run this script and re-copy the links below — the JSON specs may have changed.

| Form | Questions | Estimated minutes | Link | Sent to |
|---|---|---|---|---|
| F01 | 8 | 44.7 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F01 | Riccardo |
| F02 | 8 | 37.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F02 | Kaushik |
| F03 | 8 | 36.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F03 | Eva |
| F04 | 8 | 33.1 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F04 |  |
| F05 | 8 | 51.6 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F05 |  |
| F06 | 8 | 34.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F06 |  |
| F07 | 8 | 44.4 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F07 |  |
| F08 | 8 | 40.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F08 |  |
| F09 | 8 | 45.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F09 |  |
| F10 | 8 | 43.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F10 |  |
| F11 | 8 | 49.4 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F11 |  |
| F12 | 8 | 39.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F12 |  |
| F13 | 8 | 45.4 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F13 |  |
| F14 | 8 | 46.4 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F14 |  |
| F15 | 8 | 33.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F15 |  |
| F16 | 8 | 53.3 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F16 |  |
| F17 | 8 | 48.8 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F17 |  |
| F18 | 8 | 35.9 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F18 |  |
| F19 | 8 | 49.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F19 |  |
| F20 | 8 | 41.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F20 |  |
| F21 | 8 | 46.3 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F21 |  |
| F22 | 8 | 43.6 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F22 |  |
| F23 | 8 | 42.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F23 |  |
| F24 | 8 | 39.1 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F24 |  |
| F25 | 8 | 42.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F25 |  |
| F26 | 8 | 36.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F26 |  |
| F27 | 8 | 41.2 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F27 |  |
| F28 | 8 | 36.5 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F28 |  |
| F29 | 8 | 31.1 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F29 |  |
| F30 | 8 | 37.0 | https://claude.ai/artifact/ABPNVA5fYPtHPtjj3DkKXo?form=F30 |  |
