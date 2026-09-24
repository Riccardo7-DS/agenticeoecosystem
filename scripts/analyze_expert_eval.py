#!/usr/bin/env python3
"""Collect and analyse expert-evaluation results (see expert_eval/README.md).

Usage:
    python scripts/analyze_expert_eval.py [--responses-dir expert_eval/responses]
                                           [--key expert_eval/PRIVATE_key.csv]
                                           [--out-dir expert_eval/out/analysis]
                                           [--bootstrap 2000] [--seed 42]

Reads every *.json file dropped into --responses-dir (the files experts
download from the web app and email back), unblinds letter -> model using
PRIVATE_key.csv, and reports:

    * per-model accuracy / rank summary,
    * a full 6-model ranking via Plackett-Luce (MM algorithm) with a
      percentile bootstrap CI, resampled over (question, rater) rankings,
    * Krippendorff's alpha (ordinal) for accuracy and for relevance,
    * Kendall's W (and mean pairwise Kendall tau) for agreement on ranking,
    * questions whose mean relevance is below 3, for the organiser to review.

Also writes long-format CSVs under --out-dir for further analysis in a
spreadsheet or notebook. Only the standard library is used.
"""

import argparse
import collections
import csv
import glob
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSES_DIR = ROOT_DIR / "expert_eval" / "responses"
DEFAULT_KEY = ROOT_DIR / "expert_eval" / "PRIVATE_key.csv"
DEFAULT_OUT_DIR = ROOT_DIR / "expert_eval" / "out" / "analysis"

ACCURACY_CATEGORIES = [1, 2, 3, 4, 5]
RELEVANCE_CATEGORIES = [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_key(path):
    """(form, item position) -> list of {letter, model, task_id, label}, in key-file order."""
    key = collections.defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key[(row["form"], int(row["item"]))].append({
                "letter": row["letter"], "model": row["model"],
                "task_id": row["task_id"], "label": row["label"],
            })
    return key


def load_responses(responses_dir):
    files = sorted(glob.glob(str(Path(responses_dir) / "*.json")))
    responses = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {fp}: could not read ({e})")
            continue
        data["_source_file"] = fp
        responses.append(data)
    return responses


# --------------------------------------------------------------------------
# Unblinding + validation
# --------------------------------------------------------------------------
def unblind(responses, key):
    """Return (answer_ratings, question_ratings, problems).

    answer_ratings: one row per (response, item, model) -- accuracy + rank.
    question_ratings: one row per (response, item) -- relevance/familiarity/comment.
    """
    answer_ratings, question_ratings, problems = [], [], []

    for resp in responses:
        form = resp.get("form_code", "")
        rater = (resp.get("name") or "").strip() or "(unnamed)"
        src = resp.get("_source_file", "?")
        for item in resp.get("items", []):
            position = item.get("position")
            key_rows = key.get((form, position))
            if not key_rows:
                problems.append(f"{src}: no key entry for form={form} item={position} (skipped)")
                continue
            letter_to_model = {r["letter"]: r["model"] for r in key_rows}
            task_id = key_rows[0]["task_id"]
            label = key_rows[0]["label"]
            letters = set(letter_to_model)

            rank = item.get("rank_best_to_worst")
            if len(letters) < 2:
                rank = None  # nothing to rank when only one model answered -- not an error
            elif not rank or set(rank) != letters or len(rank) != len(letters):
                problems.append(f"{src}: form={form} item={position} ranking isn't a permutation of {sorted(letters)} (skipped)")
                rank = None

            accuracy = item.get("accuracy") or {}
            missing_acc = letters - set(accuracy)
            if missing_acc:
                problems.append(f"{src}: form={form} item={position} missing accuracy for {sorted(missing_acc)}")

            question_ratings.append({
                "task_id": task_id, "label": label, "form": form, "rater": rater,
                "relevance": item.get("relevance"),
                "familiarity": item.get("familiarity"),
                "comment": item.get("comment") or "",
            })

            for letter, model in letter_to_model.items():
                acc = accuracy.get(letter)
                if acc is not None and acc not in ACCURACY_CATEGORIES:
                    problems.append(f"{src}: form={form} item={position} accuracy[{letter}]={acc!r} out of range (dropped)")
                    acc = None
                r = (rank.index(letter) + 1) if rank else None
                if acc is None and r is None:
                    continue
                answer_ratings.append({
                    "task_id": task_id, "label": label, "form": form, "rater": rater,
                    "model": model, "letter": letter, "accuracy": acc, "rank": r,
                })
    return answer_ratings, question_ratings, problems


# --------------------------------------------------------------------------
# Per-model summary
# --------------------------------------------------------------------------
def model_summary(answer_ratings, models):
    by_model = collections.defaultdict(lambda: {"acc": [], "rank": []})
    for row in answer_ratings:
        if row["accuracy"] is not None:
            by_model[row["model"]]["acc"].append(row["accuracy"])
        if row["rank"] is not None:
            by_model[row["model"]]["rank"].append(row["rank"])

    rows = []
    for model in models:
        acc, rank = by_model[model]["acc"], by_model[model]["rank"]
        rows.append({
            "model": model,
            "n_accuracy": len(acc),
            "mean_accuracy": round(statistics.mean(acc), 3) if acc else None,
            "sd_accuracy": round(statistics.pstdev(acc), 3) if len(acc) > 1 else None,
            "n_ranked": len(rank),
            "mean_rank": round(statistics.mean(rank), 3) if rank else None,
            "top1_rate": round(sum(r == 1 for r in rank) / len(rank), 3) if rank else None,
        })
    return rows


# --------------------------------------------------------------------------
# Plackett-Luce (MM algorithm, Hunter 2004) + bootstrap CI
# --------------------------------------------------------------------------
def collect_rankings(answer_ratings, models):
    """(form, task_id, rater) -> ranking (list of models, best to worst).

    Questions vary in how many models answered them (see build_expert_eval.py),
    so a "complete ranking" here means complete for THAT question, not
    necessarily covering all of `models` -- Plackett-Luce natively supports
    rankings over different item subsets, it doesn't require every ranking to
    span the full model roster.
    """
    by_key = collections.defaultdict(dict)
    for row in answer_ratings:
        if row["rank"] is not None:
            by_key[(row["form"], row["task_id"], row["rater"])][row["rank"]] = row["model"]
    rankings = []
    for ranks in by_key.values():
        n = len(ranks)
        if n >= 2 and set(ranks) == set(range(1, n + 1)):
            rankings.append([ranks[k] for k in range(1, n + 1)])
    return rankings


def plackett_luce_mm(rankings, items, n_iter=300, tol=1e-10):
    """Fit Plackett-Luce strengths (mean-normalised to 1) via the MM algorithm."""
    gamma = {it: 1.0 for it in items}
    if not rankings:
        return gamma
    wins = {it: 0 for it in items}
    for ranking in rankings:
        for it in ranking[:-1]:  # last place never "wins" a stage
            wins[it] += 1

    for _ in range(n_iter):
        denom = {it: 0.0 for it in items}
        for ranking in rankings:
            for k in range(len(ranking) - 1):
                still_in = ranking[k:]
                total = sum(gamma[j] for j in still_in)
                if total <= 0:
                    continue
                for j in still_in:
                    denom[j] += 1.0 / total
        new_gamma = {}
        for it in items:
            new_gamma[it] = (wins[it] / denom[it]) if denom[it] > 0 else gamma[it]
        norm = sum(new_gamma.values()) / len(items)
        if norm > 0:
            new_gamma = {it: v / norm for it, v in new_gamma.items()}
        diff = max(abs(new_gamma[it] - gamma[it]) for it in items)
        gamma = new_gamma
        if diff < tol:
            break
    return gamma


def bootstrap_pl(rankings, items, n_boot, rng):
    samples = {it: [] for it in items}
    n = len(rankings)
    if n == 0:
        return {it: (float("nan"), float("nan")) for it in items}
    for _ in range(n_boot):
        resample = [rankings[rng.randrange(n)] for _ in range(n)]
        gamma = plackett_luce_mm(resample, items, n_iter=100)
        for it in items:
            samples[it].append(gamma[it])
    ci = {}
    for it in items:
        vals = sorted(samples[it])
        lo = vals[int(0.025 * len(vals))]
        hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
        ci[it] = (lo, hi)
    return ci


# --------------------------------------------------------------------------
# Krippendorff's alpha (ordinal), coincidence-matrix formulation
# --------------------------------------------------------------------------
def krippendorff_alpha_ordinal(units, categories):
    """units: iterable of rating-lists (one list per unit, entries in `categories`,
    one list per unit; units with fewer than 2 ratings are ignored)."""
    cat_index = {c: i for i, c in enumerate(categories)}
    ncat = len(categories)
    coincidence = [[0.0] * ncat for _ in range(ncat)]
    for ratings in units:
        ratings = [r for r in ratings if r is not None]
        m = len(ratings)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                a, b = cat_index[ratings[i]], cat_index[ratings[j]]
                coincidence[a][b] += 1.0 / (m - 1)

    n_v = [sum(coincidence[v]) for v in range(ncat)]
    n = sum(n_v)
    if n < 2:
        return None

    def delta2(v, w):
        if v > w:
            v, w = w, v
        s = sum(n_v[g] for g in range(v, w + 1)) - (n_v[v] + n_v[w]) / 2.0
        return s * s

    Do = sum(coincidence[v][w] * delta2(v, w) for v in range(ncat) for w in range(ncat)) / n
    De_num = sum(n_v[v] * n_v[w] * delta2(v, w) for v in range(ncat) for w in range(ncat))
    De = De_num / (n * (n - 1))
    if De == 0:
        return 1.0 if Do == 0 else 0.0
    return 1.0 - Do / De


# --------------------------------------------------------------------------
# Kendall's W (per question, averaged) and mean pairwise tau
# --------------------------------------------------------------------------
def kendall_w(rank_lists):
    """rank_lists: list of {item: rank} dicts (one per rater) for ONE question,
    all sharing the same item set and no ties."""
    m = len(rank_lists)
    if m < 2:
        return None
    items = list(rank_lists[0])
    n = len(items)
    totals = {it: sum(rl[it] for rl in rank_lists) for it in items}
    mean_total = m * (n + 1) / 2
    S = sum((totals[it] - mean_total) ** 2 for it in items)
    denom = m * m * (n ** 3 - n)
    if denom == 0:
        return None
    return 12 * S / denom


def kendall_tau_pair(rank_a, rank_b, items):
    """Kendall tau-b for two strict rankings (dicts item->rank) over the same items."""
    pairs = list(itertools.combinations(items, 2))
    concordant = discordant = 0
    for i, j in pairs:
        sign_a = rank_a[i] - rank_a[j]
        sign_b = rank_b[i] - rank_b[j]
        if sign_a * sign_b > 0:
            concordant += 1
        elif sign_a * sign_b < 0:
            discordant += 1
    total = len(pairs)
    return (concordant - discordant) / total if total else None


def ranking_agreement(answer_ratings, models):
    """Per-question rank dicts -> mean Kendall's W and mean pairwise tau across questions.

    Grouped by task_id alone (not form): the same question is rated on three
    DIFFERENT forms, one per round, so a (form, task_id) key would never see
    more than one rater and agreement could never be computed. `models` is
    the full candidate roster, used only as a fallback item order -- the
    actual item set compared is whatever models that specific question's
    raters actually ranked (see build_expert_eval.py: not every question has
    all of `models` answering it).
    """
    by_question = collections.defaultdict(lambda: collections.defaultdict(dict))
    for row in answer_ratings:
        if row["rank"] is not None:
            by_question[row["task_id"]][row["rater"]][row["model"]] = row["rank"]

    ws, taus = [], []
    for raters in by_question.values():
        candidates = [rd for rd in raters.values() if len(rd) >= 2]
        if len(candidates) < 2:
            continue
        # Raters of the same question should all see the same model subset by
        # construction; keep only those agreeing on which subset, in case a
        # malformed response slipped through.
        by_keyset = collections.Counter(frozenset(rd) for rd in candidates)
        keyset, _ = by_keyset.most_common(1)[0]
        rank_dicts = [rd for rd in candidates if frozenset(rd) == keyset]
        if len(rank_dicts) < 2:
            continue
        items_here = list(keyset)
        w = kendall_w(rank_dicts)
        if w is not None:
            ws.append(w)
        for a, b in itertools.combinations(rank_dicts, 2):
            t = kendall_tau_pair(a, b, items_here)
            if t is not None:
                taus.append(t)
    return (
        round(statistics.mean(ws), 3) if ws else None, len(ws),
        round(statistics.mean(taus), 3) if taus else None, len(taus),
    )


# --------------------------------------------------------------------------
# Relevance
# --------------------------------------------------------------------------
def relevance_summary(question_ratings):
    by_q = collections.defaultdict(list)
    labels = {}
    for row in question_ratings:
        if row["relevance"] is not None:
            by_q[row["task_id"]].append(row["relevance"])
            labels[row["task_id"]] = row["label"]
    out = []
    for task_id, scores in by_q.items():
        out.append({"task_id": task_id, "label": labels[task_id], "n": len(scores),
                     "mean_relevance": round(statistics.mean(scores), 2)})
    out.sort(key=lambda r: r["mean_relevance"])
    return out


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------
def write_csv(rows, fieldnames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses-dir", default=str(DEFAULT_RESPONSES_DIR))
    parser.add_argument("--key", default=str(DEFAULT_KEY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for the Plackett-Luce CI")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    responses_dir.mkdir(parents=True, exist_ok=True)
    key_path = Path(args.key)
    out_dir = Path(args.out_dir)

    if not key_path.exists():
        sys.exit(f"No blinding key at {key_path} -- run scripts/build_expert_eval.py first.")

    key = load_key(key_path)
    models = sorted({r["model"] for rows in key.values() for r in rows})

    responses = load_responses(responses_dir)
    print(f"Found {len(responses)} response file(s) in {responses_dir}")
    if not responses:
        print("Nothing to analyse yet. Drop the .json files experts email you into that folder and re-run.")
        return

    answer_ratings, question_ratings, problems = unblind(responses, key)
    if problems:
        print(f"\n{len(problems)} issue(s) while unblinding (rows involved were skipped/partial):")
        for p in problems[:30]:
            print("  -", p)
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more")

    raters = sorted({r["rater"] for r in question_ratings})
    print(f"\n{len(raters)} rater(s): {', '.join(raters)}")

    # --- per-model summary -------------------------------------------------
    summary = model_summary(answer_ratings, models)
    print("\nPer-model summary (accuracy 1-5, rank 1=best..%d=worst)" % len(models))
    print(f"{'model':<22}{'n_acc':>7}{'mean_acc':>10}{'n_rank':>8}{'mean_rank':>11}{'top1_rate':>11}")
    for row in summary:
        print(f"{row['model']:<22}{row['n_accuracy']:>7}{('' if row['mean_accuracy'] is None else row['mean_accuracy']):>10}"
              f"{row['n_ranked']:>8}{('' if row['mean_rank'] is None else row['mean_rank']):>11}"
              f"{('' if row['top1_rate'] is None else row['top1_rate']):>11}")

    # --- Plackett-Luce full ranking -----------------------------------------
    rankings = collect_rankings(answer_ratings, models)
    gamma = plackett_luce_mm(rankings, models)
    rng = random.Random(args.seed)
    ci = bootstrap_pl(rankings, models, args.bootstrap, rng) if rankings else {}
    pl_rows = []
    for model in sorted(models, key=lambda m: -gamma[m]):
        lo, hi = ci.get(model, (float("nan"), float("nan")))
        pl_rows.append({"model": model, "strength": round(gamma[model], 4),
                         "ci_low": round(lo, 4) if lo == lo else None,
                         "ci_high": round(hi, 4) if hi == hi else None})
    print(f"\nPlackett-Luce ranking from {len(rankings)} complete ranking(s) (higher strength = better; "
          f"{args.bootstrap}x bootstrap 95% CI over rankings):")
    for row in pl_rows:
        print(f"  {row['model']:<22} strength={row['strength']:<8} 95% CI [{row['ci_low']}, {row['ci_high']}]")

    # --- agreement -----------------------------------------------------------
    # Units are keyed by task_id alone (not form): a question is rated on
    # three DIFFERENT forms, one per round, so including "form" in the key
    # would put each rating in its own singleton unit and agreement would
    # always come out undefined.
    acc_units = collections.defaultdict(list)
    for row in answer_ratings:
        acc_units[(row["task_id"], row["model"])].append(row["accuracy"])
    rel_units = collections.defaultdict(list)
    for row in question_ratings:
        rel_units[row["task_id"]].append(row["relevance"])

    alpha_acc = krippendorff_alpha_ordinal(acc_units.values(), ACCURACY_CATEGORIES)
    alpha_rel = krippendorff_alpha_ordinal(rel_units.values(), RELEVANCE_CATEGORIES)
    mean_w, n_w, mean_tau, n_tau = ranking_agreement(answer_ratings, models)

    print("\nAgreement")
    print(f"  Krippendorff's alpha, accuracy:  {alpha_acc}")
    print(f"  Krippendorff's alpha, relevance: {alpha_rel}")
    print(f"  Mean Kendall's W across {n_w} question(s) with >=2 raters: {mean_w}")
    print(f"  Mean pairwise Kendall's tau across {n_tau} rater pair(s): {mean_tau}")
    print("  (alpha/W/tau near 0 = chance agreement, 1 = perfect agreement; alpha below ~0.67 "
          "is usually considered too low to draw firm conclusions from)")

    # --- relevance flags -------------------------------------------------------
    rel_rows = relevance_summary(question_ratings)
    low_rel = [r for r in rel_rows if r["mean_relevance"] < 3]
    if low_rel:
        print(f"\n{len(low_rel)} question(s) flagged with mean relevance < 3 (consider for GT review):")
        for r in low_rel:
            print(f"  {r['task_id']} ({r['label']}): mean relevance {r['mean_relevance']} over {r['n']} rating(s)")
    else:
        print("\nNo questions flagged with mean relevance < 3.")

    # --- write CSVs --------------------------------------------------------
    write_csv(answer_ratings, ["task_id", "label", "form", "rater", "model", "letter", "accuracy", "rank"],
              out_dir / "answer_ratings.csv")
    write_csv(question_ratings, ["task_id", "label", "form", "rater", "relevance", "familiarity", "comment"],
              out_dir / "question_ratings.csv")
    write_csv(summary, ["model", "n_accuracy", "mean_accuracy", "sd_accuracy", "n_ranked", "mean_rank", "top1_rate"],
              out_dir / "model_summary.csv")
    write_csv(pl_rows, ["model", "strength", "ci_low", "ci_high"], out_dir / "model_ranking_plackett_luce.csv")
    write_csv(rel_rows, ["task_id", "label", "n", "mean_relevance"], out_dir / "relevance_by_question.csv")
    print(f"\nCSVs written to {out_dir}")


if __name__ == "__main__":
    main()
