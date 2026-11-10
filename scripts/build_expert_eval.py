#!/usr/bin/env python3
"""Build the expert-evaluation study (a self-contained web form) from the benchmark.

Pipeline: pool filtering -> routing to expert groups -> coverage-balanced
batching within each group (one batch = one expert form) -> exports.

Usage:
    python scripts/build_expert_eval.py [--experts eoss=4 dashboard=4 mooc=4 general=20]
                                        [--items-per-form 8] [--seed 42]

Design (see expert_eval/README.md):
    * a question is eligible as soon as AT LEAST ONE candidate model produced
      a usable answer for it,
    * EVERY eligible question is included in the study -- there is no
      subset-selection step,
    * every question shows every model that actually answered it (no
      placeholder slots for models that didn't), so answers per question
      varies (1 to len(MODELS)); ranking is only asked when 2+ answers exist,
    * every question belongs to exactly one expert group (see
      EXPERT_GROUPS): a specialist group if its reference solution uses that
      group's sub-agent, otherwise the general group. Each group's questions
      go only to that group's forms,
    * one form per expert. Within a group, every form has the same length:
      `--items-per-form`, or more if that is needed for every question in
      the group to get at least one form; any slack from
      `n_experts x items_per_form` beyond the group's pool size is spent
      giving as many questions as possible a second (or further) rater,
      spread evenly rather than concentrated.

Outputs (all under expert_eval/):
    out/STUDY_OVERVIEW.md          human-readable study summary + full text dump
    out/selected_questions.csv     study design overview (no model identities)
    out/forms/form_XX.json         machine-readable form specs (read by the web app)
    out/DISTRIBUTION.md            per-expert links to send out (fill in the base URL)
    PRIVATE_key.csv                blinding key (letter -> model). Never share.

Only the standard library is used.
"""

import argparse
import base64
import collections
import csv
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TASKS_FILE = ROOT_DIR / "data" / "tasks" / "benchmark_tasks.json"
RESULTS_DIR = ROOT_DIR / "output" / "ESAEOBench"
EVAL_DIR = ROOT_DIR / "expert_eval"
OUT_DIR = EVAL_DIR / "out"
COVERAGE_SCRIPT = ROOT_DIR / ".claude" / "skills" / "eomcpevaluation" / "scripts" / "analyze_coverage.py"
VLM_SERVER = ROOT_DIR / "src" / "agenticeoecosystem" / "servers" / "vlm_tools" / "server.py"

# Candidate models: display label -> run directory under output/ESAEOBench
MODEL_RUNS = {
    "EVE-Instruct": "eve_run_20260817_075802",
    "Gemma4-31B": "gemma4b",
    "Mistral-Large-2512": "mistral_run_20260914_123522",
    "Mistral-Small-2603": "mistral_small4_run_20260917_084350",
    "Mistral-Small-2506": "mistral_small_run_20260916_094331",
    "Qwen3.6-27B": "qwen27b_run_20260818_082329",
}
MODELS = list(MODEL_RUNS)

LABELS = ["Tool Grounding", "Sequential Reasoning", "Cross-agent Synthesis"]

MIN_ANSWERS_TO_INCLUDE = 1        # a question is eligible with at least this many usable answers
LETTERS = "ABCDEF"
assert len(LETTERS) >= len(MODELS)

# Reading-load / time model (informational -- printed per form, not enforced;
# with every eligible question in the study, a form's length is driven by
# --experts/--items-per-form, not by a fixed per-form budget).
READING_WPM = 200
INTRO_MINUTES = 1.5
JUDGEMENT_MINUTES_PER_ITEM = 1.5

# Sanity bound on answer length shown in the form.
MAX_ANSWER_CHARS = 3500
# Degenerate outputs such as "(continuing)" are not answers; they are excluded
# from the expert study (report their frequency separately).
MIN_ANSWER_WORDS = 8

# Some answers describe an image the model looked at (via load_images/
# resolve_images, or a GeoTIFF written by netcdf_to_geotiff etc.); when that
# file still exists on disk, it's included inline so experts aren't judging
# a description of a picture they can't see. Most don't survive -- these
# runs are months old and the tool's default download locations aren't
# meant to be permanent storage.
_IMAGE_PATH_RE = re.compile(r'"(/[^"\s]+\.(?:jpg|jpeg|png|tif|tiff))"', re.IGNORECASE)
_IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
MIN_IMAGE_BYTES = 2000  # skip empty/corrupt files
MAX_TIF_BYTES = 100_000_000  # skip absurdly large rasters (e.g. an unclipped full-scene merge)
MAX_TIF_PIXELS = 480  # render at most this many pixels per side -- a preview, not a full-res copy
MAX_IMAGES_PER_ANSWER = 2  # every image is embedded inline (base64) in the form's own JSON; keep this tight
_INDEX_NAME_HINTS = ("ndvi", "ndwi", "ndmi", "nbr", "evi", "savi")


_SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__"}
_INDEXABLE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def build_image_basename_index(root):
    """basename -> absolute path for every jpg/png/tif under `root`.

    Recorded tool-call paths go stale not just from the 2026-09-16 "derouted
    output" reorg but from ordinary manual tidying (images moved to other
    folders after a run). These filenames carry a random hash or a
    timestamp, so a basename match elsewhere in the repo is safe to trust --
    collisions are checked for below, not assumed away. Uses os.walk with
    directory pruning (not Path.rglob) so it doesn't crawl .venv/.git.
    """
    index = {}
    collisions = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            if filename.lower().endswith(_INDEXABLE_EXTS):
                path = Path(dirpath) / filename
                if filename in index and index[filename] != path:
                    collisions.add(filename)
                index[filename] = path
    for name in collisions:
        del index[name]  # ambiguous -- don't guess which copy is the real one
    return index, collisions


def _resolve_image_path(path, basename_index):
    """A 2026-09-16 repo reorg ("derouted output") moved old output/<x>
    downloads to output/old/<x> without updating any path recorded in a
    result file; check that, then fall back to a basename match anywhere
    under the repo (see build_image_basename_index) for images that were
    manually moved/reorganised since."""
    for candidate in (path, path.replace("/output/", "/output/old/", 1) if "/output/" in path else None):
        if candidate and os.path.isfile(candidate) and os.path.getsize(candidate) > MIN_IMAGE_BYTES:
            return candidate
    by_name = basename_index.get(os.path.basename(path))
    if by_name and by_name.stat().st_size > MIN_IMAGE_BYTES:
        return str(by_name)
    return None


def render_geotiff_preview(path):
    """Render a GeoTIFF to JPEG bytes, or None if it can't be rendered
    honestly without guessing what its bands mean.

    - 3 bands: treated as an RGB composite (what a "true colour" fetch
      produces), percentile-stretched per band.
    - 1 band: rendered as plain grayscale intensity, UNLESS the filename
      names a known vegetation/water index (ndvi, ndwi, ...), in which case
      it's rendered with a diverging colormap on a [-1, 1] scale, since
      that's the actual convention those indices are defined on.
    - Anything else (2, 4, 12 bands, ...): skipped. Guessing a band order
      for an arbitrary raw multi-band download risks showing a plausible
      but wrong picture, which is worse than showing none.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image
    import matplotlib as mpl
    import io
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. NotGeoreferencedWarning -- irrelevant for a preview
            with rasterio.open(path) as src:
                if src.count not in (1, 3):
                    return None
                scale = min(1.0, MAX_TIF_PIXELS / max(src.width, src.height))
                out_shape = (src.count, max(1, round(src.height * scale)), max(1, round(src.width * scale)))
                data = src.read(out_shape=out_shape, resampling=Resampling.average).astype("float64")
    except Exception:
        return None  # corrupt, unreadable, or an unsupported raster format

    nodata_mask = ~np.isfinite(data)
    if nodata_mask.all():
        return None
    data = np.where(nodata_mask, np.nan, data)

    def stretch(band, lo=2, hi=98):
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            return np.zeros_like(band)
        p_lo, p_hi = np.nanpercentile(valid, [lo, hi])
        if p_hi <= p_lo:
            return np.zeros_like(band)
        return np.clip((band - p_lo) / (p_hi - p_lo), 0, 1)

    if data.shape[0] == 3:
        rgb = np.stack([stretch(data[i]) for i in range(3)], axis=-1)
        rgb = np.nan_to_num(rgb)
        arr = (rgb * 255).astype("uint8")
    else:
        band = data[0]
        name = os.path.basename(path).lower()
        if any(hint in name for hint in _INDEX_NAME_HINTS):
            normalized = np.clip((np.nan_to_num(band, nan=0.0) + 1) / 2, 0, 1)  # index convention: [-1, 1]
            cmap = mpl.colormaps["RdYlGn"]
        else:
            normalized = np.nan_to_num(stretch(band))
            cmap = mpl.colormaps["gray"]
        arr = (cmap(normalized)[:, :, :3] * 255).astype("uint8")

    # JPEG, not PNG: this is a percentile-stretched preview, not data anyone
    # needs pixel-exact -- JPEG is far smaller for continuous-tone imagery,
    # which matters since every image ships as base64 inside the form's own
    # JSON file (no separate asset store, see the "why" in the commit log).
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=78, optimize=True)
    return buf.getvalue()


def find_answer_images(steps, basename_index, render_cache):
    """{"src": data URI, "rendered": bool} for viewable images referenced in
    a solution's tool-call steps, in step order, deduplicated. "rendered" is
    true for a GeoTIFF converted to a preview here (see
    render_geotiff_preview) rather than the literal file the model produced
    -- the web app flags these differently so experts don't mistake a
    percentile-stretched/colormapped approximation for the original.
    `render_cache` is a plain dict shared across the whole build so the same
    file (jpg/png read, or GeoTIFF render) is never redone."""
    found, seen = [], set()
    for path in _IMAGE_PATH_RE.findall(json.dumps(steps)):
        if path in seen:
            continue
        seen.add(path)
        resolved = _resolve_image_path(path, basename_index)
        if not resolved:
            continue
        if resolved not in render_cache:
            render_cache[resolved] = image_data_uri(resolved)
        entry = render_cache[resolved]
        if entry:
            found.append(entry)
        if len(found) >= MAX_IMAGES_PER_ANSWER:
            break
    return found


def image_data_uri(path):
    """{"src": data URI, "rendered": bool}, or None if it's a GeoTIFF that
    can't be rendered honestly (see render_geotiff_preview)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        if os.path.getsize(path) > MAX_TIF_BYTES:
            return None
        jpeg_bytes = render_geotiff_preview(path)
        if jpeg_bytes is None:
            return None
        src = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")
        return {"src": src, "rendered": True}
    mime = _IMAGE_MIME[ext]
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"src": f"data:{mime};base64,{b64}", "rendered": False}

# Ambiguity of the `GT` flag is documented in README; 0 = flagged incorrect.
GT_EXCLUDED = {0}
GT_VERIFIED = 1

# Tools missing from the coverage script's inventory, mapped to sub-agents.
EXTRA_TOOL_TO_AGENT = {
    "search_eoss": "graph_rag_agent",
    "query_ontology": "graph_rag_agent",
    "list_ontology_nodes": "graph_rag_agent",
    "compute_nbr": "gis_agent",
    "compute_ndwi": "gis_agent",
    "compute_ndmi": "gis_agent",
    "compute_evi": "gis_agent",
    "clip_raster_with_bbox": "gis_agent",
    "project_geometry": "gis_agent",
}
# Coverage-script server key -> sub-agent (local servers listed first, so a tool
# name shared with an external server, e.g. compute_s2_ndvi, maps to the local one).
SERVER_TO_AGENT = {
    "eodashboard": "eo_stac_agent",
    "gee-mcp": "gee_agent",
    "moocs": "mooc_rag_agent",
    "eoss_graphrag": "graph_rag_agent",
    "geos_ops": "eumetsat_agent",
    "climate": "climate_agent",
    "cdse": "cdse_agent",
    "arxiv-mcp-server": "arxiv_rag_agent",
    "gis-mcp": "gis_agent",
}
VLM_AGENT = "vlm_agent"
AGENTS = sorted(set(SERVER_TO_AGENT.values()) | {VLM_AGENT})

# Expert groups. A question goes to a specialist group if its reference
# solution uses that group's sub-agent (even alongside generic agents), else to
# GENERAL_GROUP. Dict order is the routing priority for questions using several
# specialist agents: smallest group first, so it doesn't run short of questions.
EXPERT_GROUPS = {
    "mooc": {"mooc_rag_agent"},
    "eoss": {"graph_rag_agent"},
    "dashboard": {"eo_stac_agent"},
}
GENERAL_GROUP = "general"
GROUPS = [*EXPERT_GROUPS, GENERAL_GROUP]
DEFAULT_GROUP_EXPERTS = {"mooc": 4, "eoss": 4, "dashboard": 4, "general": 20}
assert set(DEFAULT_GROUP_EXPERTS) == set(GROUPS)

# Identity strings that must not appear in what experts read.
IDENTITY_PATTERN = re.compile(
    r"\b(EVE|Gemma\S*|Mistral\S*|Qwen\S*|Pi ?School|Earth Virtual Expert|OpenAI|GPT\S*|Claude|Llama\S*)\b",
    re.IGNORECASE,
)
LEAK_PATTERNS = {
    "absolute path": re.compile(r"(?:/shared/|/home/|/tmp/|/mnt/|tool_output/)"),
    "identity": IDENTITY_PATTERN,
    "tool error marker": re.compile(r"TOOL ERROR"),
    "template token/HTML": re.compile(r"<\|?/?\w*\|?>|<(?:p|div|center|figure)\b"),
}


# --------------------------------------------------------------------------
# Answer cleaning (blinding hygiene)
# --------------------------------------------------------------------------
_ABS_PATH = re.compile(r"(?:file://)?/(?:shared|home|tmp|mnt|scratch|var|data)/[^\s`'\")\]>]+")
_REL_PATH = re.compile(r"\b(?:tool_output|output)/[\w.\-]+(?:/[\w.\-]+)*\.\w{2,5}\b")
# Chat-template tokens (e.g. "<|channel>thought<channel|>") and layout HTML leak
# from some models and would identify them.
_SPECIAL_TOKEN = re.compile(r"<\|[^>\n]*>|<[^>\n|]*\|>")
_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?(?:p|center|div|figure|figcaption|sup|sub|span)\b[^>]*>|<!--.*?-->", re.IGNORECASE | re.DOTALL)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️‍]")


def _basename(match):
    """Reduce a path match to its file name, dropping trailing punctuation."""
    path = match.group(0).rstrip(".,;:")
    return path.rsplit("/", 1)[-1]


def to_plain_text(markdown):
    """Convert model Markdown to uniform plain text and remove local paths/emoji."""
    text = markdown.replace("\r\n", "\n")
    text = re.sub(r"<\|channel>\s*thought\s*<channel\|>", "", text)
    text = _SPECIAL_TOKEN.sub("", text)
    text = _HTML_BREAK.sub("\n", text)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "[image not shown]", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = _ABS_PATH.sub(_basename, text)
    text = _REL_PATH.sub(_basename, text)
    text = re.sub(r"^\s*```\w*\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", "", text, flags=re.M)  # table separators
    text = re.sub(r"^\s*\|(.*)\|\s*$", lambda m: m.group(1).strip(), text, flags=re.M)
    text = re.sub(r"^(\s*)[*-]\s+", lambda m: m.group(1) + "• ", text, flags=re.M)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\*(\S[^*\n]*\S|\S)\*", r"\1", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def n_words(text):
    return len(text.split())


def find_leaks(text):
    return [name for name, pattern in LEAK_PATTERNS.items() if pattern.search(text)]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_tool_to_agent():
    """Map tool name -> sub-agent, reusing the repo's MCP tool inventory."""
    spec = importlib.util.spec_from_file_location("analyze_coverage", COVERAGE_SCRIPT)
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    inventory = coverage.build_mcp_inventory(ROOT_DIR)

    tool_to_agent = {}
    for server, agent in SERVER_TO_AGENT.items():
        for tool in inventory[server]["tool_names"]:
            tool_to_agent.setdefault(tool, agent)
    for tool in coverage._extract_tools_from_file(VLM_SERVER):
        tool_to_agent.setdefault(tool, VLM_AGENT)
    for tool, agent in EXTRA_TOOL_TO_AGENT.items():
        tool_to_agent.setdefault(tool, agent)
    return tool_to_agent


def expert_group(agents):
    """First specialist group (in EXPERT_GROUPS order) whose agent is used, else GENERAL_GROUP."""
    for group, specialist_agents in EXPERT_GROUPS.items():
        if specialist_agents & set(agents):
            return group
    return GENERAL_GROUP


def load_answers():
    """Return {model: {task_ID: {"text": raw final_answer, "images": [data URIs]}}}
    for non-empty answers."""
    basename_index, collisions = build_image_basename_index(ROOT_DIR)
    if collisions:
        print(f"  (image index: {len(collisions)} basename(s) had >1 match under {ROOT_DIR} -- "
              f"skipped, ambiguous: {sorted(collisions)[:5]}{'...' if len(collisions) > 5 else ''})")
    render_cache = {}  # resolved local path -> data URI (or None); shared so nothing is read/rendered twice

    answers = {}
    for model, run_dir in MODEL_RUNS.items():
        files = list((RESULTS_DIR / run_dir).glob("results_*.json"))
        if len(files) != 1:
            sys.exit(f"Expected exactly one results file in {run_dir}, found {len(files)}")
        with open(files[0]) as f:
            tasks = json.load(f)["tasks"]
        model_answers = {}
        for t in tasks:
            sol = t.get("generated_solution")
            text = (sol.get("final_answer") or "").strip() if sol else ""
            if not text:
                continue
            model_answers[t["task_ID"]] = {
                "text": sol["final_answer"],
                "images": find_answer_images(sol.get("steps", []), basename_index, render_cache),
            }
        answers[model] = model_answers
    return answers


def build_pool(tool_to_agent):
    """Eligible questions with cleaned answers, agents and reading-load stats.

    A question is eligible as soon as at least MIN_ANSWERS_TO_INCLUDE model(s)
    have a usable answer for it. Experts only ever see the models that
    actually answered a given question -- a model with no answer, or an
    answer that fails the filters below, gets no slot at all rather than a
    placeholder, so the answer count varies per question (see `allowed`).
    """
    with open(TASKS_FILE) as f:
        tasks = json.load(f)["tasks"]
    raw_answers = load_answers()

    pool = []
    for task in tasks:
        label = task["task_labels"][0]
        if label not in LABELS or task["GT"] in GT_EXCLUDED:
            continue
        task_id = task["task_ID"]

        cleaned = {m: to_plain_text(raw_answers[m][task_id]["text"]) for m in MODELS if task_id in raw_answers[m]}
        allowed = [
            m for m in MODELS  # keep MODELS order, not dict/set order, for stable output
            if m in cleaned and MIN_ANSWER_WORDS <= n_words(cleaned[m]) and len(cleaned[m]) <= MAX_ANSWER_CHARS
            and not find_leaks(cleaned[m])
        ]
        if len(allowed) < MIN_ANSWERS_TO_INCLUDE:
            continue

        tools = [s["function_name"] for s in task["reference_solutions"][0]["steps"]]
        unmapped = sorted({t for t in tools if t not in tool_to_agent})
        if unmapped:
            sys.exit(f"{task_id}: tools without sub-agent mapping: {unmapped}")

        pool.append({
            "task_id": task_id,
            "label": label,
            "gt": task["GT"],
            "question": task["task_text"].strip(),
            "tools": sorted(set(tools)),
            "agents": sorted({tool_to_agent[t] for t in tools}),
            "group": expert_group({tool_to_agent[t] for t in tools}),
            "answers": cleaned,
            "allowed": allowed,
            "words": {m: n_words(cleaned[m]) for m in allowed},
            "images": {m: raw_answers[m][task_id]["images"] for m in allowed},
        })
    return pool


# --------------------------------------------------------------------------
# Coverage and batching -- every pool question goes into the study; this
# only decides how many times each is rated and which forms carry it.
# --------------------------------------------------------------------------
def build_slot_multiset(pool, total_slots, rng):
    """Repeat the pool (stratified by label) via shuffled full passes until
    there are `total_slots` entries. Coverage per question ends up nearly
    uniform within each label: every question in a label appears
    `label_slots // len(label_pool)` times, and `label_slots % len(label_pool)`
    of them (picked afresh from the label's shuffled order, so it varies
    across labels/builds rather than always landing on the same questions)
    get one extra appearance.
    """
    by_label = collections.defaultdict(list)
    for q in pool:
        by_label[q["label"]].append(q)

    # Split total_slots across labels proportional to each label's pool size,
    # then patch rounding so the per-label counts sum exactly to total_slots.
    label_slots = {label: round(total_slots * len(qs) / len(pool)) for label, qs in by_label.items()}
    diff = total_slots - sum(label_slots.values())
    labels_by_size = sorted(by_label, key=lambda l: -len(by_label[l]))
    i = 0
    while diff:
        label = labels_by_size[i % len(labels_by_size)]
        label_slots[label] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1

    slots = []
    for label, qs in by_label.items():
        need = label_slots[label]
        while need > 0:
            pass_order = rng.sample(qs, len(qs))
            take = min(len(pass_order), need)
            slots.extend(pass_order[:take])
            need -= take
    rng.shuffle(slots)
    return slots


def distribute_into_forms(pool, n_experts, items_per_form, rng):
    """Deal every pool question (each repeated per build_slot_multiset's
    coverage) into n_experts forms of items_per_form questions each, with no
    question appearing twice in the same form."""
    total_slots = n_experts * items_per_form
    if total_slots < len(pool):
        sys.exit(f"experts x --items-per-form ({n_experts} x {items_per_form} = {total_slots}) is less than "
                  f"the pool ({len(pool)}) -- every question needs at least one slot. Raise one of them.")

    slots = build_slot_multiset(pool, total_slots, rng)
    copies = collections.Counter(q["task_id"] for q in slots)
    question_by_id = {q["task_id"]: q for q in pool}
    forms = [[] for _ in range(n_experts)]
    # Put all k copies of a question into the k least-filled forms (random
    # tie-break). Fill levels then never differ by more than one, so every
    # form ends at exactly items_per_form and no form holds a question twice
    # -- unlike first-fit dealing, which can dead-end when coverage is tight.
    for task_id in dict.fromkeys(q["task_id"] for q in slots):  # shuffled order, deduplicated
        k = copies[task_id]
        if k > n_experts:
            sys.exit(f"{task_id} needs {k} forms but there are only {n_experts} -- lower --items-per-form.")
        targets = sorted(range(n_experts), key=lambda i: (len(forms[i]), rng.random()))[:k]
        for i in targets:
            forms[i].append(question_by_id[task_id])
    for form_items in forms:
        rng.shuffle(form_items)  # question order within the form
    return forms


# --------------------------------------------------------------------------
# Form content
# --------------------------------------------------------------------------
INTRO_TEXT = """Thank you for helping us evaluate the performance of different LLMs in Agentic AI frameworks for EO!

WHAT THIS IS
We built a benchmark to test the performance of different LLMs in agentic settings. This evaluation form is to compare the answers given by these LLMs when answering a question using different tools related to the agentic framework that we built - including different EO endpoints (Copernicus/Sentinel, Landsat), ERA5 climate data, EO Dashboard narratives, ESA science strategy documents, GIS analysis and scientific literature search.

You will be given {n_items} questions, and for each question you see every anonymous answer that an assistant actually produced for it -- usually several, occasionally just one if only one assistant answered that particular question.

WHAT YOU DO
(About {target_minutes:.0f} minutes for this form -- your progress is saved in this browser, so it's fine to do this over more than one sitting.) The form will guide you through it, but for every question you will
1. Rate how relevant and realistic the question is for EO users.
2. Score the quality of each answer shown on three metrics: coverage (does it contain all the content necessary?), focus (is the content present relevant?), and correctness (is the content present right?).
3. If there's more than one answer, rank them all from best to worst quality-wise. Your ranking can be independent of the scores given in 2. Skip this step when only one answer is shown -- there's nothing to compare it to.

HOW TO JUDGE COVERAGE, FOCUS and CORRECTNESS
All three of these will be scored from 1 to 5, with 1 being the worst and 5 the best.
1. Coverage: the extent to which the answer addresses the question and has the content needed to address the question. An answer lacking coverage addresses only part of the question or misses steps needed for the answer.
2. Focus: the extent to which the content in the reply is appropriate for the given question. An answer lacking focus contains information or reasoning that is not relevant for the answer.
3. Correctness: the extent to which the content in the reply is correct, such as having sound reasoning and factually accurate information. An answer lacking correctness contains falsehoods, misinterprets literature or data, or displays faulty reasoning.

For clarity, given the question, "How old is the president of the USA?":
A. "I am a chatbot" lacks both coverage and focus;
B. "He is the president of the USA, so he must be at least 35 according to the US constitution" lacks coverage;
C. "I have checked Wikipedia and major newspaper outlets, and he is 80. I have also found this video of him that I would like to show you" lacks focus;
D. A, B and C are still all correct.

A minority of answers (marked "▣ image" next to the answer letter) include the actual image the assistant looked at -- check the description against it directly. For every other answer you cannot open the files or images it says it produced: treat claims that you find implausible or unsupported as errors. Do not reward length or formatting. If you cannot judge a question, say so in the comment field and give your best estimate.

Answers were produced automatically and may contain errors. Local file paths were removed from the answers. Long answers are collapsed by default -- click an answer to expand it.

PRIVACY
Your name is recorded so we can follow up if needed, and is not shown to other experts. Responses are used only for the research evaluation of this benchmark and are reported in aggregate. Participation is voluntary; you can stop at any time by closing the page."""

PROFILE_FIELDS = [
    {"type": "text", "title": "Name", "help": "So we know who a response came from. Not shown to other experts.", "required": True},
    {"type": "scale", "title": "How would you rate your Earth Observation expertise?", "low": "Newcomer", "high": "Expert", "required": True},
    {"type": "checkbox", "title": "Which areas are you experienced in?",
     "choices": ["Optical / SAR satellite imagery", "Climate and atmospheric data (ERA5, trace gases)",
                 "Geospatial / GIS analysis", "Scientific literature",
                 "ESA Earth Observation Science Strategy (EOSS)", "EO Dashboard and its narratives",
                 "EO training material (MOOCs)"],
     "other": True, "required": False},
    {"type": "checkbox", "title": "Consent", "choices": ["I agree that my responses are used for the research evaluation of this benchmark."], "required": True},
]

RELEVANCE_TITLE = "How relevant and realistic is this question for EO users (researchers, analysts, practitioners)?"
RELEVANCE_HELP = "1 = not relevant or unrealistic; 3 = plausible but marginal or somewhat artificial; 5 = highly relevant and realistic."
COVERAGE_TITLE = "Coverage: does the answer have the content needed to address the question?"
COVERAGE_HELP = ("1 = missing all or fundamental parts of the content; 2 = major omissions; 3 = partially incomplete; "
                 "4 = minor omissions; 5 = the answer contains everything needed to address the question. Equal scores are allowed.")
FOCUS_TITLE = "Focus: is the content of the answer appropriate for the question?"
FOCUS_HELP = ("1 = the content of the answer is irrelevant to the question; 2 = mostly irrelevant; 3 = partially relevant; "
                 "4 = mostly appropriate; 5 = the content of the answer is completely appropriate for the question. Equal scores are allowed.")
CORRECTNESS_TITLE = "Correctness: is the content of the answer correct?"
CORRECTNESS_HELP = ("1 = the content of the answer is false and/or the reasoning is deeply flawed; 2 = mostly incorrect; 3 = partially incorrect; "
                 "4 = mostly correct; 5 = the content of the answer is completely true and the reasoning is sound. Equal scores are allowed.")
RANK_TITLE = "Ranking: order all answers from best to worst."
RANK_HELP = "Drag, or use the arrows, to put the best answer at the top and the worst at the bottom. Ties are not allowed."
FAMILIARITY_TITLE = "How familiar are you with the topic of this question?"
COMMENT_TITLE = "Comment (optional)"
COMMENT_HELP = "For example: an error you spotted, why a question is not relevant, or that you could not judge it."


def form_code(index):
    return f"F{index:02d}"


def words_to_read(items):
    return sum(n_words(i["question"]) + sum(n_words(a["text"]) for a in i["answers"]) for i in items)


def estimated_minutes(items):
    return INTRO_MINUTES + JUDGEMENT_MINUTES_PER_ITEM * len(items) + words_to_read(items) / READING_WPM


def build_form_spec(index, group, form_items, key_rows, rng):
    """Blinded form spec; appends the letter->model key rows for the organiser."""
    code = form_code(index)
    items = []
    for position, item in enumerate(form_items, start=1):
        models = list(item["allowed"])  # only models that actually answered this question
        rng.shuffle(models)  # display order differs per form
        letters = LETTERS[:len(models)]
        answers = []
        for letter, model in zip(letters, models):
            image_uris = item["images"].get(model, [])  # already-encoded data URIs, see load_answers
            answer = {"letter": letter, "text": item["answers"][model]}
            if image_uris:
                answer["images"] = image_uris
            answers.append(answer)
            key_rows.append({
                "form": code, "group": group, "item": position, "task_id": item["task_id"],
                "label": item["label"], "letter": letter, "model": model,
            })
        items.append({"position": position, "label": item["label"], "question": item["question"], "answers": answers})

    return {
        "code": code,
        "group": group,
        "title": f"Human Expert Evaluation of Agentic AI for EO, form {code}",
        "description": INTRO_TEXT.format(n_items=len(items), target_minutes=estimated_minutes(items)),
        "profile_fields": PROFILE_FIELDS,
        "item_fields": {
            "relevance": {"title": RELEVANCE_TITLE, "help": RELEVANCE_HELP, "low": "Not relevant", "high": "Highly relevant"},
            "coverage": {"title": COVERAGE_TITLE, "help": COVERAGE_HELP},
            "focus": {"title": FOCUS_TITLE, "help": FOCUS_HELP},
            "correctness": {"title": CORRECTNESS_TITLE, "help": CORRECTNESS_HELP},
            "ranking": {"title": RANK_TITLE, "help": RANK_HELP},
            "familiarity": {"title": FAMILIARITY_TITLE, "low": "Unfamiliar", "high": "Expert"},
            "comment": {"title": COMMENT_TITLE, "help": COMMENT_HELP},
        },
        "items": items,
    }


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------
def write_study_overview(specs, minutes, path):
    lines = [
        "# Expert evaluation: study overview",
        "",
        "Reference / audit copy of every form's content. The actual survey is the web app "
        "(`expert_eval/out/DISTRIBUTION.md` has the links to send out) — this file is for reading "
        "offline and for a paper appendix, not for building anything by hand.",
        "",
        "## Overview",
        "",
        "| Form | Group | Questions | Labels in display order | Words to read | Est. minutes |",
        "|---|---|---|---|---|---|",
    ]
    for spec, mins in zip(specs, minutes):
        labels = " / ".join(i["label"] for i in spec["items"])
        lines.append(f"| {spec['code']} | {spec['group']} | {len(spec['items'])} | {labels} | {words_to_read(spec['items'])} | {mins:.1f} |")

    lines += ["", "## Form contents", ""]
    for spec, mins in zip(specs, minutes):
        lines += [f"### Form {spec['code']}", "", f"Group: {spec['group']}. Estimated {mins:.1f} min", ""]
        for item in spec["items"]:
            lines += [f"#### {spec['code']} / Question {item['position']} of {len(spec['items'])}", "",
                      "**Question**", "", item["question"], ""]
            for answer in item["answers"]:
                heading = f"**Answer {answer['letter']}**" + (" (includes image)" if answer.get("images") else "")
                lines += [heading, "", "```", answer["text"], "```", ""]
        lines += ["---", ""]
    path.write_text("\n".join(lines))


def read_existing_assignments(path):
    """Best-effort: if `path` already has a filled-in 'Sent to' column, carry
    those values forward keyed by (form code, group), so rebuilding the study
    doesn't wipe out who the organiser already noted a link went to."""
    if not path.exists():
        return {}
    assignments = {}
    for line in path.read_text().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not re.fullmatch(r"F\d+", cells[0]):
            continue
        if cells[5]:
            assignments[(cells[0], cells[1])] = cells[5]
    return assignments


def write_distribution_manifest(specs, minutes, path, existing_assignments=None):
    existing_assignments = existing_assignments or {}
    lines = [
        "# Sending out the forms",
        "",
        "Live app: `{ARTIFACT_URL}` — replace that placeholder everywhere below with the real "
        "link once it's published, and **first open its Share menu in claude.ai and set it to "
        "\"Anyone with the link can view\"**, otherwise the links 404 for your experts.",
        "",
        "Send each expert their own row's link below (one per expert — each opens a different, "
        "unique set of questions). Match the expert to the \"Group\" column: "
        + ", ".join(f"`{g}` = {'/'.join(sorted(a))}" for g, a in EXPERT_GROUPS.items())
        + f" specialists; `{GENERAL_GROUP}` = everyone else (questions using none of those agents)."
        " **Also tell them their code in plain text** (the \"Form\" "
        "column) — claude.ai's own routing has been observed to strip the `?form=` part of the "
        "link before the page loads, in which case the page falls back to a plain \"enter your "
        "code\" box instead of failing outright. Every expert downloads their own results file at "
        "the end and emails it back to you (the page prompts for this — nothing is submitted "
        "automatically). Drop all the files you receive into `expert_eval/responses/` and run "
        "`python scripts/analyze_expert_eval.py` to collect and analyse them — see expert_eval/README.md.",
        "",
        "Fill in \"Sent to\" as you send each link out, to keep track of who has which form. "
        "Rebuilding the study preserves whatever you've filled in here, matched by form code -- "
        "but only if the form codes still mean the same thing (same --experts/--items-per-form/--seed); a rebuild "
        "with different settings reshuffles which questions each code holds, so treat old "
        "assignments as stale and re-send in that case.",
        "",
        "If you rebuild the study or publish the web app again, re-run this script and re-copy "
        "the links below — the JSON specs may have changed.",
        "",
        "| Form | Group | Questions | Estimated minutes | Link | Sent to |",
        "|---|---|---|---|---|---|",
    ]
    for spec, mins in zip(specs, minutes):
        sent_to = existing_assignments.get((spec["code"], spec["group"]), "")
        lines.append(f"| {spec['code']} | {spec['group']} | {len(spec['items'])} | {mins:.1f} | {{ARTIFACT_URL}}?form={spec['code']} | {sent_to} |")
    path.write_text("\n".join(lines) + "\n")


def write_selected_questions(pool, coverage, forms_by_task, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "group", "label", "gt", "n_answers", "coverage", "agents", "n_tools", "tools", "question_words", "forms"])
        for q in sorted(pool, key=lambda q: (GROUPS.index(q["group"]), LABELS.index(q["label"]), q["task_id"])):
            writer.writerow([q["task_id"], q["group"], q["label"], q["gt"], len(q["allowed"]), coverage[q["task_id"]],
                             ";".join(q["agents"]), len(q["tools"]), ";".join(q["tools"]), n_words(q["question"]),
                             ";".join(forms_by_task[q["task_id"]])])


def write_private_key(key_rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["form", "group", "item", "task_id", "label", "letter", "model"])
        writer.writeheader()
        writer.writerows(key_rows)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify(specs, pool, key_rows, items_per_group):
    problems = []

    coverage = collections.Counter()
    forms_by_task = collections.defaultdict(list)
    task_ids_per_form = collections.defaultdict(set)
    for r in key_rows:
        if r["task_id"] not in task_ids_per_form[r["form"]]:
            coverage[r["task_id"]] += 1
            forms_by_task[r["task_id"]].append(r["form"])
        task_ids_per_form[r["form"]].add(r["task_id"])

    for q in pool:
        if coverage[q["task_id"]] < 1:
            problems.append(f"{q['task_id']}: appears in 0 forms")

    group_by_task = {q["task_id"]: q["group"] for q in pool}
    for r in key_rows:
        if r["group"] != group_by_task[r["task_id"]]:
            problems.append(f"{r['form']}: {r['task_id']} is a {group_by_task[r['task_id']]} question on a {r['group']} form")

    # A task_id should occupy exactly one question position within any given
    # form -- distribute_into_forms is built to guarantee this, verify it did.
    positions_by_form_task = collections.defaultdict(set)
    for r in key_rows:
        positions_by_form_task[(r["form"], r["task_id"])].add(r["item"])
    for (form, task_id), positions in positions_by_form_task.items():
        if len(positions) != 1:
            problems.append(f"{form}: {task_id} appears at more than one position ({sorted(positions)})")

    for spec in specs:
        n_expected = items_per_group[spec["group"]]
        if len(spec["items"]) != n_expected:
            problems.append(f"{spec['code']}: has {len(spec['items'])} questions, expected {n_expected}")
        for item in spec["items"]:
            letters = {a["letter"] for a in item["answers"]}
            expected = set(LETTERS[:len(item["answers"])])
            if letters != expected or not item["answers"]:
                problems.append(f"{spec['code']}/{item['position']}: answer letters wrong: {sorted(letters)}")
        # Scan only actual text fields, not embedded image data: base64 is
        # long enough now (up to tens of MB across a form) that short leak
        # patterns like "EVE" or "GPT" turn up in it by pure chance.
        scannable = [spec["title"], spec["description"]]
        for item in spec["items"]:
            scannable.append(item["question"])
            scannable.extend(a["text"] for a in item["answers"])
        leaks = find_leaks("\n".join(scannable))
        if leaks:
            problems.append(f"{spec['code']}: leaks {leaks}")

    # A question's model set is fixed once (see build_pool's `allowed`) and
    # reused verbatim across every form it appears in -- only the letters and
    # display order are reshuffled per form. Verify that consistency.
    models_by_form_question = collections.defaultdict(set)
    for r in key_rows:
        models_by_form_question[(r["form"], r["task_id"])].add(r["model"])
    model_sets_per_question = collections.defaultdict(set)
    for (_, task_id), models in models_by_form_question.items():
        model_sets_per_question[task_id].add(frozenset(models))
    if any(len(sets) != 1 for sets in model_sets_per_question.values()):
        problems.append("a question shows a different set of models on different forms")

    return problems, coverage, forms_by_task


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experts", nargs="+", default=[], metavar="GROUP=N",
                        help="experts (= forms) per group, overriding the defaults "
                             + " ".join(f"{g}={n}" for g, n in DEFAULT_GROUP_EXPERTS.items()))
    parser.add_argument("--items-per-form", type=int, default=8,
                        help="minimum questions per form (default 8); a group's forms get more if that's "
                             "needed for every question in the group to get at least one form")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    group_experts = dict(DEFAULT_GROUP_EXPERTS)
    for value in args.experts:
        group, _, n = value.partition("=")
        if group not in GROUPS or not n.isdigit() or int(n) < 1:
            sys.exit(f"--experts: expected GROUP=N with GROUP in {GROUPS} and N >= 1, got {value!r}")
        group_experts[group] = int(n)

    rng = random.Random(args.seed)
    tool_to_agent = load_tool_to_agent()
    pool = build_pool(tool_to_agent)
    print(f"Pool: {len(pool)} questions with >={MIN_ANSWERS_TO_INCLUDE} usable answer(s) "
          + ", ".join(f"{label}={sum(q['label'] == label for q in pool)}" for label in LABELS))
    answer_count_dist = collections.Counter(len(q["allowed"]) for q in pool)
    print("  answers per pool question:", dict(sorted(answer_count_dist.items())))

    # Forms are numbered consecutively across groups, in GROUPS order.
    forms = []  # (group, form_items)
    items_per_group = {}
    for group in GROUPS:
        group_pool = [q for q in pool if q["group"] == group]
        n_experts = group_experts[group]
        if not group_pool:
            sys.exit(f"Group {group!r} has no questions -- check EXPERT_GROUPS.")
        items_per_form = max(args.items_per_form, -(-len(group_pool) // n_experts))  # ceil
        items_per_group[group] = items_per_form
        total_slots = n_experts * items_per_form
        base_coverage, extra = divmod(total_slots, len(group_pool))
        print(f"Group {group}: {n_experts} forms x {items_per_form} questions/form = {total_slots} slot(s) for "
              f"{len(group_pool)} questions -> every question seen >={base_coverage} time(s)"
              + (f", {extra} of them seen {base_coverage + 1} times" if extra else ""))
        forms += [(group, form_items) for form_items in distribute_into_forms(group_pool, n_experts, items_per_form, rng)]

    key_rows = []
    specs = [build_form_spec(i, group, form_items, key_rows, rng) for i, (group, form_items) in enumerate(forms, start=1)]
    minutes = [estimated_minutes(s["items"]) for s in specs]

    problems, coverage, forms_by_task = verify(specs, pool, key_rows, items_per_group)

    (OUT_DIR / "forms").mkdir(parents=True, exist_ok=True)
    for stale in (OUT_DIR / "forms").glob("form_*.json"):  # a smaller rebuild must not leave old forms live
        stale.unlink()
    for spec in specs:
        (OUT_DIR / "forms" / f"form_{spec['code'][1:]}.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=1))
    write_study_overview(specs, minutes, OUT_DIR / "STUDY_OVERVIEW.md")
    existing_assignments = read_existing_assignments(OUT_DIR / "DISTRIBUTION.md")
    write_distribution_manifest(specs, minutes, OUT_DIR / "DISTRIBUTION.md", existing_assignments)
    write_selected_questions(pool, coverage, forms_by_task, OUT_DIR / "selected_questions.csv")
    write_private_key(key_rows, EVAL_DIR / "PRIVATE_key.csv")

    agent_counts = collections.Counter(a for q in pool for a in q["agents"])
    print("\nWhole-pool stats (every question here is in the study)")
    print("  questions per group:", {g: sum(q["group"] == g for q in pool) for g in GROUPS})
    print("  questions per label:", dict(collections.Counter(q["label"] for q in pool)))
    print("  questions per agent:", dict(sorted(agent_counts.items())))
    print("  multi-agent questions:", sum(len(q["agents"]) > 1 for q in pool), "/", len(pool))
    print("  verified (GT=1):", sum(q["gt"] == GT_VERIFIED for q in pool), "/", len(pool))
    single_answer = sum(len(q["allowed"]) == 1 for q in pool)
    print(f"  answers per question: min/max = {min(len(q['allowed']) for q in pool)}/{max(len(q['allowed']) for q in pool)}"
          + (f" ({single_answer} question(s) have only 1 answer -- ranking is skipped for those)" if single_answer else ""))
    cov_values = list(coverage.values())
    print(f"  coverage (forms per question): min/max/mean = {min(cov_values)}/{max(cov_values)}/{sum(cov_values) / len(cov_values):.2f}")
    questions_with_images = sum(1 for q in pool if any(q["images"].values()))
    n_images = sum(len(imgs) for q in pool for imgs in q["images"].values())
    print(f"  questions with a viewable image attached: {questions_with_images}/{len(pool)} ({n_images} image(s) total)")
    print(f"  words to read per form: min/max = {min(words_to_read(s['items']) for s in specs)}/{max(words_to_read(s['items']) for s in specs)}")
    for group in GROUPS:
        group_minutes = [m for s, m in zip(specs, minutes) if s["group"] == group]
        print(f"  estimated minutes per {group} form: min/max/mean = {min(group_minutes):.1f}/{max(group_minutes):.1f}/"
              f"{sum(group_minutes) / len(group_minutes):.1f}")
    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print("  -", problem)
        sys.exit(1)
    print("\nAll checks passed. Outputs written to", EVAL_DIR)


if __name__ == "__main__":
    main()
