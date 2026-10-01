"""
numerical_metrics.py — Tolerance-based scoring for `numerical_answer` tasks.

Implements Hit@tol and NumScore from TerraBench ("Can Agents Reason Over
Heterogeneous Earth-System Data?", arXiv:2606.13148), verified against the
paper directly rather than only the pasted summary:

  lambda = max(tau_abs, tau_rel * |y|, floor)     — effective per-field tolerance
  AE     = |y_hat - y|                            — absolute error
  n      = AE / lambda                            — unit-normalized error

  Hit@tol_field  = 1 if AE <= lambda else 0                       — binary pass/fail
  NumScore_field = 1 if n <= 1 else 2 ** -(n - 1)                 — smooth partial credit
                   (errors of 2*lambda, 3*lambda, 4*lambda score 1/2, 1/4, 1/8)

These correspond to `task_evaluation_strategies.json`'s declared-but-not-yet-
implemented `"evaluation_methods"` strings for every `numerical_answer` task:
  numeric_tolerance_accuracy -> hit_at_tol()
  unit_normalized_error      -> num_score()      (the metric IS the normalized error n)

Per the paper: tau_abs/tau_rel/floor are field-specific — there is no
universal default. Callers must supply a ToleranceSpec per field (or a
single default applied uniformly, see `uniform_tolerances()`); this module
deliberately does not invent numbers the paper didn't give.

Multi-field aggregation follows the paper exactly: fields are aligned by
key when keys are unique and match on both sides, else by position.
A ground-truth field missing from the prediction scores 0 on both metrics
(the paper's "missing predictions receive zero credit") rather than being
skipped — a candidate can't improve its average by omitting hard fields.
Non-numeric fields use exact match, contributing the same 0/1 value to
both metrics (the paper specifies exact-match scoring for non-numeric
fields but doesn't separately define NumScore's behavior there — treating
them identically to Hit@tol for that field is the natural reading, since
"exact or not" has no partial-credit notion to begin with).

Aggregation is two-level, exactly as described: average over fields within
an item, then average over items (unweighted row means) — the item, not
the individual field or tool step, is the unit of evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Per-field tolerance spec and the core single-value metrics
# ---------------------------------------------------------------------------

@dataclass
class ToleranceSpec:
    """lambda = max(tau_abs, tau_rel * |y|, floor) for one numeric field."""
    tau_abs: float = 0.0
    tau_rel: float = 0.0
    floor: float = 0.0

    def __post_init__(self):
        if self.tau_abs < 0 or self.tau_rel < 0 or self.floor < 0:
            raise ValueError("tau_abs, tau_rel, and floor must all be >= 0.")
        if self.tau_abs == 0 and self.tau_rel == 0 and self.floor == 0:
            raise ValueError(
                "At least one of tau_abs/tau_rel/floor must be > 0, or lambda "
                "collapses to 0 and any nonzero error becomes an automatic miss."
            )


def effective_tolerance(y: float, spec: ToleranceSpec) -> float:
    """lambda = max(tau_abs, tau_rel * |y|, floor)"""
    return max(spec.tau_abs, spec.tau_rel * abs(y), spec.floor)


def hit_at_tol(y_hat: float, y: float, spec: ToleranceSpec) -> int:
    """numeric_tolerance_accuracy — 1 if |y_hat - y| <= lambda, else 0."""
    lam = effective_tolerance(y, spec)
    return int(abs(y_hat - y) <= lam)


def num_score(y_hat: float, y: float, spec: ToleranceSpec) -> float:
    """unit_normalized_error — 1 if n <= 1, else 2**-(n-1), where n = AE / lambda."""
    lam = effective_tolerance(y, spec)
    n = abs(y_hat - y) / lam
    return 1.0 if n <= 1 else 2.0 ** -(n - 1)


def uniform_tolerances(fields: List[str], tau_abs: float = 0.0,
                        tau_rel: float = 0.0, floor: float = 0.0) -> Dict[str, ToleranceSpec]:
    """Convenience: the same ToleranceSpec for every field in `fields`, for
    callers that don't need per-field tuning."""
    spec = ToleranceSpec(tau_abs=tau_abs, tau_rel=tau_rel, floor=floor)
    return {f: spec for f in fields}


# ---------------------------------------------------------------------------
# Fixed presets for the two field categories our curated ground truths
# actually contain — band statistics (min/max/mean/std, NDVI/EVI) and lat/lon
# coordinates. Grounded in this project's own data/incidents, not round
# numbers picked out of the air:
#
# BAND_STAT_TOLERANCE — tau_rel=0.05, floor=0.01, tau_abs=0.
#   Our curated band values span two genuinely different unit conventions:
#   CDSE Sentinel-2 fetches return true reflectance (observed range here:
#   -0.9954 to 2.303 -- the evalscript requests units:"REFLECTANCE", see
#   compute_index()'s docstring), while GEE's Landsat C2 L2 SR product keeps
#   *raw scaled integers* (observed here: SR_B5 mean ~17895, max 30803 --
#   USGS doesn't apply the ~2.75e-5 scale+offset server-side). A relative
#   tolerance handles both scales automatically (5% of ~0.1 reflectance is
#   a sensible ~0.005; 5% of ~17895 DN is a sensible ~895); the absolute
#   floor only matters near zero (CDSE reflectance shadows/water, NDVI/EVI
#   values near their zero-crossing), where it keeps tau_rel*|y| from
#   collapsing to an unreasonably tight tolerance.
#
# LATLON_TOLERANCE — tau_abs=0.02 degrees (~2.2 km at the equator), tau_rel=0.
#   Relative tolerance is meaningless for an absolute coordinate (5% of a
#   near-zero longitude is ~0; 5% of -155 degrees is ~7.75 degrees --
#   neither relates to "how wrong is this location"), so this is pure
#   tau_abs. The number itself is calibrated against every geocode_place
#   divergence actually caught this session: central Barcelona (0.039/0.067
#   deg off -- a power plant, wrong place), Pavia (>0.1 deg -- also wrong),
#   Mopti (>0.1 deg -- also wrong). 0.02 deg clears all of them as misses
#   while still tolerating ordinary sub-km geocoder noise for the *same*
#   named landmark (e.g. facility entrance vs. facility centroid).
#   Known simplification: lat and lon are scored as independent fields
#   (per TerraBench's field-by-field design), so this doesn't account for
#   longitude's physical-distance compression at high latitude -- a fixed
#   degree threshold is slightly more forgiving on longitude near the poles
#   than near the equator.
# ---------------------------------------------------------------------------

BAND_STAT_TOLERANCE = ToleranceSpec(tau_abs=0.0, tau_rel=0.05, floor=0.01)
LATLON_TOLERANCE = ToleranceSpec(tau_abs=0.02, tau_rel=0.0, floor=0.0)


def band_stat_tolerances(fields: List[str]) -> Dict[str, ToleranceSpec]:
    """BAND_STAT_TOLERANCE applied to each given field name (e.g. ['min',
    'max', 'mean', 'std'] or band names like ['B02', 'B03', 'NDVI'])."""
    return {f: BAND_STAT_TOLERANCE for f in fields}


def latlon_tolerances(lat_key: str = "lat", lon_key: str = "lon") -> Dict[str, ToleranceSpec]:
    """LATLON_TOLERANCE for a {lat_key, lon_key} pair — defaults to the field
    names this project's own ground truths actually use ("lat"/"lon")."""
    return {lat_key: LATLON_TOLERANCE, lon_key: LATLON_TOLERANCE}


_STAT_KEYS = {"min", "max", "mean", "std"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_band_stat_group(d: Any) -> bool:
    """A dict of >=2 of {min,max,mean,std}, all numeric — e.g. one band's
    or one index's (NDVI/EVI) statistics, regardless of what it's nested
    under (a band name, 'ndvi_stats', 'band_statistics', ...)."""
    if not isinstance(d, dict) or len(d) < 2:
        return False
    keys = set(d.keys())
    return keys <= _STAT_KEYS and all(_is_number(v) for v in d.values())


def _is_latlon_group(d: Any) -> bool:
    """A dict with numeric 'lat' and 'lon' keys, wherever it's nested."""
    return isinstance(d, dict) and "lat" in d and "lon" in d and _is_number(d["lat"]) and _is_number(d["lon"])


def extract_scorable_fields(gt: Dict[str, Any]) -> "tuple[Dict[str, float], Dict[str, ToleranceSpec]]":
    """
    Recursively walk a curated ground-truth blob (e.g. one entry from the
    lean data/tasks/numerical_ground_truths.json) and pull out only the field groups
    this module has a fixed tolerance preset for: band-stat groups
    (BAND_STAT_TOLERANCE) and lat/lon pairs (LATLON_TOLERANCE), wherever
    they're nested, under dotted-path keys (e.g. "band_statistics.B02.min",
    "point1_ESRIN.lat").

    Everything else — distance_km, area_km2, a projected x/y transform, a
    UTM zone/CRS code string, a bare percentage — is deliberately left out,
    not guessed at with an invented tolerance. Call this, check whether the
    result is non-empty, and the caller decides what "not fully covered"
    means for its purposes (numerical_judge.py reports it, doesn't hide it).

    Returns (values, tolerances), both keyed by the same dotted paths, ready
    to pass straight into score_item() as the `ground_truth` and
    `tolerances` arguments.
    """
    values: Dict[str, float] = {}
    tolerances: Dict[str, ToleranceSpec] = {}

    def walk(node: Any, prefix: str) -> None:
        if _is_band_stat_group(node):
            for stat, v in node.items():
                key = f"{prefix}.{stat}" if prefix else stat
                values[key] = v
                tolerances[key] = BAND_STAT_TOLERANCE
            return
        if _is_latlon_group(node):
            for axis in ("lat", "lon"):
                key = f"{prefix}.{axis}" if prefix else axis
                values[key] = node[axis]
                tolerances[key] = LATLON_TOLERANCE
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "task_ID":
                    continue
                walk(v, f"{prefix}.{k}" if prefix else k)
        # Plain scalars reaching here (not part of a recognized group) are
        # intentionally dropped — see docstring.

    walk(gt, "")
    return values, tolerances


# ---------------------------------------------------------------------------
# Multi-field, single-item scoring
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    field: str
    y_hat: Any
    y: Any
    hit: float          # 0 or 1 (int for numeric fields, exact-match bool-as-int otherwise)
    num_score: float     # 0..1
    missing: bool        # True if the field was absent from the prediction
    numeric: bool        # False for exact-match (non-numeric) fields


@dataclass
class ItemResult:
    fields: List[FieldResult]

    @property
    def hit_at_tol(self) -> float:
        """Field-averaged Hit@tol for this one item."""
        return sum(f.hit for f in self.fields) / len(self.fields) if self.fields else 0.0

    @property
    def num_score(self) -> float:
        """Field-averaged NumScore for this one item."""
        return sum(f.num_score for f in self.fields) / len(self.fields) if self.fields else 0.0


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def score_item(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    tolerances: Dict[str, ToleranceSpec],
    *,
    align_by_key: bool = True,
) -> ItemResult:
    """
    Score one item's predicted fields against its ground-truth fields.

    Args:
        prediction   : the candidate's extracted field -> value dict.
        ground_truth : the curated ground truth's field -> value dict
                       (e.g. one entry's content from the lean
                       data/tasks/numerical_ground_truths.json).
        tolerances   : field name -> ToleranceSpec. Every *numeric*
                       ground-truth field must have an entry here — this
                       module won't guess a tolerance for you (see module
                       docstring: the paper gives no universal default).
        align_by_key : True (default) aligns by matching dict keys, the
                       paper's primary alignment rule. Set False to align
                       by position instead (its documented fallback) —
                       e.g. both sides are ordered lists under the hood.

    Returns:
        ItemResult with one FieldResult per ground-truth field.
    """
    gt_items = list(ground_truth.items())
    pred_items = list(prediction.items())

    if align_by_key:
        pairs = [(k, v, prediction.get(k, _MISSING)) for k, v in gt_items]
    else:
        pairs = [
            (gt_items[i][0], gt_items[i][1], pred_items[i][1] if i < len(pred_items) else _MISSING)
            for i in range(len(gt_items))
        ]

    results = []
    for key, y, y_hat in pairs:
        missing = y_hat is _MISSING
        numeric = _is_number(y)

        if missing:
            results.append(FieldResult(key, None, y, 0.0, 0.0, True, numeric))
            continue

        if numeric:
            if not _is_number(y_hat):
                # Candidate gave a non-numeric value for a numeric field — a miss.
                results.append(FieldResult(key, y_hat, y, 0.0, 0.0, False, True))
                continue
            spec = tolerances.get(key)
            if spec is None:
                raise KeyError(
                    f"No ToleranceSpec given for numeric field '{key}' — see module "
                    f"docstring: tolerances are field-specific, this module won't "
                    f"invent one. Pass it via `tolerances`."
                )
            h = hit_at_tol(float(y_hat), float(y), spec)
            s = num_score(float(y_hat), float(y), spec)
            results.append(FieldResult(key, y_hat, y, h, s, False, True))
        else:
            exact = int(y_hat == y)
            results.append(FieldResult(key, y_hat, y, exact, float(exact), False, False))

    return ItemResult(fields=results)


class _Missing:
    def __repr__(self):
        return "<MISSING>"


_MISSING = _Missing()


# ---------------------------------------------------------------------------
# Cross-item aggregation — "averaged over fields, then items"
# ---------------------------------------------------------------------------

def aggregate_items(items: List[ItemResult]) -> Dict[str, float]:
    """Unweighted mean of each item's (already field-averaged) Hit@tol/NumScore.
    The item, not the field or tool step, is the unit of evaluation."""
    if not items:
        return {"hit_at_tol": 0.0, "num_score": 0.0, "n_items": 0}
    return {
        "hit_at_tol": sum(it.hit_at_tol for it in items) / len(items),
        "num_score": sum(it.num_score for it in items) / len(items),
        "n_items": len(items),
    }
