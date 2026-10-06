"""Generic scalar tolerance scoring, including distance, area and statistics.

For each scalar, effective_tol = max(abs_tol, rel_tol * abs(y), floor_scale).
Hit@tol is one within tolerance; NumScore is one within tolerance and
2 ** (-(normalized_error - 1)) outside it. Scores average over all required
GT fields, including missing/invalid predictions as zeros. Extra predictions
are ignored; field-key alignment is the default.

Legacy band/coordinate tolerance presets and constructor aliases are retained.
Other numeric fields require explicitly configured tolerances. GT creation,
source suitability, units and artifact verification are separate from scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import math
from numbers import Real


# ---------------------------------------------------------------------------
# Per-field tolerance spec and the core single-value metrics
# ---------------------------------------------------------------------------

@dataclass(init=False)
class ToleranceSpec:
    """Finite, nonnegative scalar tolerances in the same units as the field.

    Legacy tau_abs/tau_rel/floor names remain accepted as constructor aliases.
    A strictly positive effective tolerance is required for each ground truth.
    """
    abs_tol: float
    rel_tol: float
    floor_scale: float

    def __init__(self, abs_tol: float = 0.0, rel_tol: float = 0.0,
                 floor_scale: float = 0.0, *, tau_abs: float | None = None,
                 tau_rel: float | None = None, floor: float | None = None) -> None:
        pairs = (("abs_tol", abs_tol, tau_abs), ("rel_tol", rel_tol, tau_rel),
                 ("floor_scale", floor_scale, floor))
        for name, value, alias in pairs:
            if alias is not None:
                if value != 0.0 and value != alias:
                    raise ValueError(f"Conflicting canonical and legacy tolerance: {name}")
                value = alias
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            setattr(self, name, float(value))
        if not any((self.abs_tol, self.rel_tol, self.floor_scale)):
            raise ValueError("At least one tolerance must be positive.")

    @property
    def tau_abs(self) -> float:
        return self.abs_tol

    @property
    def tau_rel(self) -> float:
        return self.rel_tol

    @property
    def floor(self) -> float:
        return self.floor_scale


def effective_tolerance(y: float, spec: ToleranceSpec) -> float:
    """Compute a finite, positive effective tolerance; never invent a floor."""
    if not _is_number(y) or not math.isfinite(y):
        raise ValueError("Ground truth must be a finite numeric scalar.")
    tolerance = max(spec.abs_tol, spec.rel_tol * abs(y), spec.floor_scale)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Effective tolerance must be finite and positive; supply abs_tol or floor_scale near zero.")
    return tolerance


def hit_at_tol(y_hat: Any, y: float, spec: ToleranceSpec) -> int:
    """Scalar Hit@tol; missing, invalid and unparseable predictions get zero."""
    return int(score_scalar(y_hat, y, spec).hit)


def num_score(y_hat: Any, y: float, spec: ToleranceSpec) -> float:
    """Scalar NumScore using the same tolerance rule for every quantity."""
    return score_scalar(y_hat, y, spec).num_score


def uniform_tolerances(fields: List[str], tau_abs: float = 0.0,
                       tau_rel: float = 0.0, floor: float = 0.0, *,
                       abs_tol: float | None = None, rel_tol: float | None = None,
                       floor_scale: float | None = None) -> Dict[str, ToleranceSpec]:
    """Apply one explicitly supplied tolerance configuration to all fields."""
    spec = ToleranceSpec(abs_tol=abs_tol or 0.0, rel_tol=rel_tol or 0.0,
                         floor_scale=floor_scale or 0.0,
                         tau_abs=tau_abs if abs_tol is None or tau_abs != 0.0 else None,
                         tau_rel=tau_rel if rel_tol is None or tau_rel != 0.0 else None,
                         floor=floor if floor_scale is None or floor != 0.0 else None)
    return {field: spec for field in fields}


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
    return isinstance(value, Real) and not isinstance(value, bool)


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


def extract_scorable_fields(
    gt: Dict[str, Any],
    field_tolerances: Dict[str, ToleranceSpec] | None = None,
) -> "tuple[Dict[str, float], Dict[str, ToleranceSpec]]":
    """Select explicitly configured scalar fields from nested GT by dotted path.

    With field_tolerances, only those fields are scored, including distances,
    areas, projected coordinates, percentages or counts. No metric depends on
    task or quantity type. Unknown paths/invalid GT are configuration errors.
    Without it, preserve legacy band-stat/lat-lon extraction and presets.
    """
    if field_tolerances is not None:
        selected = {}
        for path, spec in field_tolerances.items():
            value = gt
            remaining = path
            while remaining:
                if not isinstance(value, dict):
                    raise KeyError(f"Configured numerical GT field not found: {path}")
                parts = remaining.split(".")
                # Prefer literal keys: percentages such as gt_0.5 contain dots.
                match = next((index for index in range(len(parts), 0, -1)
                              if ".".join(parts[:index]) in value), None)
                if match is None:
                    raise KeyError(f"Configured numerical GT field not found: {path}")
                value = value[".".join(parts[:match])]
                remaining = ".".join(parts[match:])
            effective_tolerance(value, spec)
            selected[path] = float(value)
        return selected, dict(field_tolerances)

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
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "task_ID":
                    continue
                walk(v, f"{prefix}.{k}" if prefix else k)
        # Plain scalars reaching here (not part of a recognized group) are
        # intentionally dropped — see docstring.

    walk(gt, "")
    for field, value in values.items():
        effective_tolerance(value, tolerances[field])
    return values, tolerances


# ---------------------------------------------------------------------------
# Multi-field, single-item scoring
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    """Per-field credit and unrounded error diagnostics."""
    field: str
    y_hat: Any
    y: Any
    hit: float
    num_score: float
    missing: bool
    numeric: bool
    effective_tol: float | None = None
    absolute_error: float | None = None
    normalized_error: float | None = None
    invalid: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe diagnostics; invalid predictions have no calculable error."""
        from dataclasses import asdict
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                data[key] = float(value) if math.isfinite(value) else str(value)
        return data


@dataclass
class ItemResult:
    """Mean credit across all required fields; missing fields stay in denominator."""
    fields: List[FieldResult]

    @property
    def hit_at_tol(self) -> float:
        return sum(field.hit for field in self.fields) / len(self.fields) if self.fields else 0.0

    @property
    def num_score(self) -> float:
        return sum(field.num_score for field in self.fields) / len(self.fields) if self.fields else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"hit_at_tol": self.hit_at_tol, "num_score": self.num_score,
                "fields": [field.to_dict() for field in self.fields]}


class _Missing:
    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _Missing()


def _parse_prediction(value: Any) -> float | None:
    """Accept finite numeric scalars or numeric strings, without guessing units."""
    if isinstance(value, bool) or not isinstance(value, (Real, str)):
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def score_scalar(y_pred: Any, y: float, spec: ToleranceSpec,
                 *, field: str = "value") -> FieldResult:
    """Score an ordinary scalar, including geodetic distance, with one formula."""
    tolerance = effective_tolerance(y, spec)
    missing = y_pred is _MISSING or y_pred is None
    predicted = None if missing else _parse_prediction(y_pred)
    if predicted is None:
        return FieldResult(field, None if y_pred is _MISSING else y_pred, y,
                           0.0, 0.0, missing, True, effective_tol=tolerance,
                           invalid=not missing)
    absolute_error = abs(predicted - y)
    normalized_error = absolute_error / tolerance
    hit = int(normalized_error <= 1)
    credit = 1.0 if hit else 2.0 ** (-(normalized_error - 1))
    return FieldResult(field, y_pred, y, hit, credit, False, True,
                       tolerance, absolute_error, normalized_error)


def score_item(prediction: Dict[str, Any], ground_truth: Dict[str, Any],
               tolerances: Dict[str, ToleranceSpec], *, align_by_key: bool = True) -> ItemResult:
    """Align required fields and average scalar scores, counting missing as zero.

    By default align on GT keys and ignore extra predictions. Explicit positional
    alignment is retained for older callers. Non-numeric exact matching is a
    legacy convenience, separate from the scalar numerical metric.
    """
    predicted_values = list(prediction.values())
    results = []
    for index, (key, y) in enumerate(ground_truth.items()):
        y_hat = prediction.get(key, _MISSING) if align_by_key else (
            predicted_values[index] if index < len(predicted_values) else _MISSING)
        if _is_number(y):
            if key not in tolerances:
                raise KeyError(f"No ToleranceSpec given for numeric field '{key}'.")
            results.append(score_scalar(y_hat, y, tolerances[key], field=key))
        else:
            missing = y_hat is _MISSING or y_hat is None
            exact = 0 if missing else int(y_hat == y)
            results.append(FieldResult(key, None if y_hat is _MISSING else y_hat,
                                       y, exact, float(exact), missing, False))
    return ItemResult(results)


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
