"""
analyses.py — Fixed, deterministic analyses for the lab's execute mode.

The experiment the lab designs is tested with code, not with an LLM: each analysis is a
well-defined computation on Copernicus data served by Sentinel Hub on the Copernicus Data
Space (CDSE_CLIENT_ID / CDSE_CLIENT_SECRET; it uses the account's free processing quota,
not openEO credits). Dates are chosen from the data (clearest scenes), never by a model.

  optical_conditions     per Sentinel-2 acquisition in the padded window: cloud / cloud-shadow
                         fraction (scene classification) and mean aerosol optical thickness
                         (smoke proxy), split before / during / after the event.
  s2_dnbr_burned_area    dNBR between the clearest pre- and post-event Sentinel-2 scenes,
                         clouds and water masked: burned area and usable fraction.
  s1_backscatter_change  Sentinel-1 VH backscatter change (dB) between pre- and post-event
                         composites, terrain-corrected: burned area and usable fraction.
  s1_s2_agreement        overlap of the two burned-area maps (IoU, area ratio).

All maps share one grid (TARGET_RES_M, capped at the Process API's 2500 px), so they can be
compared pixel by pixel. Network access goes through an injectable client with async
`statistics(body)` and `process(body) -> ndarray (bands, rows, cols)`, which keeps tests offline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Awaitable, Callable

import httpx
import numpy as np

logger = logging.getLogger(__name__)

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_URL = "https://sh.dataspace.copernicus.eu"
HTTP_TIMEOUT_S = 180.0
CRS84 = "http://www.opengis.net/def/crs/EPSG/0/4326"

TARGET_RES_M = 20.0
MAX_PX = 2500
STATS_RES_DEG = 0.0006          # ~60 m: enough for scene-level cloud fractions
CLEAR_CLOUD_MAX = 0.10          # a scene with at most this cloud/shadow fraction counts as clear
MIN_COVERAGE = 0.9              # ... and at least this share of the bbox imaged
DNBR_BURNED = 0.27              # USGS/Key & Benson: moderate-low severity and above
S1_VH_DROP_DB = -1.5            # VH decrease that marks burned vegetation (C-band, assumption)
S1_NOISE_FLOOR = 0.001          # linear gamma0 (-30 dB): darker samples are frame/swath-edge noise
SMOOTH_PX = 5                   # box filter on the S1 change map (speckle)
# Sentinel-2 L2A scene classes: 0 no data, 1 saturated, 3 cloud shadow, 6 water,
# 8/9 cloud medium/high probability, 10 thin cirrus, 11 snow.
SCL_CLOUD = (3, 8, 9, 10)
SCL_UNUSABLE = (0, 1, 3, 6, 8, 9, 10, 11)

ANALYSES: dict[str, str] = {
    "optical_conditions": "Cloud/cloud-shadow fraction and aerosol (smoke) per Sentinel-2 scene, "
                          "before / during / after the event; picks the clearest pre/post scenes.",
    "s2_dnbr_burned_area": "Sentinel-2 dNBR burned area between the clearest pre- and post-event "
                           f"scenes (dNBR >= {DNBR_BURNED}), clouds and water masked.",
    "s1_backscatter_change": "Sentinel-1 VH backscatter change between pre- and post-event "
                             f"composites (drop <= {S1_VH_DROP_DB} dB = burned); unaffected by cloud/smoke.",
    "s1_s2_agreement": "Overlap of the Sentinel-1 and Sentinel-2 burned-area maps (IoU, area ratio).",
}
DEPENDS = {"s2_dnbr_burned_area": ("optical_conditions",),
           "s1_backscatter_change": (),
           "s1_s2_agreement": ("s2_dnbr_burned_area", "s1_backscatter_change")}


# ---------------------------------------------------------------------------
# Sentinel Hub client
# ---------------------------------------------------------------------------

class SceneClient(Protocol):
    """What the analyses need from a data client (SentinelHubClient, or a test fake)."""

    async def statistics(self, body: dict) -> dict: ...

    async def process(self, body: dict) -> np.ndarray: ...


class SentinelHubClient:
    """Minimal async client for the CDSE Sentinel Hub Statistical and Process APIs."""

    def __init__(self, client_id: str, client_secret: str, base_url: str = SH_URL) -> None:
        self._id, self._secret, self._base = client_id, client_secret, base_url
        self._token: str | None = None
        self._expires = 0.0

    @classmethod
    def from_env(cls) -> SentinelHubClient | None:
        cid, secret = os.getenv("CDSE_CLIENT_ID"), os.getenv("CDSE_CLIENT_SECRET")
        return cls(cid, secret) if cid and secret else None

    async def _headers(self) -> dict[str, str]:
        if self._token is None or time.monotonic() > self._expires:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(TOKEN_URL, data={"grant_type": "client_credentials",
                                                          "client_id": self._id,
                                                          "client_secret": self._secret})
                resp.raise_for_status()
                data = resp.json()
            self._token = data["access_token"]
            self._expires = time.monotonic() + float(data.get("expires_in", 600)) - 60
        return {"Authorization": f"Bearer {self._token}"}

    async def statistics(self, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            resp = await client.post(f"{self._base}/api/v1/statistics", json=body,
                                     headers=await self._headers())
            resp.raise_for_status()
            return resp.json()

    async def process(self, body: dict) -> np.ndarray:
        import rasterio   # heavy; only needed here
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            resp = await client.post(f"{self._base}/process/v1", json=body,
                                     headers={**await self._headers(), "Accept": "image/tiff"})
            resp.raise_for_status()
            content = resp.content
        with rasterio.MemoryFile(content) as mem, mem.open() as ds:
            return ds.read().astype("float32")


# ---------------------------------------------------------------------------
# Grid and request helpers
# ---------------------------------------------------------------------------

@dataclass
class AnalysisContext:
    bbox: list[float]
    start: str                  # event period, YYYY-MM-DD
    end: str
    pad_days: int = 14
    out_dir: Path | None = None
    memo: dict[str, Any] = field(default_factory=dict)   # arrays shared between analyses

    @property
    def window(self) -> tuple[str, str]:
        return (_shift(self.start, -self.pad_days), _shift(self.end, self.pad_days))

    def grid(self) -> tuple[int, int, float]:
        """(width, height, pixel area km2) of the shared analysis grid."""
        west, south, east, north = self.bbox
        lat = math.radians((south + north) / 2)
        w_m = (east - west) * 111_320.0 * math.cos(lat)
        h_m = (north - south) * 110_540.0
        res = max(TARGET_RES_M, w_m / MAX_PX, h_m / MAX_PX)
        width, height = max(1, round(w_m / res)), max(1, round(h_m / res))
        return width, height, (w_m / width) * (h_m / height) / 1e6


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day[:10]) + timedelta(days=days)).isoformat()


def _bounds(bbox: list[float]) -> dict:
    return {"bbox": bbox, "properties": {"crs": CRS84}}


def _process_body(ctx: AnalysisContext, data: dict, evalscript: str) -> dict:
    width, height, _ = ctx.grid()
    return {"input": {"bounds": _bounds(ctx.bbox), "data": [data]},
            "output": {"width": width, "height": height,
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": evalscript}


def _time_range(start: str, end: str) -> dict:
    return {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"}


def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """k x k moving average with edge padding (NaN-free input)."""
    if k <= 1:
        return a
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k))[: a.shape[0], : a.shape[1]]


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------------------
# Analyses: each returns {"status", "summary", "metrics", "details", "artifacts"}
# ---------------------------------------------------------------------------

_ES_CONDITIONS = """//VERSION=3
function setup(){return{input:[{bands:["SCL","AOT","dataMask"]}],
 output:[{id:"cloud",bands:1,sampleType:"FLOAT32"},{id:"aot",bands:1,sampleType:"FLOAT32"},
         {id:"dataMask",bands:1}]};}
function evaluatePixel(s){var c=[3,8,9,10].indexOf(s.SCL)>=0?1:0;
 return{cloud:[c],aot:[s.AOT],dataMask:[s.dataMask]};}"""


def _phase(day: str, ctx: AnalysisContext) -> str:
    return "before" if day < ctx.start else "after" if day > ctx.end else "during"


async def optical_conditions(ctx: AnalysisContext, client) -> dict:
    lo, hi = ctx.window
    body = {"input": {"bounds": _bounds(ctx.bbox),
                      "data": [{"type": "sentinel-2-l2a", "dataFilter": {"mosaickingOrder": "leastCC"}}]},
            "aggregation": {"timeRange": _time_range(lo, hi), "aggregationInterval": {"of": "P1D"},
                            "evalscript": _ES_CONDITIONS, "resx": STATS_RES_DEG, "resy": STATS_RES_DEG}}
    data = await client.statistics(body)
    days = []
    for interval in data.get("data", []):
        out = interval.get("outputs") or {}
        try:
            cloud = out["cloud"]["bands"]["B0"]["stats"]
            aot = out["aot"]["bands"]["B0"]["stats"]
        except KeyError:
            continue
        n = cloud.get("sampleCount") or 0
        coverage = 1 - (cloud.get("noDataCount") or 0) / n if n else 0.0
        if coverage < 0.5 or cloud.get("mean") is None:
            continue   # swath edge: the scene barely touches the bbox
        day = interval["interval"]["from"][:10]
        days.append({"date": day, "phase": _phase(day, ctx), "cloud": round(float(cloud["mean"]), 3),
                     "aot": round(float(aot.get("mean") or 0), 3), "coverage": round(coverage, 3)})
    if not days:
        return {"status": "failed", "summary": "No Sentinel-2 acquisitions over the bbox in the window.",
                "metrics": {}, "details": {"window": [lo, hi]}, "artifacts": []}

    def clear(d):
        return d["cloud"] <= CLEAR_CLOUD_MAX and d["coverage"] >= MIN_COVERAGE

    by_phase = {p: [d for d in days if d["phase"] == p] for p in ("before", "during", "after")}
    # Closest clear scene to the event on each side (least cloudy if none is clear).
    pre = (sorted([d for d in by_phase["before"] if clear(d)], key=lambda d: d["date"])[-1:]
           or sorted(by_phase["before"], key=lambda d: d["cloud"])[:1])
    post = (sorted([d for d in by_phase["after"] if clear(d)], key=lambda d: d["date"])[:1]
            or sorted(by_phase["after"], key=lambda d: d["cloud"])[:1])
    ctx.memo["s2_pre"], ctx.memo["s2_post"] = (pre[0] if pre else None), (post[0] if post else None)
    during = by_phase["during"]
    metrics = {"n_acquisitions": float(len(days)),
               **{f"n_clear_{p}": float(sum(clear(d) for d in v)) for p, v in by_phase.items()},
               **{f"n_{p}": float(len(v)) for p, v in by_phase.items()}}
    if during:
        metrics.update(mean_cloud_during=round(float(np.mean([d["cloud"] for d in during])), 3),
                       max_cloud_during=max(d["cloud"] for d in during),
                       mean_aot_during=round(float(np.mean([d["aot"] for d in during])), 3))
    if by_phase["before"]:
        metrics["mean_aot_before"] = round(float(np.mean([d["aot"] for d in by_phase["before"]])), 3)
    summary = (f"{len(days)} Sentinel-2 acquisitions {lo}..{hi}: clear (cloud <= {_pct(CLEAR_CLOUD_MAX)}) "
               f"{int(metrics['n_clear_before'])}/{len(by_phase['before'])} before, "
               f"{int(metrics['n_clear_during'])}/{len(during)} during, "
               f"{int(metrics['n_clear_after'])}/{len(by_phase['after'])} after the event. "
               + (f"Mean cloud/smoke-flagged fraction during the event {_pct(metrics['mean_cloud_during'])} "
                  f"(max {_pct(metrics['max_cloud_during'])}). " if during else "")
               + (f"Clearest pre-event scene {pre[0]['date']} ({_pct(pre[0]['cloud'])} cloud), "
                  if pre else "No pre-event scene, ")
               + (f"post-event {post[0]['date']} ({_pct(post[0]['cloud'])} cloud)." if post
                  else "no post-event scene.")
               + " Thick smoke is often classified as cloud by the Sentinel-2 scene classification.")
    return {"status": "ok", "summary": summary, "metrics": metrics,
            "details": {"window": [lo, hi], "days": days,
                        "pre_scene": pre[0] if pre else None, "post_scene": post[0] if post else None},
            "artifacts": []}


_ES_S2 = """//VERSION=3
function setup(){return{input:[{bands:["B8A","B12","SCL","dataMask"]}],output:{bands:4,sampleType:"FLOAT32"}};}
function evaluatePixel(s){return[s.B8A,s.B12,s.SCL,s.dataMask];}"""


async def _s2_scene(ctx: AnalysisContext, client, day: str) -> np.ndarray:
    data = {"type": "sentinel-2-l2a",
            "dataFilter": {"timeRange": _time_range(day, day), "mosaickingOrder": "leastCC"}}
    return await client.process(_process_body(ctx, data, _ES_S2))


async def s2_dnbr_burned_area(ctx: AnalysisContext, client) -> dict:
    pre, post = ctx.memo.get("s2_pre"), ctx.memo.get("s2_post")
    if not pre or not post:
        return {"status": "failed", "metrics": {}, "details": {}, "artifacts": [],
                "summary": "No pre- or post-event Sentinel-2 scene in the window: dNBR not computed."}
    a, b = await asyncio.gather(_s2_scene(ctx, client, pre["date"]), _s2_scene(ctx, client, post["date"]))

    def nbr(x):
        return (x[0] - x[1]) / np.maximum(x[0] + x[1], 1e-6)

    def usable(x):
        return (x[3] > 0) & ~np.isin(np.rint(x[2]), SCL_UNUSABLE)

    valid = usable(a) & usable(b)
    land = (b[3] > 0) & (np.rint(b[2]) != 6) & (np.rint(a[2]) != 6)
    dnbr = np.where(valid, nbr(a) - nbr(b), np.nan)
    burned = valid & (dnbr >= DNBR_BURNED)
    _, _, px_km2 = ctx.grid()
    n_land = int(land.sum()) or 1
    ctx.memo.update(s2_dnbr=dnbr, s2_burned=burned, s2_valid=valid, land=land)
    metrics = {"burned_km2": round(float(burned.sum() * px_km2), 1),
               "usable_fraction_of_land": round(float((valid & land).sum() / n_land), 3),
               "burned_fraction_of_usable": round(float(burned.sum() / max(int(valid.sum()), 1)), 3),
               "pre_cloud": pre["cloud"], "post_cloud": post["cloud"]}
    if burned.any():
        metrics["dnbr_median_burned"] = round(float(np.nanmedian(dnbr[burned])), 3)
    summary = (f"dNBR {pre['date']} -> {post['date']}: {metrics['burned_km2']} km2 burned "
               f"(dNBR >= {DNBR_BURNED}); {_pct(metrics['usable_fraction_of_land'])} of land pixels usable "
               f"in both scenes (clouds, shadow, water masked).")
    return {"status": "ok", "summary": summary, "metrics": metrics,
            "details": {"pre_date": pre["date"], "post_date": post["date"], "threshold": DNBR_BURNED},
            "artifacts": []}


_ES_S1 = """//VERSION=3
function setup(){return{input:[{bands:["VH","VV","dataMask"]}],output:{bands:3,sampleType:"FLOAT32"},
 mosaicking:"ORBIT"};}
function evaluatePixel(samples){var vh=0,vv=0,n=0;
 for(var i=0;i<samples.length;i++){var s=samples[i];
  if(s.dataMask>0&&s.VH>NOISE&&s.VV>NOISE){vh+=s.VH;vv+=s.VV;n++;}}
 return n>0?[vh/n,vv/n,n]:[0,0,0];}""".replace("NOISE", str(S1_NOISE_FLOOR))


async def _s1_composite(ctx: AnalysisContext, client, start: str, end: str, direction: str) -> np.ndarray:
    data = {"type": "sentinel-1-grd",
            "dataFilter": {"timeRange": _time_range(start, end), "acquisitionMode": "IW",
                           "polarization": "DV", "resolution": "HIGH", "orbitDirection": direction},
            "processing": {"backCoeff": "GAMMA0_TERRAIN", "orthorectify": True,
                           "demInstance": "COPERNICUS_30"}}
    return await client.process(_process_body(ctx, data, _ES_S1))


async def s1_backscatter_change(ctx: AnalysisContext, client) -> dict:
    pre_win = (_shift(ctx.start, -ctx.pad_days), _shift(ctx.start, -1))
    post_win = (_shift(ctx.end, 1), _shift(ctx.end, ctx.pad_days))
    chosen = None
    for direction in ("DESCENDING", "ASCENDING"):   # same geometry before and after
        pre, post = await asyncio.gather(_s1_composite(ctx, client, *pre_win, direction),
                                         _s1_composite(ctx, client, *post_win, direction))
        if (pre[2] > 0).mean() > 0.5 and (post[2] > 0).mean() > 0.5:
            chosen = (direction, pre, post)
            break
    if chosen is None:
        return {"status": "failed", "metrics": {}, "details": {"pre_window": pre_win, "post_window": post_win},
                "artifacts": [], "summary": "No Sentinel-1 IW acquisitions covering the bbox both before "
                                            "and after the event in one orbit direction."}
    direction, pre, post = chosen
    pre, post = np.nan_to_num(pre), np.nan_to_num(post)
    valid = (pre[2] > 0) & (post[2] > 0)
    land = ctx.memo.get("land")
    if land is not None and land.shape == valid.shape:
        valid &= land
    with np.errstate(divide="ignore", invalid="ignore"):
        pre_db = _box_mean(np.where(valid, 10 * np.log10(np.maximum(pre[0], 1e-6)), 0), SMOOTH_PX)
        post_db = _box_mean(np.where(valid, 10 * np.log10(np.maximum(post[0], 1e-6)), 0), SMOOTH_PX)
    change = np.where(valid, post_db - pre_db, np.nan)
    burned = valid & (change <= S1_VH_DROP_DB)
    _, _, px_km2 = ctx.grid()
    ctx.memo.update(s1_change=change, s1_burned=burned, s1_valid=valid)
    n_ref = int(land.sum()) if land is not None and land.shape == valid.shape else valid.size
    metrics = {"burned_km2": round(float(burned.sum() * px_km2), 1),
               "usable_fraction_of_land": round(float(valid.sum() / max(n_ref, 1)), 3),
               "burned_fraction_of_usable": round(float(burned.sum() / max(int(valid.sum()), 1)), 3),
               "n_pre_acquisitions": float(np.nanmax(pre[2])), "n_post_acquisitions": float(np.nanmax(post[2]))}
    if burned.any():
        metrics["vh_change_median_burned_db"] = round(float(np.nanmedian(change[burned])), 2)
    summary = (f"Sentinel-1 VH change ({direction.lower()} orbits, {pre_win[0]}..{pre_win[1]} vs "
               f"{post_win[0]}..{post_win[1]}; {int(metrics['n_pre_acquisitions'])} and "
               f"{int(metrics['n_post_acquisitions'])} acquisitions): {metrics['burned_km2']} km2 burned "
               f"(drop <= {S1_VH_DROP_DB} dB); {_pct(metrics['usable_fraction_of_land'])} of land pixels "
               "usable (radar is unaffected by cloud and smoke).")
    return {"status": "ok", "summary": summary, "metrics": metrics,
            "details": {"orbit_direction": direction, "pre_window": pre_win, "post_window": post_win,
                        "threshold_db": S1_VH_DROP_DB},
            "artifacts": []}


async def s1_s2_agreement(ctx: AnalysisContext, client) -> dict:
    m = ctx.memo
    if m.get("s2_burned") is None or m.get("s1_burned") is None:
        return {"status": "failed", "metrics": {}, "details": {}, "artifacts": [],
                "summary": "Agreement needs both burned-area maps; at least one is missing."}
    both = m["s2_valid"] & m["s1_valid"]
    s2, s1 = m["s2_burned"] & both, m["s1_burned"] & both
    inter, union = int((s2 & s1).sum()), int((s2 | s1).sum())
    _, _, px_km2 = ctx.grid()
    metrics = {"iou": round(inter / union, 3) if union else 0.0,
               "s2_burned_km2_common": round(float(s2.sum() * px_km2), 1),
               "s1_burned_km2_common": round(float(s1.sum() * px_km2), 1),
               "s1_detects_share_of_s2": round(inter / max(int(s2.sum()), 1), 3),
               "s2_detects_share_of_s1": round(inter / max(int(s1.sum()), 1), 3)}
    metrics["area_ratio_s1_to_s2"] = round(metrics["s1_burned_km2_common"] / metrics["s2_burned_km2_common"], 2) \
        if metrics["s2_burned_km2_common"] else 0.0
    artifacts = _quicklook(ctx) if ctx.out_dir else []
    summary = (f"Where both maps are usable: S2 {metrics['s2_burned_km2_common']} km2 vs S1 "
               f"{metrics['s1_burned_km2_common']} km2 burned; IoU {metrics['iou']}; S1 finds "
               f"{_pct(metrics['s1_detects_share_of_s2'])} of the S2 burned area. No independent reference: "
               "agreement is not accuracy.")
    return {"status": "ok", "summary": summary, "metrics": metrics, "details": {}, "artifacts": artifacts}


def _quicklook(ctx: AnalysisContext) -> list[str]:
    """PNG with the dNBR map, the S1 change map and their agreement."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception:  # plotting is optional
        return []
    m = ctx.memo
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.out_dir / "burned_area_s2_vs_s1.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = [ctx.bbox[0], ctx.bbox[2], ctx.bbox[1], ctx.bbox[3]]
    im0 = axes[0].imshow(m["s2_dnbr"], cmap="YlOrRd", vmin=0, vmax=1, extent=extent)
    axes[0].set_title(f"Sentinel-2 dNBR (burned >= {DNBR_BURNED})")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(m["s1_change"], cmap="RdBu", vmin=-4, vmax=4, extent=extent)
    axes[1].set_title(f"Sentinel-1 VH change dB (burned <= {S1_VH_DROP_DB})")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    both = m["s2_valid"] & m["s1_valid"]
    agree = np.zeros(both.shape, dtype=int)
    agree[both] = 1
    agree[both & m["s2_burned"] & ~m["s1_burned"]] = 2
    agree[both & m["s1_burned"] & ~m["s2_burned"]] = 3
    agree[both & m["s1_burned"] & m["s2_burned"]] = 4
    axes[2].imshow(agree, cmap=ListedColormap(["#d4d4d4", "#f5f5f4", "#f97316", "#3b82f6", "#7f1d1d"]),
                   vmin=0, vmax=4, extent=extent, interpolation="nearest")
    axes[2].set_title("Burned: S2 only (orange) · S1 only (blue) · both (red)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return [str(path)]


RUNNERS: dict[str, Callable[[AnalysisContext, Any], Awaitable[dict]]] = {
    "optical_conditions": optical_conditions,
    "s2_dnbr_burned_area": s2_dnbr_burned_area,
    "s1_backscatter_change": s1_backscatter_change,
    "s1_s2_agreement": s1_s2_agreement,
}


def with_dependencies(selected: list[str]) -> list[str]:
    """Selected analyses plus what they need, in run order (catalog order)."""
    wanted: set[str] = set()

    def add(name: str) -> None:
        if name in RUNNERS and name not in wanted:
            wanted.add(name)
            for dep in DEPENDS.get(name, ()):
                add(dep)

    for name in selected:
        add(name)
    return [n for n in RUNNERS if n in wanted]


# ---------------------------------------------------------------------------
# Results on disk: analyses run as separate calls (the cdse_analysis MCP server)
# ---------------------------------------------------------------------------
# The lab runs its analyses in one process and shares maps through ctx.memo. A tool call
# cannot hand an array to the next call, so each study (bbox, period, padding) gets a
# directory holding the memo (maps.npz, memo.json) and every analysis result (results.json).
# A later call loads them, so s1_s2_agreement reuses the maps the burned-area calls made.

@dataclass(frozen=True)
class AnalysisStore:
    """Saved maps and results of analysis studies, one directory per study under `root`.

    A study is an AnalysisContext's bbox, period and padding; its directory holds the memo
    (maps.npz for arrays, memo.json for the rest) and every analysis result (results.json).

    Args:
        root: directory the study directories are created in.
    """

    root: Path
    MAPS = "maps.npz"
    MEMO = "memo.json"
    RESULTS = "results.json"

    @staticmethod
    def study_key(bbox: list[float], start: str, end: str, pad_days: int) -> str:
        """Directory name of one study: readable, and identical for the same inputs."""
        box = "_".join(f"{v:.4f}" for v in bbox)
        return f"{box}__{start[:10]}_{end[:10]}__pad{pad_days}"

    def study_dir(self, ctx: AnalysisContext) -> Path:
        return self.root / self.study_key(ctx.bbox, ctx.start, ctx.end, ctx.pad_days)

    def load(self, ctx: AnalysisContext) -> dict[str, dict]:
        """Fill ctx.memo from the study's directory; return the results saved so far. An
        unreadable store (e.g. a write cut short) counts as empty, so its analyses run again."""
        study = self.study_dir(ctx)
        maps, memo, results = study / self.MAPS, study / self.MEMO, study / self.RESULTS
        try:
            if maps.exists():
                with np.load(maps) as npz:
                    ctx.memo.update({k: npz[k] for k in npz.files})
            if memo.exists():
                ctx.memo.update(json.loads(memo.read_text()))
            return json.loads(results.read_text()) if results.exists() else {}
        except (OSError, ValueError) as exc:   # JSONDecodeError and bad .npz are ValueErrors
            logger.warning("Unreadable analysis store %s (%s): recomputing", study, exc)
            ctx.memo.clear()
            return {}

    def save(self, ctx: AnalysisContext, results: dict[str, dict]) -> None:
        """Write ctx.memo and the results to the study's directory."""
        study = self.study_dir(ctx)
        study.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in ctx.memo.items() if isinstance(v, np.ndarray)}
        if arrays:
            np.savez_compressed(study / self.MAPS, **arrays)
        rest = {k: v for k, v in ctx.memo.items() if not isinstance(v, np.ndarray)}
        (study / self.MEMO).write_text(json.dumps(rest))
        (study / self.RESULTS).write_text(json.dumps(results, indent=1, default=str))

    async def run(self, name: str, ctx: AnalysisContext, client: SceneClient,
                  refresh: bool = False) -> dict:
        """Run one analysis on ctx's study, first running the analyses it needs that have no
        successful saved result; reuse the saved result unless refresh. Artifacts go to the
        study directory unless ctx.out_dir is set. Adds 'reused' (a saved result was
        returned), 'ran' (analyses run by this call) and 'study_dir'."""
        if name not in RUNNERS:
            raise ValueError(f"Unknown analysis {name!r}; one of {', '.join(RUNNERS)}.")
        study = self.study_dir(ctx)
        if ctx.out_dir is None:
            ctx.out_dir = study
        results = self.load(ctx)
        if not refresh and results.get(name, {}).get("status") == "ok":
            return {**results[name], "reused": True, "ran": [], "study_dir": str(study)}
        ran: list[str] = []
        for step in with_dependencies([name]):
            if step != name and results.get(step, {}).get("status") == "ok":
                continue
            results[step] = await self._run_one(step, ctx, client)
            ran.append(step)
            if results[step]["status"] != "ok" and step != name:
                break   # the requested analysis cannot run without it
        self.save(ctx, results)
        if name in ran:
            result = results[name]
        else:   # a dependency failed first
            failed = ran[-1]
            result = _failed(f"Not run: {failed}, which it needs, failed: {results[failed]['summary']}")
        return {**result, "reused": False, "ran": ran, "study_dir": str(study)}

    @staticmethod
    async def _run_one(step: str, ctx: AnalysisContext, client: SceneClient) -> dict:
        try:
            return await RUNNERS[step](ctx, client)
        except Exception as exc:  # network / API errors: report as a failed result
            logger.warning("analysis %s failed: %s", step, exc)
            return _failed(f"{step} failed: {exc}")


def _failed(summary: str) -> dict:
    return {"status": "failed", "summary": summary, "metrics": {}, "details": {}, "artifacts": []}
