import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import ee
from ee import data
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator

from agenticeoecosystem.definitions import ROOT_DIR


class DownloadParams(BaseModel):
	latitude: Optional[float] = Field(None, description="Latitude of the point of interest.")
	longitude: Optional[float] = Field(None, description="Longitude of the point of interest.")
	region_geojson: Optional[str] = Field(None, description="A GeoJSON string defining the region of interest.")
	bounding_box: Optional[List[float]] = Field(None, description="A bounding box as a list of four coordinates: [min_lon, min_lat, max_lon, max_lat].")
	start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
	end_date: str = Field(..., description="End date (YYYY-MM-DD)")
	scale: int = Field(10, description="Pixel resolution in meters")
	buffer_m: int = Field(1000, description="Buffer distance in meters around the point (only used with latitude/longitude).")
	bands: Optional[str] = Field("B4,B3,B2", description="Comma-separated band names to download, e.g., B4,B3,B2")
	dataset: str = Field(
		"COPERNICUS/S2_SR_HARMONIZED",
		description="ImageCollection ID, e.g., COPERNICUS/S2_SR_HARMONIZED or LANDSAT/LC08/C02/T1_L2 or COPERNICUS/S5P/NRTI/L3_NO2",
	)
	max_cloud_cover: Optional[float] = Field(50.0, description="Maximum cloud cover percentage (0-100).")
	image_count: Optional[int] = Field(1, description="Number of least cloudy images to download.")
	as_geotiff: bool = Field(
		True,
		description=(
			"If True (default), download a raw, georeferenced GeoTIFF with the "
			"original band values — for downstream quantitative use (GIS analysis, "
			"NDVI/band math, zonal stats). NOT viewable by the VLM sub-agent (TIFF "
			"is unsupported there). If False, download an 8-bit, auto-contrast-"
			"stretched PNG quicklook instead (≤3 bands) — for visual inspection "
			"(e.g. passing to the VLM sub-agent), not quantitative analysis."
		),
	)

	@model_validator(mode='before')
	@classmethod
	def check_coords_or_geojson(cls, data: Any) -> Any:
		"""Ensure either coordinates, a GeoJSON string, or a bounding box is provided."""
		if not isinstance(data, dict):
			return data
		
		lat, lon = data.get('latitude'), data.get('longitude')
		geojson = data.get('region_geojson')
		bbox = data.get('bounding_box')
		coords_provided = lat is not None and lon is not None
		geojson_provided = geojson is not None
		bbox_provided = bbox is not None

		provided_options = sum([coords_provided, geojson_provided, bbox_provided])

		if provided_options == 0:
			raise ValueError("Either latitude/longitude, a region_geojson string, or a bounding_box must be provided.")
		if provided_options > 1:
			raise ValueError("Provide only one of latitude/longitude, a region_geojson string, or a bounding_box.")
		
		if bbox_provided and len(bbox) != 4:
			raise ValueError("The bounding_box must contain exactly four coordinates: [min_lon, min_lat, max_lon, max_lat].")

		return data


# Define GEE limitations
MAX_PIXELS_PER_DOWNLOAD = 4e7  # Reduced for safety, well under the ~50M byte limit
MAX_AOI_PIXELS = 4e7           # Pre-flight rejection threshold (same value)


def _estimate_pixels(bbox: List[float], scale_m: int) -> float:
    """Rough pixel-count estimate for a lat/lon bounding box at the given scale."""
    import math
    min_lon, min_lat, max_lon, max_lat = bbox
    # A bbox crossing the antimeridian (e.g. west=170, east=-170, describing the
    # narrow strip that wraps through 180°) has max_lon < min_lon numerically.
    # Without this, (max_lon - min_lon) goes negative, silently defeating the
    # AOI-too-large check below instead of correctly flagging a huge region.
    lon_span = max_lon - min_lon
    if lon_span < 0:
        lon_span += 360
    lat_c = (min_lat + max_lat) / 2
    width_m  = lon_span * 111_000 * math.cos(math.radians(lat_c))
    height_m = (max_lat - min_lat) * 111_000
    return (width_m / scale_m) * (height_m / scale_m)


def _min_safe_scale(bbox: List[float], step: int = 10) -> int:
    """Return the smallest scale (in metres, multiple of `step`) that keeps the AOI under the limit."""
    for s in range(step, 1001, step):
        if _estimate_pixels(bbox, s) <= MAX_AOI_PIXELS:
            return s
    return 1000

# Configure logging to write to a file and stderr
log_file_path = os.path.join(os.path.dirname(__file__), "server.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SERVER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite log on each run
        logging.StreamHandler()
    ],
    force=True # Required to reconfigure logging in some environments
)

load_dotenv()
gee_project = os.getenv("GEE_PROJECT")
map_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

def setup_gee() -> None:
	logging.info("Start the GEE authentication process")
	# Try authenticating with gee-key.json first
	key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gee-key.json"))
	if os.path.exists(key_path):
		try:
			with open(key_path) as f:
				key_data = json.load(f)
			project = gee_project or key_data.get("project_id")
			if not project:
				raise ValueError("GEE project ID not found in key file or GEE_PROJECT env var.")
			credentials = ee.ServiceAccountCredentials(key_data["client_email"], key_path)
			ee.Initialize(credentials, project=project)
			logging.info("Authenticated to GEE using gee-key.json")
			return
		except Exception as e:
			logging.warning(f"Could not authenticate with gee-key.json, falling back. Error: {e}")

	# If key file fails, try service account from env vars as fallback
	if not gee_project:
		raise RuntimeError("GEE_PROJECT is not set and gee-key.json failed or was not found.")

	service_account = os.getenv("GEE_SERVICE_ACCOUNT") or os.getenv("EE_SERVICE_ACCOUNT")
	env_key_path = os.getenv("GEE_PRIVATE_KEY_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if service_account and env_key_path and os.path.exists(env_key_path):
		try:
			credentials = ee.ServiceAccountCredentials(service_account, env_key_path)
			ee.Initialize(credentials, project=gee_project)
			logging.info("Authenticated to GEE using environment variables.")
			return
		except Exception as e:
			logging.warning(f"Could not authenticate with GEE env vars, falling back. Error: {e}")

	# Fallback to default user credentials or interactive auth
	try:
		ee.Initialize(project=gee_project)
		logging.info("Authenticated to GEE using default user credentials.")
	except Exception as default_cred_error:
		logging.warning("Default credential initialization failed: %s", default_cred_error)
		try:
			logging.info("Attempting interactive GEE authentication...")
			auth_mode = os.getenv("GEE_AUTH_MODE", "gcloud")
			ee.Authenticate(auth_mode=auth_mode)
			ee.Initialize(project=gee_project)
			logging.info("Interactive GEE authentication successful.")
		except Exception as auth_error:
			raise RuntimeError(f"Fatal: All GEE authentication methods failed. Last error: {auth_error}")


# helper function to check if a region is too large to download and split it into smaller tiles if necessary
def _check_and_split_region(region: ee.Geometry, scale: int) -> List[ee.Geometry]:
	"""
	Checks if a region is too large to download by calculating estimated pixel count
	and splits it into smaller tiles if necessary.
	"""
	# Use ee.Image.pixelArea() to get the size of each pixel, then calculate total area
	try:
		# Create a dummy image to calculate the pixel area
		proj = ee.Projection('EPSG:4326').atScale(scale)
		pixel_area_image = ee.Image.pixelArea().reproject(proj)
		
		# Calculate the area of the region
		area = region.area(maxError=1).getInfo()
		
		# Calculate the number of pixels
		# Get the area of a single pixel to estimate total pixels
		# We sample a single point at the centroid for this
		pixel_area_at_centroid = pixel_area_image.reduceRegion(
			reducer=ee.Reducer.first(),
			geometry=region.centroid(maxError=1),
			scale=scale
		).get('area').getInfo()

		if pixel_area_at_centroid is None or pixel_area_at_centroid == 0:
			raise ValueError("Could not determine pixel area.")

		total_pixels = area / pixel_area_at_centroid

	except Exception as e:
		logging.error(f"Could not compute region size: {e}. Assuming it's too large and splitting.")
		total_pixels = MAX_PIXELS_PER_DOWNLOAD + 1 # Force split

	if total_pixels <= MAX_PIXELS_PER_DOWNLOAD:
		logging.info(f"Region size is acceptable ({total_pixels:.0f} pixels).")
		return [region]
	else:
		# Simple 2x2 split
		logging.info(f"Region too large ({total_pixels:.0f} pixels). Splitting into 4 quadrants.")
		bounds = region.bounds().getInfo()['coordinates'][0]
		center = region.centroid(maxError=1).getInfo()['coordinates']
		lon_c, lat_c = center[0], center[1]
		
		min_lon, min_lat = bounds[0][0], bounds[0][1]
		max_lon, max_lat = bounds[2][0], bounds[2][1]

		# Define the four new quadrants
		bottom_left = ee.Geometry.Rectangle([min_lon, min_lat, lon_c, lat_c])
		bottom_right = ee.Geometry.Rectangle([lon_c, min_lat, max_lon, lat_c])
		top_left = ee.Geometry.Rectangle([min_lon, lat_c, lon_c, max_lat])
		top_right = ee.Geometry.Rectangle([lon_c, lat_c, max_lon, max_lat])
		
		# Recursively check and split each new tile
		tiles = []
		tiles.extend(_check_and_split_region(bottom_left, scale))
		tiles.extend(_check_and_split_region(bottom_right, scale))
		tiles.extend(_check_and_split_region(top_left, scale))
		tiles.extend(_check_and_split_region(top_right, scale))
		
		return tiles


def _split_region_quadrants(region: ee.Geometry) -> List[ee.Geometry]:
	"""Split a region into four rectangular quadrants using its bounds and centroid."""
	bounds = region.bounds().getInfo()['coordinates'][0]
	center = region.centroid(maxError=1).getInfo()['coordinates']
	lon_c, lat_c = center[0], center[1]
	min_lon, min_lat = bounds[0][0], bounds[0][1]
	max_lon, max_lat = bounds[2][0], bounds[2][1]
	return [
		ee.Geometry.Rectangle([min_lon, min_lat, lon_c, lat_c]),
		ee.Geometry.Rectangle([lon_c, min_lat, max_lon, lat_c]),
		ee.Geometry.Rectangle([min_lon, lat_c, lon_c, max_lat]),
		ee.Geometry.Rectangle([lon_c, lat_c, max_lon, max_lat]),
	]


def _download_with_fallback(selected_bands: ee.Image, region: ee.Geometry, scale: int, prefix: str, depth: int = 0) -> List[str]:
	"""Try to download a region. On size-limit error, split into quadrants recursively; as a last resort, increase scale."""
	# Prepare output directory
	download_dir = os.environ.get("GEE_OUTPUT_DIR", str(ROOT_DIR / "tool_output" / "gee"))
	os.makedirs(download_dir, exist_ok=True)

	# Build download parameters
	download_region = region.bounds().getInfo()["coordinates"]
	download_params = {
		"scale": scale,
		"region": json.dumps(download_region),
		"format": "GEO_TIFF",
	}
	try:
		url = selected_bands.getDownloadURL(download_params)
		file_path = os.path.join(download_dir, f"{prefix}.tif")
		with requests.get(url, stream=True) as r:
			r.raise_for_status()
			with open(file_path, "wb") as f:
				for chunk in r.iter_content(chunk_size=8192):
					if chunk:
						f.write(chunk)
		return [os.path.abspath(file_path)]
	except Exception as e:
		err_msg = str(e)
		# GEE size limit error → split or scale backoff
		if "Total request size" in err_msg and "must be less than or equal to" in err_msg:
			logging.info(f"Tile too large at scale={scale}. depth={depth}. Attempting fallback...")
			if depth < 2:
				results: List[str] = []
				for idx, sub in enumerate(_split_region_quadrants(region), start=1):
					results.extend(_download_with_fallback(selected_bands, sub, scale, f"{prefix}_q{idx}", depth + 1))
					break
				return results
			# Last resort: increase scale (coarser resolution)
			if scale < 160 and depth < 5:
				new_scale = min(scale * 2, 160)
				logging.info(f"Increasing scale to {new_scale} and retrying for prefix={prefix}")
				return _download_with_fallback(selected_bands, region, new_scale, f"{prefix}_s{new_scale}", depth + 1)
			# Give up if still failing
			logging.warning(f"Giving up on region after fallbacks for prefix={prefix}: {err_msg}")
			return []
		# Non-size error → re-raise to be handled by caller
		raise


def _download_visual_quicklook(selected_bands: ee.Image, region: ee.Geometry, scale: int, prefix: str, dimensions: int = 768) -> Optional[str]:
	"""Download an 8-bit PNG quicklook, auto-stretched to a 2nd-98th percentile
	range per band (computed over the region) so it renders sensibly regardless
	of the dataset's native value range (reflectance, radiance, temperature,
	etc.) — GEE datasets vary too widely to hardcode a single min/max. Intended
	for visual inspection (e.g. the VLM sub-agent), not quantitative analysis."""
	download_dir = os.environ.get("GEE_OUTPUT_DIR", str(ROOT_DIR / "tool_output" / "gee"))
	os.makedirs(download_dir, exist_ok=True)

	band_names = selected_bands.bandNames().getInfo()
	vis_bands = band_names[:3]
	img = selected_bands.select(vis_bands)

	mins = maxs = None
	try:
		stats = img.reduceRegion(
			reducer=ee.Reducer.percentile([2, 98]),
			geometry=region,
			scale=scale,
			bestEffort=True,
			maxPixels=1e9,
		).getInfo()
		mins = [stats.get(f"{b}_p2") for b in vis_bands]
		maxs = [stats.get(f"{b}_p98") for b in vis_bands]
	except Exception as e:
		logging.warning(f"Could not compute auto-stretch stats for quicklook prefix={prefix}: {e}")

	if not mins or any(v is None for v in mins) or any(v is None for v in maxs):
		vis_image = img.visualize(bands=vis_bands)
	else:
		vis_image = img.visualize(bands=vis_bands, min=mins, max=maxs)

	try:
		url = vis_image.getThumbURL({
			"region": region.bounds(),
			"dimensions": dimensions,
			"format": "png",
		})
		file_path = os.path.join(download_dir, f"{prefix}.png")
		with requests.get(url, stream=True) as r:
			r.raise_for_status()
			with open(file_path, "wb") as f:
				for chunk in r.iter_content(chunk_size=8192):
					if chunk:
						f.write(chunk)
		return os.path.abspath(file_path)
	except Exception as e:
		logging.warning(f"Could not download PNG quicklook for prefix={prefix}: {e}")
		return None


app = FastMCP("gee-mcp")

_gee_initialized = False


def _ensure_gee() -> None:
    """Call setup_gee() once, redirecting stdout at the fd level so any gcloud
    subprocess output doesn't reach the MCP JSON-RPC channel on fd 1."""
    global _gee_initialized
    if _gee_initialized:
        return
    import os as _os
    import sys as _sys
    _saved_fd1 = _os.dup(1)
    _os.dup2(2, 1)
    _sys.stdout = _sys.stderr
    try:
        setup_gee()
        _gee_initialized = True
    finally:
        _os.dup2(_saved_fd1, 1)
        _os.close(_saved_fd1)
        _sys.stdout = _os.fdopen(1, "w", buffering=1)
	

_GEE_CATALOG_ROOT = "https://storage.googleapis.com/earthengine-stac/catalog/catalog.json"
_MAX_CATALOG_FETCH = 250  # cap on how many full dataset records we fetch per search call

_catalog_index_cache: Optional[List[Dict[str, str]]] = None
_provider_codes_cache: Optional[set] = None


def _fetch_catalog_children(url: str) -> List[Dict[str, str]]:
    try:
        data = requests.get(url, timeout=15).json()
    except Exception as e:
        logging.warning(f"Could not fetch GEE catalog node {url}: {e}")
        return []
    return [
        {"title": link["title"], "href": link["href"]}
        for link in data.get("links", [])
        if link.get("rel") == "child"
    ]


def _build_catalog_index() -> List[Dict[str, str]]:
    """Fetch and cache a flat {title, href} index of every dataset in the official
    GEE STAC catalog (one request per top-level provider, cached for the lifetime
    of this server process). `title` here is the provider-prefixed asset slug
    (e.g. "MODIS_061_MYD28M1"), not a human-readable name."""
    global _catalog_index_cache, _provider_codes_cache
    if _catalog_index_cache is not None:
        return _catalog_index_cache

    providers = _fetch_catalog_children(_GEE_CATALOG_ROOT)
    _provider_codes_cache = {p["title"].lower() for p in providers}

    index: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        for children in pool.map(_fetch_catalog_children, [p["href"] for p in providers]):
            index.extend(children)

    logging.info(f"Built GEE catalog index with {len(index)} datasets.")
    _catalog_index_cache = index
    return index


def _fetch_catalog_item(href: str) -> Optional[Dict[str, Any]]:
    try:
        return requests.get(href, timeout=15).json()
    except Exception as e:
        logging.warning(f"Could not fetch GEE catalog item {href}: {e}")
        return None


@app.tool()
def search_gee_catalog(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the official Google Earth Engine STAC catalog for dataset IDs matching a
    keyword query (e.g. a sensor name, variable, or product like 'MODIS land surface
    temperature' or 'NOAA sea surface temperature').

    Use this whenever you are not fully certain of the exact, current ImageCollection ID
    to pass as `dataset` to download_gee_product. GEE dataset IDs get re-versioned over
    time (e.g. a collection moving from .../006/... to .../061/...), so do not rely on a
    memorized ID for anything beyond the handful of very common ones (Sentinel-2 SR,
    Landsat 8/9 SR, Sentinel-5P NO2) — look it up here first instead.

    Tip: include the sensor/agency name if you know it (e.g. "MODIS", "NOAA",
    "COPERNICUS", "LANDSAT") — this narrows the search to that provider's catalog and
    gives far more accurate results, since many products (e.g. from NASA/NOAA) are
    identified by opaque codes rather than descriptive names.

    An empty result means nothing matched — that can mean the exact product you're
    thinking of isn't hosted on GEE at all (e.g. MODIS SST is not); try a different
    provider or a broader/looser query before giving up.

    Returns up to `max_results` candidate datasets, each with the real Earth Engine asset
    `id` (pass this exact string as `dataset`), `title`, `type` (image or
    image_collection), and a short `description`.
    """
    index = _build_catalog_index()
    terms = [t for t in query.lower().replace("-", " ").split() if t]
    if not terms:
        return []

    provider_codes = _provider_codes_cache or set()
    matched_providers = {t for t in terms if t in provider_codes}

    if matched_providers:
        candidates = [
            e for e in index
            if e["title"].split("_")[0].lower() in matched_providers
        ]
    else:
        def _slug_score(entry: Dict[str, str]) -> int:
            slug_l = entry["title"].lower()
            tokens = slug_l.replace("-", "_").split("_")
            return sum(1 for t in terms if t in slug_l or any(t in tok for tok in tokens))

        candidates = sorted(
            (e for e in index if _slug_score(e) > 0),
            key=_slug_score,
            reverse=True,
        )

    if not candidates:
        return []
    candidates = candidates[:_MAX_CATALOG_FETCH]

    def _fetch_and_score(entry: Dict[str, str]):
        item = _fetch_catalog_item(entry["href"])
        if not item:
            return None
        text = " ".join(filter(None, [
            item.get("title"), item.get("description"), " ".join(item.get("keywords") or []),
        ])).lower()
        score = sum(text.count(t) for t in terms)
        return score, item

    scored_items = []
    with ThreadPoolExecutor(max_workers=25) as pool:
        for result in pool.map(_fetch_and_score, candidates):
            if result:
                scored_items.append(result)

    # Different versions of the SAME product (e.g. .../V1, .../V2, .../V3, or
    # MODIS/006/... vs MODIS/061/...) often score very differently on raw keyword
    # count alone — older/more verbose descriptions can out-score a concise newer
    # one — so text score can't be trusted to naturally surface the current
    # version. Group by product family (id with version-like segments stripped)
    # and, within each family, keep only the best version: not deprecated, then
    # most recent temporal coverage — independent of text score. The family's
    # best text score is kept for ranking families against each other.
    def _version_pref_key(item: Dict[str, Any]) -> tuple:
        not_deprecated = "[deprecated]" not in (item.get("title") or "").lower()
        extent = _extract_temporal_extent(item)
        end_date = extent[1] if extent and extent[1] else "9999-99-99"
        return (not_deprecated, end_date)

    def _family_key(dataset_id: str) -> str:
        parts = (dataset_id or "").split("/")
        kept = [p for p in parts if not re.fullmatch(r"[Vv]?\d+(?:_\d+)?", p)]
        return "/".join(kept) if kept else dataset_id

    families: Dict[str, Dict[str, Any]] = {}
    for score, item in scored_items:
        fam = _family_key(item.get("id") or "")
        existing = families.get(fam)
        if existing is None:
            families[fam] = {"score": score, "item": item}
        else:
            existing["score"] = max(existing["score"], score)
            if _version_pref_key(item) > _version_pref_key(existing["item"]):
                existing["item"] = item

    deduped = [(f["score"], f["item"]) for f in families.values()]
    deduped.sort(key=lambda x: x[0], reverse=True)

    top_items = [item for score, item in deduped if score > 0][:max_results]
    return [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "type": item.get("gee:type"),
            "description": (item.get("description") or "")[:300],
            "temporal_extent": _extract_temporal_extent(item),
        }
        for item in top_items
    ]


def _extract_temporal_extent(item: Dict[str, Any]) -> Optional[List[Optional[str]]]:
    """Pull [start, end] (ISO date strings, or None if open-ended/unavailable) out of
    a GEE STAC item's `extent.temporal.interval`. Many datasets — especially older
    sensor versions superseded by a newer one — only cover a fixed historical period;
    querying outside it silently returns zero images, so this must be checked before
    picking start_date/end_date."""
    try:
        interval = item.get("extent", {}).get("temporal", {}).get("interval", [])
        if interval and interval[0]:
            start, end = interval[0][0], interval[0][1]
            return [start.split("T")[0] if start else None, end.split("T")[0] if end else None]
    except Exception:
        pass
    return None


def _dataset_id_to_catalog_url(dataset_id: str) -> str:
    provider = dataset_id.split("/")[0]
    slug = dataset_id.replace("/", "_")
    return f"{_GEE_CATALOG_ROOT.rsplit('/', 1)[0]}/{provider}/{slug}.json"


def _fetch_dataset_temporal_extent(dataset_id: str) -> Optional[List[Optional[str]]]:
    """Look up a dataset's official temporal extent directly by id (used as a
    fallback diagnostic when a date-range query returns zero images)."""
    item = _fetch_catalog_item(_dataset_id_to_catalog_url(dataset_id))
    return _extract_temporal_extent(item) if item else None


@app.tool()
def get_lat_lon(address: str) -> List[float]:
    """Get latitude and longitude for a given address using Google Maps API."""
    # API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    if not map_api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment.")

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": map_api_key
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if not data.get("results"):
        # Surface Google's actual status/error_message (e.g. REQUEST_DENIED
        # with "This API is not activated on your API project" — observed to
        # fire for the same (key, address) pair depending on the caller's
        # egress IP, consistent with an IP-restricted API key rather than a
        # genuine "place not found") instead of a generic message that hides
        # the real cause. No fallback geocoder here by design: GEE and CDSE
        # are deliberately kept as separate tools with separate geocoders so
        # the agent's own tool choice (CDSE by default, GEE only for
        # collections CDSE doesn't cover) is what determines reliability —
        # see the "geocode via CDSE first" routing rule in both the
        # multi-agent orchestrator and single-agent system prompts.
        status = data.get("status", "UNKNOWN")
        detail = data.get("error_message")
        msg = f"Geocoding failed for {address!r}: status={status}"
        if detail:
            msg += f" — {detail}"
        raise ValueError(msg)

    loc = data["results"][0]["geometry"]["location"]
    return [loc["lat"], loc["lng"]]


@app.tool()
def download_gee_product(args: DownloadParams) -> Dict[str, Any]:
	"""Download satellite images from GEE and return a dictionary with file paths and a status message.
	Arguments include coordinates, date range, dataset, bands, scale, buffer, cloud cover, and image count.

	Set `as_geotiff=True` (default) for a raw analysis-ready GeoTIFF (GIS/quantitative use).
	Set `as_geotiff=False` for an 8-bit, auto-stretched PNG quicklook instead — use this when the
	output is meant for visual inspection (e.g. passed to the VLM sub-agent), since TIFF isn't
	viewable there.
	"""
	# Pre-flight AOI size check — reject immediately if the request would take forever.
	if args.bounding_box:
		estimated = _estimate_pixels(args.bounding_box, args.scale)
		if estimated > MAX_AOI_PIXELS:
			min_scale = _min_safe_scale(args.bounding_box)
			raise ValueError(
				f"AOI too large: ~{estimated / 1e6:.0f}M pixels at scale={args.scale}m "
				f"(limit is {MAX_AOI_PIXELS / 1e6:.0f}M). "
				f"Retry with scale>={min_scale}m or reduce the bounding box."
			)

	_ensure_gee()
	# Determine the region of interest from either GeoJSON, a bounding box, or a point
	if args.region_geojson:
		try:
			geojson_dict = json.loads(args.region_geojson)
			logging.info(f"GeoJSON: {geojson_dict}")
			region = ee.Geometry(geojson_dict)
		except Exception as e:
			raise ValueError(f"Invalid GeoJSON provided: {e}")
	elif args.bounding_box:
		try:
			min_lon, min_lat, max_lon, max_lat = args.bounding_box
			logging.info(f"Bounding box: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
			# A bbox crossing the antimeridian (e.g. west=170, east=-170, describing
			# the narrow strip that wraps through 180°) has max_lon < min_lon
			# numerically. Shifting the east edge by +360 keeps the vertices
			# monotonically increasing so GEE builds the intended narrow polygon
			# instead of interpreting it as wrapping the other way (~half the globe).
			max_lon_adj = max_lon + 360 if max_lon < min_lon else max_lon
			# Define the four corners of the rectangle to create a Polygon
			coords = [
				[min_lon, min_lat],
				[max_lon_adj, min_lat],
				[max_lon_adj, max_lat],
				[min_lon, max_lat],
				[min_lon, min_lat] # Close the polygon
			]
			region = ee.Geometry.Polygon(coords, None, False)
		except Exception as e:
			raise ValueError(f"Invalid bounding_box provided. Ensure it is a list of four numbers. Error: {e}")
	else:
		point = ee.Geometry.Point([args.longitude, args.latitude])
		logging.info(f"Point: {point}")
		region = point.buffer(args.buffer_m)

	# --- Pre-flight check and AOI Tiling ---
	logging.info("Performing pre-flight checks for region size...")
	tiles = _check_and_split_region(region, args.scale)
	if len(tiles) > 1:
		logging.info(f"Region is too large for a single download. Splitting into {len(tiles)} tiles.")

	downloaded_files = []
	extent_note = None
	for i, tile_region in enumerate(tiles):
		if len(tiles) > 1:
			logging.info(f"Processing tile {i + 1} of {len(tiles)}...")

		collection = (
			ee.ImageCollection(args.dataset)
			.filterDate(args.start_date, args.end_date)
			.filterBounds(tile_region)
		)

		# Filter by cloud cover if the dataset supports it and the param is provided
		if args.dataset.startswith("COPERNICUS/S2") and args.max_cloud_cover is not None:
			collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', args.max_cloud_cover))

		# First, check if any images meet the cloud cover criteria before proceeding
		collection_size = collection.size().getInfo()
		if collection_size == 0:
			logging.warning(f"No images found for tile {i + 1} that meet the specified cloud cover rate. Skipping.")
			if extent_note is None:
				extent = _fetch_dataset_temporal_extent(args.dataset)
				if extent:
					start, end = extent
					extent_note = (
						f"No images matched start_date={args.start_date}, end_date={args.end_date} for "
						f"dataset '{args.dataset}' — this dataset's actual temporal coverage is "
						f"{start or 'unknown'} to {end or 'present'}. If your requested range falls "
						f"outside that, retry with dates inside it, or use search_gee_catalog to find "
						f"a newer version of this product with more recent coverage."
					)
			continue

		# Sort by cloud cover and limit to the number of images requested
		collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE").limit(args.image_count)

		image_list = collection.toList(args.image_count)
		num_images_found = image_list.size().getInfo()
		
		for j in range(num_images_found):
			image = ee.Image(image_list.get(j))

			# Get the image's capture date for the filename
			date_string = ee.Date(image.get('system:time_start')).format('YYYYMMdd').getInfo()

			# Select bands for each image
			selected_bands = None
			if args.bands:
				band_list = [b.strip() for b in args.bands.split(",") if b.strip()]
				available_bands = image.bandNames().getInfo()
				missing = [b for b in band_list if b not in available_bands]
				if missing:
					raise ValueError(
						f"Band(s) {missing} not found in dataset '{args.dataset}'. "
						f"Available bands for this image: {available_bands}. "
						f"Retry with one of these exact band names instead of guessing."
					)
				selected_bands = image.select(band_list)
			else:
				selected_bands = image
			
			# Generate a filename prefix including capture date and tile index
			prefix = f"gee_image_{date_string}_t{i+1}_n{j+1}"

			if args.as_geotiff:
				# Use robust fallback downloader (recursive split/scale backoff)
				file_paths = _download_with_fallback(selected_bands, tile_region, args.scale, prefix)
			else:
				quicklook_path = _download_visual_quicklook(selected_bands, tile_region, args.scale, prefix)
				file_paths = [quicklook_path] if quicklook_path else []

			if not file_paths:
				logging.warning(f"No files produced for tile {i+1}, image {j+1} after fallbacks.")
			else:
				downloaded_files.extend(file_paths)

	message = f"Successfully downloaded {len(downloaded_files)} image(s) across {len(tiles)} tile(s)."
	if len(tiles) > 1:
		message += f" (Note: Some images were skipped due to insufficient cloud cover or no images found in some tiles.)"
	if extent_note:
		message += " " + extent_note

	return {"files": downloaded_files, "message": message}

@app.prompt()
def get_llm_prompt(query: str) -> str:
	"""Generates a a prompt for the LLM to use to answer the query"""

	return f"""
	You are an expert GEE assistant responsible for translating user requests into precise tool calls.
	Your task is to answer the user's query about satellite image downloads by using the `download_gee_product` tool.

	**VERY IMPORTANT:** You must determine the user's region of interest and format it correctly.
	- If the user provides a latitude and longitude, use the `latitude` and `longitude` parameters.
	- If the user provides a bounding box (e.g., [min_lon, min_lat, max_lon, max_lat]), you have two options:
	  1. **(Preferred)** Pass the coordinates as a simple list to the `bounding_box` parameter.
	  2. Convert it into a valid GeoJSON string and pass it to the `region_geojson` parameter.
	- Use the `bounding_box` parameter for simple rectangular areas. Use `region_geojson` for more complex shapes like polygons.
	- Coordinates MUST be in [longitude, latitude] order. Do NOT swap to [latitude, longitude].
	- For bounding boxes, preserve the exact order [min_lon, min_lat, max_lon, max_lat].
	- For GeoJSON, every coordinate pair must be [lon, lat].
	- Provide only one of: coordinates, a bounding box, or GeoJSON.

	Extract all other parameters like dates, cloud cover, and image count from the query.

	Query: {query}
	"""


if __name__ == "__main__":
    app.run(transport="stdio")

