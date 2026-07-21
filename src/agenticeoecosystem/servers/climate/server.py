import os
import re
import cdsapi
from dotenv import load_dotenv
import xarray as xr
import rioxarray  # noqa: F401 -- registers the .rio accessor used by netcdf_to_geotiff
import pathlib
import time
import uuid
from pathlib import Path
from fastmcp import FastMCP
import asyncio
from typing import Union, List, Optional
import logging
import sys

from agenticeoecosystem.definitions import ROOT_DIR
from agenticeoecosystem.utils.output_paths import resolve_run_path

# --- Logging Setup ---
log_file_path = os.path.join(os.path.dirname(__file__), "era5_server.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ERA5 - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler(sys.stderr),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# --- MCP Server Setup ---
mcp = FastMCP(name="ERA5 Climate Data Server")

# --- Env & Auth ---
env_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(env_path)
URL = os.getenv("CDS_URL")
KEY = os.getenv("CDS_KEY")

# ---------------------------------------------------------------------------
# Output directory / default save-path handling
#
# Every write from this server is forced onto disk under OUTPUT_DIR, no
# matter what path the caller supplies:
#   - output_filename omitted            -> auto-generated name under OUTPUT_DIR
#   - a relative path (e.g. "data/x.nc") -> re-rooted under OUTPUT_DIR
#   - an absolute path elsewhere         -> re-rooted under OUTPUT_DIR
#   - "output/<subfolder>/x.nc"          -> re-rooted to OUTPUT_DIR/<subfolder>/x.nc,
#                                            preserving any subfolder structure
#                                            below a leading "output/" segment
#                                            (matches this benchmark's own
#                                            "output/<task_id>/file.nc" convention)
#
# This also fixes a real bug: previously nothing ever created the parent
# directory of output_filename, so any path whose directory didn't already
# exist (e.g. a relative "data/..." the model invented) failed with a
# generic "Error during CDS API request: [Errno 2] No such file or
# directory", which the model then "fixed" by blindly retrying with a
# different path (e.g. /tmp, since that happens to already exist) instead
# of the retry actually being necessary.
# ---------------------------------------------------------------------------

OUTPUT_DIR = pathlib.Path(
    os.environ.get("ERA5_OUTPUT_DIR", str(ROOT_DIR / "output" / "era5"))
)


def _default_output_path(label: str, ext: str = "nc") -> str:
    """Build a unique default filename (no caller-supplied path) under OUTPUT_DIR."""
    ts = time.strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6]
    safe_label = "".join(c if (c.isalnum() or c in "-_") else "_" for c in label)
    return f"{safe_label}_{ts}_{uid}.{ext}"


def _coerce_output_path(output_filename: Optional[str], label: str, ext: str = "nc") -> str:
    """
    Resolve any output path (missing, relative, absolute, or elsewhere) to a
    concrete path rooted under OUTPUT_DIR, and make sure its parent
    directory exists before anything tries to write to it.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not output_filename:
        rel_parts = [_default_output_path(label, ext=ext)]
    else:
        s = str(output_filename).replace("\\", "/")
        s = re.sub(r"^[A-Za-z]:", "", s)  # strip a Windows drive letter, if any
        s = s.lstrip("/")                  # strip a leading absolute-path slash
        parts = [seg for seg in s.split("/") if seg not in ("", ".")]
        # Drop a single leading "output" segment (e.g. "output/TASK_ID/file.nc")
        # so it doesn't become OUTPUT_DIR/output/TASK_ID/file.nc.
        if parts and parts[0].lower() == "output":
            parts = parts[1:]
        if not parts:
            parts = [Path(output_filename).name or _default_output_path(label, ext=ext)]
        rel_parts = parts

    dest = OUTPUT_DIR.joinpath(*rel_parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return str(dest)


# ---------------------------------------------------------------------------
# Static catalogue data
# ---------------------------------------------------------------------------

_ERA5_SINGLE_LEVEL_VARIABLES = {
    "thermodynamic": [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "skin_temperature",
        "maximum_2m_temperature_since_previous_post_processing",
        "minimum_2m_temperature_since_previous_post_processing",
    ],
    "precipitation_and_hydrology": [
        "total_precipitation",
        "large_scale_precipitation",
        "convective_precipitation",
        "snowfall",
        "evaporation",
        "potential_evaporation",
        "runoff",
        "surface_runoff",
        "sub_surface_runoff",
        "snow_evaporation",
        "snowmelt",
        "large_scale_precipitation_fraction",
    ],
    "wind": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "instantaneous_10m_wind_gust",
        "friction_velocity",
    ],
    "radiation": [
        "surface_solar_radiation_downwards",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
        "surface_thermal_radiation_downwards",
        "toa_incident_solar_radiation",
        "top_net_solar_radiation",
        "top_net_thermal_radiation",
        "surface_net_solar_radiation_clear_sky",
        "surface_net_thermal_radiation_clear_sky",
        "surface_solar_radiation_downward_clear_sky",
        "surface_thermal_radiation_downward_clear_sky",
        "total_sky_direct_solar_radiation_at_surface",
        "uv_visible_albedo_for_diffuse_radiation",
        "uv_visible_albedo_for_direct_radiation",
        "near_ir_albedo_for_diffuse_radiation",
        "near_ir_albedo_for_direct_radiation",
    ],
    "pressure_and_dynamics": [
        "mean_sea_level_pressure",
        "surface_pressure",
        "boundary_layer_height",
        "convective_available_potential_energy",
        "convective_inhibition",
        "k_index",
        "total_totals_index",
        "trapping_layer_base_height",
        "trapping_layer_top_height",
        "zero_degree_level",
    ],
    "clouds": [
        "total_cloud_cover",
        "high_cloud_cover",
        "medium_cloud_cover",
        "low_cloud_cover",
        "total_column_cloud_ice_water",
        "total_column_cloud_liquid_water",
        "total_column_supercooled_liquid_water",
    ],
    "soil_and_snow": [
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
        "snow_depth",
        "snow_depth_water_equivalent",
        "snow_density",
        "snow_cover",
        "snow_albedo",
        "temperature_of_snow_layer",
        "high_vegetation_cover",
        "low_vegetation_cover",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation",
        "type_of_high_vegetation",
        "type_of_low_vegetation",
        "soil_type",
        "land_sea_mask",
    ],
    "ocean_and_sea_ice": [
        "sea_surface_temperature",
        "sea_ice_cover",
    ],
    "moisture_and_ozone": [
        "total_column_water",
        "total_column_water_vapour",
        "total_column_ozone",
        "total_column_rain_water",
        "total_column_snow_water",
    ],
    "heat_flux_and_stress": [
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
    ],
    "lake": [
        "lake_cover",
        "lake_depth",
        "lake_bottom_temperature",
        "lake_mix_layer_temperature",
        "lake_mix_layer_depth",
        "lake_shape_factor",
        "lake_total_layer_temperature",
        "lake_ice_depth",
        "lake_ice_surface_temperature",
    ],
}

_ERA5_PRESSURE_LEVEL_VARIABLES = [
    "divergence",
    "fraction_of_cloud_cover",
    "geopotential",
    "ozone_mass_mixing_ratio",
    "potential_vorticity",
    "relative_humidity",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_humidity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "vorticity",
]

_ERA5_PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
    225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
    775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]

_CDS_DATASETS = {
    "reanalysis-era5-single-levels": {
        "description": "ERA5 hourly data on single levels (1940–present, 0.25°). Use for surface/screen-level variables at sub-daily temporal resolution.",
        "temporal_resolution": "hourly",
        "spatial_resolution": "0.25°",
        "period": "1940–present",
    },
    "reanalysis-era5-single-levels-monthly-means": {
        "description": "ERA5 monthly mean data on single levels (1940–present, 0.25°). Fastest download for climatological studies.",
        "temporal_resolution": "monthly",
        "spatial_resolution": "0.25°",
        "period": "1940–present",
    },
    "reanalysis-era5-pressure-levels": {
        "description": "ERA5 hourly data on 37 pressure levels (1940–present, 0.25°). Use for atmospheric profile variables.",
        "temporal_resolution": "hourly",
        "spatial_resolution": "0.25°",
        "period": "1940–present",
    },
    "reanalysis-era5-pressure-levels-monthly-means": {
        "description": "ERA5 monthly mean data on pressure levels (1940–present, 0.25°).",
        "temporal_resolution": "monthly",
        "spatial_resolution": "0.25°",
        "period": "1940–present",
    },
    "reanalysis-era5-land": {
        "description": "ERA5-Land hourly data (1950–present, 0.1°). Higher resolution than ERA5, land variables only.",
        "temporal_resolution": "hourly",
        "spatial_resolution": "0.1°",
        "period": "1950–present",
    },
    "reanalysis-era5-land-monthly-means": {
        "description": "ERA5-Land monthly mean data (1950–present, 0.1°).",
        "temporal_resolution": "monthly",
        "spatial_resolution": "0.1°",
        "period": "1950–present",
    },
    "derived-era5-single-levels-daily-statistics": {
        "description": "Pre-computed ERA5 daily statistics (daily_mean, daily_maximum, daily_minimum, daily_sum) on single levels (1940–present). Avoids downloading hourly data.",
        "temporal_resolution": "daily",
        "spatial_resolution": "0.25°",
        "period": "1940–present",
    },
    "derived-era5-pressure-levels-daily-statistics": {
        "description": "Pre-computed ERA5 daily statistics on pressure levels (1940–present).",
        "temporal_resolution": "daily",
        "spatial_resolution": "0.25°",
        "period": "1940–present",
    },
    "projections-cmip6": {
        "description": "CMIP6 climate model projections from multiple modelling centres. Various scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5).",
        "temporal_resolution": "varies (daily to monthly)",
        "spatial_resolution": "varies by model",
        "period": "1850–2100 (historical + projections)",
    },
    "seasonal-original-single-levels": {
        "description": "Copernicus Climate Change Service seasonal forecast data on single levels.",
        "temporal_resolution": "monthly",
        "spatial_resolution": "varies by system",
        "period": "varies by system",
    },
    "sis-agrometeorological-indicators": {
        "description": "Agrometeorological indicators derived from ERA5 for agricultural applications (temperature, precipitation, humidity, wind).",
        "temporal_resolution": "daily",
        "spatial_resolution": "0.1°",
        "period": "1979–present",
    },
}

_UNIT_CONVERSIONS = {
    ("K", "C"): ("subtract", 273.15),
    ("K", "degC"): ("subtract", 273.15),
    ("kelvin", "celsius"): ("subtract", 273.15),
    ("m", "mm"): ("multiply", 1000.0),
    ("m s**-1", "km h**-1"): ("multiply", 3.6),
    ("m s**-1", "knots"): ("multiply", 1.94384),
    ("Pa", "hPa"): ("multiply", 0.01),
    ("J m**-2", "W m**-2"): ("multiply", 1.0 / 3600.0),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_days(days: Union[str, List[str]]) -> List[str]:
    if isinstance(days, str) and days.lower() == "all":
        return [f"{d:02d}" for d in range(1, 32)]
    return days if isinstance(days, list) else [days]


def _resolve_times(times: Union[str, List[str]]) -> List[str]:
    if isinstance(times, str) and times.lower() == "all":
        return [f"{h:02d}:00" for h in range(24)]
    return times if isinstance(times, list) else [times]


# ---------------------------------------------------------------------------
# Internal Core Logic
# ---------------------------------------------------------------------------

async def _internal_inspect_netcdf(filepath: str) -> str:
    def _sync_inspect():
        resolved = resolve_run_path(filepath)
        if not resolved.exists():
            return f"Error: File not found at '{filepath}'"
        try:
            with xr.open_dataset(resolved) as ds:
                summary = [f"--- NetCDF Inspection Summary for {resolved.name} ---", "\n[Dimensions]"]
                summary.extend(f"- {dim}: {size}" for dim, size in ds.dims.items())
                summary.append("\n[Coordinates]")
                summary.extend(
                    f"- {name} ({c.dims}): {c.attrs.get('long_name', '')} [{c.attrs.get('units', '')}]"
                    for name, c in ds.coords.items()
                )
                summary.append("\n[Variables]")
                summary.extend(
                    f"- {name} ({v.dims}): {v.attrs.get('long_name', '')} [{v.attrs.get('units', '')}]"
                    for name, v in ds.data_vars.items()
                )
                summary.append("\n--- End of Summary ---")
                return "\n".join(summary)
        except Exception as e:
            return f"Error inspecting NetCDF file: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_inspect)


async def _internal_fetch_and_inspect(dataset: str, request: dict, output_filename: str) -> str:
    # output_filename has already been forced under OUTPUT_DIR (with its
    # parent directory created) by _coerce_output_path at the call site —
    # resolve_run_path is still applied for consistency with the rest of
    # the codebase, and we defensively re-create the parent dir here too
    # in case resolve_run_path maps it somewhere unexpected.
    output_path = resolve_run_path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _blocking_download():
        logger.info("Initializing cdsapi.Client...")
        try:
            c = cdsapi.Client(url=URL, key=KEY)
            logger.info(f"Starting download for dataset '{dataset}' to '{output_path}'...")
            start_time = time.time()
            c.retrieve(dataset, request, str(output_path))
            elapsed = time.time() - start_time
            logger.info(f"Download completed in {elapsed:.2f}s.")
            inspection = asyncio.run(_internal_inspect_netcdf(str(output_path)))
            return f"Successfully downloaded data to '{output_path}'.\n\n{inspection}"
        except Exception as e:
            error_message = f"Error during CDS API request: {e}"
            logger.error(error_message)
            return error_message

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _blocking_download)


# ---------------------------------------------------------------------------
# MCP Tools — Discovery
# ---------------------------------------------------------------------------

@mcp.tool
async def list_era5_single_level_variables() -> str:
    """
    Returns all available ERA5 single-level (surface) variable names grouped by category,
    as accepted by the CDS API. Use these exact names in fetch_era5_* tools.
    """
    lines = ["ERA5 Single-Level Variables (use exact names in CDS requests):\n"]
    for category, variables in _ERA5_SINGLE_LEVEL_VARIABLES.items():
        lines.append(f"[{category.replace('_', ' ').title()}]")
        lines.extend(f"  - {v}" for v in variables)
        lines.append("")
    return "\n".join(lines)


@mcp.tool
async def list_era5_pressure_level_variables() -> str:
    """
    Returns all available ERA5 pressure-level variable names and the valid
    pressure levels (hPa) accepted by the CDS API.
    """
    variables_str = "\n".join(f"  - {v}" for v in _ERA5_PRESSURE_LEVEL_VARIABLES)
    levels_str = ", ".join(str(lv) for lv in _ERA5_PRESSURE_LEVELS)
    return (
        "ERA5 Pressure-Level Variables:\n"
        f"{variables_str}\n\n"
        f"Valid pressure levels (hPa):\n  {levels_str}"
    )


@mcp.tool
async def list_cds_datasets() -> str:
    """
    Returns the most useful CDS (Copernicus Climate Data Store) datasets with their
    descriptions, temporal/spatial resolution, and time coverage.
    """
    lines = ["Available CDS Datasets:\n"]
    for name, info in _CDS_DATASETS.items():
        lines.append(f"[{name}]")
        lines.append(f"  Description:          {info['description']}")
        lines.append(f"  Temporal resolution:  {info['temporal_resolution']}")
        lines.append(f"  Spatial resolution:   {info['spatial_resolution']}")
        lines.append(f"  Period:               {info['period']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP Tools — Inspection
# ---------------------------------------------------------------------------

@mcp.tool
async def inspect_netcdf(filepath: str) -> str:
    """
    Inspects a NetCDF file and returns a summary of its contents (dimensions, variables, coordinates).

    :param filepath: The absolute path to the .nc file to inspect.
    """
    return await _internal_inspect_netcdf(filepath)


# ---------------------------------------------------------------------------
# MCP Tools — Downloads: Monthly Means (existing)
# ---------------------------------------------------------------------------

@mcp.tool
async def fetch_era5_single_levels(
    variable: str, year: str, month: Union[str, List[str]], output_filename: Optional[str] = None
) -> str:
    """
    Downloads ERA5 monthly mean surface data (product: monthly_averaged_reanalysis).

    :param variable: The surface variable (e.g., '2m_temperature'). Use list_era5_single_level_variables() for valid names.
    :param year: The year for the data (e.g., '2023').
    :param month: The month(s) for the data (e.g., '01', '12', or ['06', '07', '08'] to fetch
        multiple months in a single request instead of one call per month).
    :param output_filename: Desired filename/path (e.g. 'output/TASK_ID/2m_temp_2023_01.nc').
        Always saved under ERA5_OUTPUT_DIR regardless of what's passed here — a leading
        "output/" segment is preserved as a subfolder, everything else is re-rooted.
        If omitted, a unique filename is auto-generated.
    """
    output_filename = _coerce_output_path(output_filename, f"single_levels_monthly_{variable}")
    request = {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': variable,
        'year': year,
        'month': month,
        'time': '00:00',
        'data_format': 'netcdf',
    }
    return await _internal_fetch_and_inspect('reanalysis-era5-single-levels-monthly-means', request, output_filename)


@mcp.tool
async def fetch_era5_pressure_levels(
    variable: str,
    pressure_level: int,
    year: Union[str, List[str]],
    month: Union[str, List[str]],
    output_filename: Optional[str] = None,
) -> str:
    """
    Downloads ERA5 monthly mean data on specific pressure levels.

    :param variable: The variable (e.g., 'geopotential', 'temperature'). Use list_era5_pressure_level_variables() for valid names.
    :param pressure_level: The pressure level in hPa (e.g., 500, 850, 1000).
    :param year: The year(s) for the data (e.g., '2023' or ['2021', '2022', '2023']).
    :param month: The month(s) for the data (e.g., '01', '12', or ['06', '07', '08'] to fetch
        multiple months in a single request instead of one call per month).
    :param output_filename: Desired filename/path. Always saved under ERA5_OUTPUT_DIR
        regardless of what's passed here (see fetch_era5_single_levels for the exact rule).
        If omitted, a unique filename is auto-generated.
    """
    output_filename = _coerce_output_path(output_filename, f"pressure_levels_monthly_{variable}_{pressure_level}")
    request = {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': variable,
        'pressure_level': str(pressure_level),
        'year': year,
        'month': month,
        'time': '00:00',
        'data_format': 'netcdf',
    }
    return await _internal_fetch_and_inspect('reanalysis-era5-pressure-levels-monthly-means', request, output_filename)


# ---------------------------------------------------------------------------
# MCP Tools — Downloads: Hourly
# ---------------------------------------------------------------------------

@mcp.tool
async def fetch_era5_hourly_single_levels(
    variable: str,
    year: str,
    month: str,
    output_filename: Optional[str] = None,
    days: Union[str, List[str]] = "all",
    times: Union[str, List[str]] = "all",
    area: Optional[List[float]] = None,
) -> str:
    """
    Downloads ERA5 hourly data on single levels (surface/screen-level variables).

    :param variable: The surface variable (e.g., 'total_precipitation', '2m_temperature'). Use list_era5_single_level_variables() for valid names.
    :param year: The year for the data (e.g., '2023').
    :param month: The month for the data (e.g., '01').
    :param output_filename: Desired filename/path. Always saved under ERA5_OUTPUT_DIR
        regardless of what's passed here (see fetch_era5_single_levels for the exact rule).
        If omitted, a unique filename is auto-generated.
    :param days: List of days (e.g., ['01','02','03']) or 'all' for the full month. Default: 'all'.
    :param times: List of times (e.g., ['00:00','06:00','12:00','18:00']) or 'all' for all 24 hours. Default: 'all'.
    :param area: Optional bounding box [north, west, south, east] in decimal degrees (e.g., [71.0, -25.0, 35.0, 45.0] for Europe).
    """
    output_filename = _coerce_output_path(output_filename, f"single_levels_hourly_{variable}_{year}{month}")
    request = {
        'product_type': 'reanalysis',
        'variable': variable,
        'year': year,
        'month': month,
        'day': _resolve_days(days),
        'time': _resolve_times(times),
        'data_format': 'netcdf',
    }
    if area:
        request['area'] = area
    return await _internal_fetch_and_inspect('reanalysis-era5-single-levels', request, output_filename)


@mcp.tool
async def fetch_era5_hourly_pressure_levels(
    variable: str,
    pressure_level: Union[int, List[int]],
    year: str,
    month: str,
    output_filename: Optional[str] = None,
    days: Union[str, List[str]] = "all",
    times: Union[str, List[str]] = "all",
    area: Optional[List[float]] = None,
) -> str:
    """
    Downloads ERA5 hourly data on specific pressure levels.

    :param variable: The pressure-level variable (e.g., 'temperature', 'geopotential'). Use list_era5_pressure_level_variables() for valid names.
    :param pressure_level: The pressure level(s) in hPa (e.g., 500, or [500, 850, 1000]).
    :param year: The year for the data (e.g., '2023').
    :param month: The month for the data (e.g., '01').
    :param output_filename: Desired filename/path. Always saved under ERA5_OUTPUT_DIR
        regardless of what's passed here (see fetch_era5_single_levels for the exact rule).
        If omitted, a unique filename is auto-generated.
    :param days: List of days (e.g., ['01','02','03']) or 'all' for the full month. Default: 'all'.
    :param times: List of times (e.g., ['00:00','06:00','12:00','18:00']) or 'all' for all 24 hours. Default: 'all'.
    :param area: Optional bounding box [north, west, south, east] in decimal degrees.
    """
    output_filename = _coerce_output_path(output_filename, f"pressure_levels_hourly_{variable}_{year}{month}")
    levels = [str(pressure_level)] if isinstance(pressure_level, int) else [str(lv) for lv in pressure_level]
    request = {
        'product_type': 'reanalysis',
        'variable': variable,
        'pressure_level': levels,
        'year': year,
        'month': month,
        'day': _resolve_days(days),
        'time': _resolve_times(times),
        'data_format': 'netcdf',
    }
    if area:
        request['area'] = area
    return await _internal_fetch_and_inspect('reanalysis-era5-pressure-levels', request, output_filename)


@mcp.tool
async def fetch_era5_land(
    variable: str,
    year: str,
    month: str,
    output_filename: Optional[str] = None,
    days: Union[str, List[str]] = "all",
    times: Union[str, List[str]] = "all",
    area: Optional[List[float]] = None,
) -> str:
    """
    Downloads ERA5-Land hourly data (0.1° spatial resolution, land surfaces only).
    ERA5-Land is higher resolution than ERA5 but restricted to land variables.

    :param variable: A land variable (e.g., 'total_precipitation', 'volumetric_soil_water_layer_1').
    :param year: The year for the data (e.g., '2023').
    :param month: The month for the data (e.g., '01').
    :param output_filename: Desired filename/path. Always saved under ERA5_OUTPUT_DIR
        regardless of what's passed here (see fetch_era5_single_levels for the exact rule).
        If omitted, a unique filename is auto-generated.
    :param days: List of days or 'all'. Default: 'all'.
    :param times: List of times or 'all'. Default: 'all'.
    :param area: Optional bounding box [north, west, south, east] in decimal degrees.
    """
    output_filename = _coerce_output_path(output_filename, f"land_{variable}_{year}{month}")
    request = {
        'variable': variable,
        'year': year,
        'month': month,
        'day': _resolve_days(days),
        'time': _resolve_times(times),
        'data_format': 'netcdf',
    }
    if area:
        request['area'] = area
    return await _internal_fetch_and_inspect('reanalysis-era5-land', request, output_filename)


# ---------------------------------------------------------------------------
# MCP Tools — Downloads: Pre-computed Daily Statistics
# ---------------------------------------------------------------------------

@mcp.tool
async def fetch_era5_daily_statistics(
    variable: str,
    statistic: str,
    year: str,
    month: str,
    output_filename: Optional[str] = None,
    days: Union[str, List[str]] = "all",
    area: Optional[List[float]] = None,
) -> str:
    """
    Downloads pre-computed ERA5 daily statistics on single levels.
    More efficient than downloading hourly data and aggregating locally.

    :param variable: The surface variable (e.g., 'total_precipitation', '2m_temperature').
    :param statistic: The daily statistic: 'daily_mean', 'daily_maximum', 'daily_minimum', or 'daily_sum'.
    :param year: The year for the data (e.g., '2023').
    :param month: The month for the data (e.g., '01').
    :param output_filename: Desired filename/path. Always saved under ERA5_OUTPUT_DIR
        regardless of what's passed here (see fetch_era5_single_levels for the exact rule).
        If omitted, a unique filename is auto-generated.
    :param days: List of days or 'all'. Default: 'all'.
    :param area: Optional bounding box [north, west, south, east] in decimal degrees.
    """
    valid_stats = {'daily_mean', 'daily_maximum', 'daily_minimum', 'daily_sum'}
    if statistic not in valid_stats:
        return f"Error: statistic must be one of {sorted(valid_stats)}, got '{statistic}'."
    output_filename = _coerce_output_path(output_filename, f"daily_stats_{statistic}_{variable}_{year}{month}")
    request = {
        'product_type': 'reanalysis',
        'variable': variable,
        'year': year,
        'month': month,
        'day': _resolve_days(days),
        'daily_statistic': statistic,
        'time_zone': 'UTC+00:00',
        'frequency': '1_hourly',
        'data_format': 'netcdf',
    }
    if area:
        request['area'] = area
    return await _internal_fetch_and_inspect('derived-era5-single-levels-daily-statistics', request, output_filename)


# ---------------------------------------------------------------------------
# MCP Tools — Downloads: Generic escape hatch
# ---------------------------------------------------------------------------

@mcp.tool
async def fetch_cds_dataset(dataset_name: str, request_dict: dict, output_filename: Optional[str] = None) -> str:
    """
    Generic CDS API fetch for any dataset not covered by the specific fetch tools
    (e.g., CMIP6, seasonal forecasts, ERA5-complete, agrometeorological indicators).
    Use list_cds_datasets() to discover dataset names, then build the request dict
    from the CDS catalogue documentation.

    :param dataset_name: The CDS dataset identifier (e.g., 'projections-cmip6').
    :param request_dict: A dictionary with the CDS API request parameters.
    :param output_filename: Desired filename/path. Always saved under ERA5_OUTPUT_DIR
        regardless of what's passed here (see fetch_era5_single_levels for the exact rule).
        If omitted, a unique filename is auto-generated.
    """
    output_filename = _coerce_output_path(output_filename, dataset_name)
    return await _internal_fetch_and_inspect(dataset_name, request_dict, output_filename)


# ---------------------------------------------------------------------------
# MCP Tools — Temporal Processing
# ---------------------------------------------------------------------------

@mcp.tool
async def convert_hourly_to_daily(
    input_filepath: str,
    output_filepath: Optional[str] = None,
    method: str = "sum",
    variable: Optional[str] = None,
) -> str:
    """
    Aggregates an hourly ERA5 NetCDF file to daily frequency using xarray resample.
    For precipitation (total_precipitation, unit: m/hour), use method='sum' to get
    daily totals in m/day (multiply by 1000 afterwards via convert_units for mm/day).
    For temperature use 'mean'; for extremes use 'max' or 'min'.

    :param input_filepath: Path to the input hourly NetCDF file.
    :param output_filepath: Desired filename/path for the output. Always saved under
        ERA5_OUTPUT_DIR regardless of what's passed here. If omitted, a unique filename
        is auto-generated.
    :param method: Aggregation method: 'sum', 'mean', 'max', or 'min'. Default: 'sum'.
    :param variable: If provided, process only this variable; otherwise all data variables.
    """
    output_filepath = _coerce_output_path(output_filepath, f"hourly_to_daily_{method}")

    def _sync():
        resolved_input = resolve_run_path(input_filepath)
        resolved_output = resolve_run_path(output_filepath)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not resolved_input.exists():
                return f"Error: File not found at '{input_filepath}'"
            valid_methods = {'sum', 'mean', 'max', 'min'}
            if method not in valid_methods:
                return f"Error: method must be one of {sorted(valid_methods)}, got '{method}'."
            with xr.open_dataset(resolved_input) as ds:
                if variable:
                    if variable not in ds.data_vars:
                        return f"Error: variable '{variable}' not found. Available: {list(ds.data_vars)}"
                    ds = ds[[variable]]
                resampled = getattr(ds.resample(time="1D"), method)()
                resampled.to_netcdf(resolved_output)
                dims = dict(resampled.dims)
                vars_info = {v: str(resampled[v].attrs.get('units', 'units unchanged')) for v in resampled.data_vars}
            return (
                f"Aggregated hourly → daily ({method}) and saved to '{output_filepath}'.\n"
                f"Dimensions: {dims}\n"
                f"Variables: {vars_info}"
            )
        except Exception as e:
            return f"Error during hourly→daily conversion: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


@mcp.tool
async def convert_daily_to_monthly(
    input_filepath: str,
    output_filepath: Optional[str] = None,
    method: str = "mean",
    variable: Optional[str] = None,
) -> str:
    """
    Aggregates a daily NetCDF file to monthly frequency using xarray resample.

    :param input_filepath: Path to the input daily NetCDF file.
    :param output_filepath: Desired filename/path for the output. Always saved under
        ERA5_OUTPUT_DIR regardless of what's passed here. If omitted, a unique filename
        is auto-generated.
    :param method: Aggregation method: 'sum', 'mean', 'max', or 'min'. Default: 'mean'.
    :param variable: If provided, process only this variable; otherwise all data variables.
    """
    output_filepath = _coerce_output_path(output_filepath, f"daily_to_monthly_{method}")

    def _sync():
        resolved_input = resolve_run_path(input_filepath)
        resolved_output = resolve_run_path(output_filepath)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not resolved_input.exists():
                return f"Error: File not found at '{input_filepath}'"
            valid_methods = {'sum', 'mean', 'max', 'min'}
            if method not in valid_methods:
                return f"Error: method must be one of {sorted(valid_methods)}, got '{method}'."
            with xr.open_dataset(resolved_input) as ds:
                if variable:
                    if variable not in ds.data_vars:
                        return f"Error: variable '{variable}' not found. Available: {list(ds.data_vars)}"
                    ds = ds[[variable]]
                resampled = getattr(ds.resample(time="ME"), method)()
                resampled.to_netcdf(resolved_output)
                dims = dict(resampled.dims)
                vars_info = {v: str(resampled[v].attrs.get('units', 'units unchanged')) for v in resampled.data_vars}
            return (
                f"Aggregated daily → monthly ({method}) and saved to '{output_filepath}'.\n"
                f"Dimensions: {dims}\n"
                f"Variables: {vars_info}"
            )
        except Exception as e:
            return f"Error during daily→monthly conversion: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


@mcp.tool
async def resample_temporal(
    input_filepath: str,
    freq: str,
    method: str,
    output_filepath: Optional[str] = None,
    variable: Optional[str] = None,
) -> str:
    """
    General-purpose temporal resampling of a NetCDF file.

    :param input_filepath: Path to the input NetCDF file.
    :param freq: Pandas-style resample frequency: 'h' (hourly), 'D' (daily), 'ME' (month-end), 'YE' (year-end).
    :param method: Aggregation method: 'sum', 'mean', 'max', or 'min'.
    :param output_filepath: Desired filename/path for the output. Always saved under
        ERA5_OUTPUT_DIR regardless of what's passed here. If omitted, a unique filename
        is auto-generated.
    :param variable: If provided, process only this variable; otherwise all data variables.
    """
    output_filepath = _coerce_output_path(output_filepath, f"resample_{freq}_{method}")

    def _sync():
        resolved_input = resolve_run_path(input_filepath)
        resolved_output = resolve_run_path(output_filepath)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not resolved_input.exists():
                return f"Error: File not found at '{input_filepath}'"
            valid_methods = {'sum', 'mean', 'max', 'min'}
            if method not in valid_methods:
                return f"Error: method must be one of {sorted(valid_methods)}, got '{method}'."
            with xr.open_dataset(resolved_input) as ds:
                if variable:
                    if variable not in ds.data_vars:
                        return f"Error: variable '{variable}' not found. Available: {list(ds.data_vars)}"
                    ds = ds[[variable]]
                resampled = getattr(ds.resample(time=freq), method)()
                resampled.to_netcdf(resolved_output)
                dims = dict(resampled.dims)
                vars_info = {v: str(resampled[v].attrs.get('units', 'units unchanged')) for v in resampled.data_vars}
            return (
                f"Resampled to freq='{freq}' ({method}) and saved to '{output_filepath}'.\n"
                f"Dimensions: {dims}\n"
                f"Variables: {vars_info}"
            )
        except Exception as e:
            return f"Error during temporal resampling: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


# ---------------------------------------------------------------------------
# MCP Tools — Unit Conversion & Spatial
# ---------------------------------------------------------------------------

@mcp.tool
async def convert_units(
    input_filepath: str,
    variable: str,
    from_unit: str,
    to_unit: str,
    output_filepath: Optional[str] = None,
) -> str:
    """
    Converts the unit of a variable in a NetCDF file and writes a new file.
    Supported conversions:
      - Temperature:  'K' → 'C'  or  'kelvin' → 'celsius'  (subtract 273.15)
      - Precipitation depth:  'm' → 'mm'  (multiply by 1000)
      - Wind speed:  'm s**-1' → 'km h**-1'  or  'm s**-1' → 'knots'
      - Pressure:  'Pa' → 'hPa'
      - Radiation (accumulated):  'J m**-2' → 'W m**-2'  (divide by 3600)

    :param input_filepath: Path to the input NetCDF file.
    :param variable: The variable to convert (e.g., 'tp', 't2m', or the long name shown by inspect_netcdf).
    :param from_unit: The current unit string (must match exactly as listed above).
    :param to_unit: The target unit string.
    :param output_filepath: Desired filename/path for the output. Always saved under
        ERA5_OUTPUT_DIR regardless of what's passed here. If omitted, a unique filename
        is auto-generated.
    """
    output_filepath = _coerce_output_path(output_filepath, f"convert_units_{variable}_{from_unit}_to_{to_unit}")

    def _sync():
        resolved_input = resolve_run_path(input_filepath)
        resolved_output = resolve_run_path(output_filepath)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not resolved_input.exists():
                return f"Error: File not found at '{input_filepath}'"
            key = (from_unit, to_unit)
            if key not in _UNIT_CONVERSIONS:
                supported = [f"  '{f}' → '{t}'" for f, t in _UNIT_CONVERSIONS]
                return "Error: Unsupported conversion '{}' → '{}'. Supported:\n{}".format(
                    from_unit, to_unit, "\n".join(supported)
                )
            op, factor = _UNIT_CONVERSIONS[key]
            with xr.open_dataset(resolved_input) as ds:
                if variable not in ds.data_vars:
                    return f"Error: variable '{variable}' not found. Available: {list(ds.data_vars)}"
                ds_out = ds.copy(deep=True)
                if op == "subtract":
                    ds_out[variable] = ds_out[variable] - factor
                elif op == "multiply":
                    ds_out[variable] = ds_out[variable] * factor
                ds_out[variable].attrs['units'] = to_unit
                ds_out.to_netcdf(resolved_output)
            return f"Converted '{variable}' from '{from_unit}' to '{to_unit}' and saved to '{output_filepath}'."
        except Exception as e:
            return f"Error during unit conversion: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


@mcp.tool
async def clip_to_bbox(
    input_filepath: str,
    north: float,
    south: float,
    east: float,
    west: float,
    output_filepath: Optional[str] = None,
) -> str:
    """
    Clips a NetCDF file to a geographic bounding box and writes a new file.
    Works with ERA5 files which use 'latitude' and 'longitude' coordinate names.

    :param input_filepath: Path to the input NetCDF file.
    :param north: Northern boundary in decimal degrees (e.g., 60.0).
    :param south: Southern boundary in decimal degrees (e.g., 35.0).
    :param east: Eastern boundary in decimal degrees (e.g., 30.0).
    :param west: Western boundary in decimal degrees (e.g., -10.0).
    :param output_filepath: Desired filename/path for the output. Always saved under
        ERA5_OUTPUT_DIR regardless of what's passed here. If omitted, a unique filename
        is auto-generated.
    """
    output_filepath = _coerce_output_path(output_filepath, "clip_to_bbox")

    def _sync():
        resolved_input = resolve_run_path(input_filepath)
        resolved_output = resolve_run_path(output_filepath)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not resolved_input.exists():
                return f"Error: File not found at '{input_filepath}'"
            with xr.open_dataset(resolved_input) as ds:
                lat_name = next((c for c in ds.coords if 'lat' in c.lower()), None)
                lon_name = next((c for c in ds.coords if 'lon' in c.lower()), None)
                if not lat_name or not lon_name:
                    return f"Error: Could not find lat/lon coordinates. Available: {list(ds.coords)}"
                # ERA5 stores latitudes in descending order (90 → -90): use slice(north, south)
                clipped = ds.sel({lat_name: slice(north, south), lon_name: slice(west, east)})
                # Fallback for ascending latitude order
                if clipped[lat_name].size == 0:
                    clipped = ds.sel({lat_name: slice(south, north), lon_name: slice(west, east)})
                clipped.to_netcdf(resolved_output)
                dims = dict(clipped.dims)
            return (
                f"Clipped to bbox [N={north}, S={south}, E={east}, W={west}] "
                f"and saved to '{output_filepath}'.\nDimensions: {dims}"
            )
        except Exception as e:
            return f"Error during spatial clip: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


@mcp.tool
async def netcdf_to_geotiff(
    input_filepath: str,
    variable: str,
    output_filepath: Optional[str] = None,
    time_index: int = 0,
    level_index: int = 0,
) -> str:
    """
    Exports a single 2D (lat, lon) slice of a NetCDF variable as a georeferenced
    GeoTIFF, so it can be viewed via the VLM server's resolve_images/load_images
    (which auto-stretch GeoTIFFs into a viewable PNG). ERA5 files themselves are
    never directly viewable — this is the required step in between.

    :param input_filepath: Path to the input NetCDF file (e.g. from fetch_era5_*
        or clip_to_bbox).
    :param variable: Name of the variable to export (e.g. 'z' for geopotential,
        't' for temperature, 'sst' for sea surface temperature). Check
        inspect_netcdf for the exact variable name in the file.
    :param output_filepath: Desired filename/path for the .tif. Always saved under
        ERA5_OUTPUT_DIR regardless of what's passed here. If omitted, a unique
        filename is auto-generated.
    :param time_index: Index along the time dimension to export, if the variable
        has one (default 0, the first time step).
    :param level_index: Index along the pressure-level dimension to export, if
        the variable has one (default 0, the first/only level).
    """
    output_filepath = _coerce_output_path(output_filepath, f"{variable}_geotiff", ext="tif")

    def _sync():
        resolved_input = resolve_run_path(input_filepath)
        resolved_output = resolve_run_path(output_filepath)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not resolved_input.exists():
                return f"Error: File not found at '{input_filepath}'"
            with xr.open_dataset(resolved_input) as ds:
                if variable not in ds.data_vars:
                    return f"Error: variable '{variable}' not found. Available: {list(ds.data_vars)}"
                da = ds[variable]
                lat_name = next((c for c in da.dims if 'lat' in c.lower()), None)
                lon_name = next((c for c in da.dims if 'lon' in c.lower()), None)
                if not lat_name or not lon_name:
                    return f"Error: Could not find lat/lon dimensions. Available: {list(da.dims)}"

                for dim in list(da.dims):
                    if dim in (lat_name, lon_name):
                        continue
                    if dim in ('time', 'valid_time', 'date'):
                        da = da.isel({dim: time_index})
                    elif 'level' in dim.lower() or 'pressure' in dim.lower():
                        da = da.isel({dim: level_index})
                    else:
                        da = da.isel({dim: 0})

                da = da.rename({lat_name: "y", lon_name: "x"})
                da.rio.write_crs("EPSG:4326", inplace=True)
                da.rio.to_raster(resolved_output)
            return f"Exported '{variable}' (time_index={time_index}, level_index={level_index}) to GeoTIFF at '{output_filepath}'."
        except Exception as e:
            return f"Error during GeoTIFF export: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


@mcp.tool
async def compute_wind_speed(
    u_filepath: str,
    v_filepath: str,
    output_filepath: Optional[str] = None,
    u_variable: Optional[str] = None,
    v_variable: Optional[str] = None,
) -> str:
    """
    Computes wind speed magnitude (sqrt(u^2 + v^2)) from separate u- and v-component
    NetCDF files and writes a new NetCDF file containing a 'wind_speed' variable.
    fetch_era5_* tools only accept one variable per call, so fetch u_component_of_wind
    and v_component_of_wind (or their 10m surface equivalents) separately first, then
    combine them here.

    :param u_filepath: Path to the NetCDF file containing the u-component (zonal wind).
    :param v_filepath: Path to the NetCDF file containing the v-component (meridional wind).
    :param output_filepath: Desired filename/path for the output. Always saved under
        ERA5_OUTPUT_DIR regardless of what's passed here. If omitted, a unique filename
        is auto-generated.
    :param u_variable: Name of the u-component variable in u_filepath. If omitted, the first data variable is used.
    :param v_variable: Name of the v-component variable in v_filepath. If omitted, the first data variable is used.
    """
    output_filepath = _coerce_output_path(output_filepath, "wind_speed")

    def _sync():
        resolved_u = resolve_run_path(u_filepath)
        resolved_v = resolve_run_path(v_filepath)
        resolved_output = resolve_run_path(output_filepath)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not resolved_u.exists():
                return f"Error: File not found at '{u_filepath}'"
            if not resolved_v.exists():
                return f"Error: File not found at '{v_filepath}'"
            with xr.open_dataset(resolved_u) as ds_u, xr.open_dataset(resolved_v) as ds_v:
                u_name = u_variable or list(ds_u.data_vars)[0]
                v_name = v_variable or list(ds_v.data_vars)[0]
                if u_name not in ds_u.data_vars:
                    return f"Error: variable '{u_name}' not found in u_filepath. Available: {list(ds_u.data_vars)}"
                if v_name not in ds_v.data_vars:
                    return f"Error: variable '{v_name}' not found in v_filepath. Available: {list(ds_v.data_vars)}"
                u = ds_u[u_name]
                v = ds_v[v_name]
                wind_speed = (u ** 2 + v ** 2) ** 0.5
                wind_speed.name = "wind_speed"
                wind_speed.attrs["units"] = u.attrs.get("units", "m s**-1")
                wind_speed.attrs["long_name"] = "wind speed"
                wind_speed.to_dataset().to_netcdf(resolved_output)
                dims = dict(wind_speed.sizes)
            return (
                f"Computed wind speed from '{u_name}' ({u_filepath}) and '{v_name}' ({v_filepath}), "
                f"saved to '{output_filepath}'.\nDimensions: {dims}"
            )
        except Exception as e:
            return f"Error computing wind speed: {e}"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()