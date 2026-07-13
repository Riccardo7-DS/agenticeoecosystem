from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Optional

import logging
logger = logging.getLogger(__name__)

_GIS_SYSTEM_PROMPT = (
    "You are a GIS analysis assistant. Use the available tools to perform spatial operations, "
    "raster analysis, coordinate transformations, spatial statistics, map visualization."
    "Available capabilities include:\n"
    "- Geometry operations (buffer, intersect, union, centroid, simplify, etc.) via Shapely\n"
    "- Vector file operations (read/write/clip/merge shapefiles and GeoJSON) via GeoPandas\n"
    "- Raster analysis (NDVI, zonal statistics, clip, reproject, hillshade) via Rasterio\n"
    "- Coordinate transforms and geodetic calculations via PyProj\n"
    "- Spatial statistics (Moran's I, LISA, spatial regression) via PySAL\n"
    "- Map creation (static PNG/PDF via create_map, interactive HTML via create_web_map)\n"
    "Extract all parameters from the user's request and chain tools as needed. "
    "Return file paths for saved outputs (maps, rasters, vectors)."
)

_GIS_TOOL_NAMES = {
    # Shapely geometry
    "buffer", "intersection", "union", "difference", "symmetric_difference",
    "convex_hull", "envelope", "minimum_rotated_rectangle", "centroid",
    "bounds", "coordinates", "geometry_type", "rotate", "scale", "translate",
    "triangulate", "voronoi", "unary_union", "length", "area",
    "is_valid", "make_valid", "simplify", "snap", "nearest_point",
    "normalize", "geometry_to_geojson", "geojson_to_geometry",
    # PyProj coordinate transforms
    "transform_coordinates", "project_geometry",
    "get_crs_info", "get_available_crs", "get_utm_zone", "get_utm_crs", "get_geocentric_crs",
    "get_geod_info", "calculate_geodetic_distance", "calculate_geodetic_point", "calculate_geodetic_area",
    # GeoPandas vector
    "read_file_gpd", "write_file_gpd", "append_gpd", "merge_gpd",
    "overlay_gpd", "dissolve_gpd", "explode_gpd", "clip_vector",
    "sjoin_gpd", "sjoin_nearest_gpd", "point_in_polygon",
    # Rasterio raster
    "metadata_raster", "get_raster_crs", "extract_band", "raster_band_statistics",
    "raster_histogram", "clip_raster_with_shapefile", "resample_raster",
    "reproject_raster", "tile_raster", "compute_ndvi", "raster_algebra",
    "concat_bands", "weighted_band_sum", "zonal_statistics",
    "reclassify_raster", "focal_statistics", "hillshade", "write_raster",
    # PySAL spatial statistics
    "morans_i", "gearys_c", "gamma_statistic", "getis_ord_g",
    "moran_local", "getis_ord_g_local", "join_counts_local", "adbscan",
    "weights_from_shapefile", "distance_band_weights", "knn_weights",
    "build_transform_and_save_weights", "ols_with_spatial_diagnostics_safe",
    "build_and_transform_weights", "spatial_markov", "dynamic_lisa", "gm_lag",
    # Visualization
    "create_map", "create_web_map",
    # Data downloads
    "download_boundaries", "download_climate_data",
    "get_species_info", "download_species_occurrences",
    "download_street_network", "calculate_shortest_path",
    "download_worldcover", "compute_s2_ndvi", "download_satellite_imagery",
}

_GIS_DOWNLOAD_TOOL_NAMES = {
    "download_boundaries", "download_climate_data",
    "get_species_info", "download_species_occurrences",
    "download_street_network", "calculate_shortest_path",
    "download_worldcover", "compute_s2_ndvi", "download_satellite_imagery",
}

_GIS_ALLOWED_TOOL_NAMES = sorted(_GIS_TOOL_NAMES - _GIS_DOWNLOAD_TOOL_NAMES)

toolset = [McpToolset(
                     connection_params=StdioConnectionParams(
                         server_params=StdioServerParameters(
                             command='uv',
                             args=[
                                    "tool",
                                    "run",
                                    "--python", "3.12",
                                    "gis-mcp"
                                ]),
                             timeout=300),
                     tool_filter=_GIS_ALLOWED_TOOL_NAMES)]


def _fetch_agent(toolset) -> Optional[Agent]:
        if toolset:
            return Agent(
                model=llm,
                name='gis_agent',
                description='A helpful assistant for performing GIS analysis, spatial operations, raster analysis, and map visualization.',
                instruction=_GIS_SYSTEM_PROMPT,
                tools=toolset,
                mode='single_turn'
            )
        else:
            logger.warning("GIS agent not initialized due to missing toolset.")
            return None


gis_agent = _fetch_agent(toolset)
