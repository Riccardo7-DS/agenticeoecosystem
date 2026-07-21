from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Optional
from pathlib import Path
import os

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
    "Return file paths for saved outputs (maps, rasters, vectors).\n"
    "To clip a raster to a region: use clip_raster_with_shapefile only if the "
    "caller already gave you a .shp path. If you only have a bounding box "
    "(west, south, east, north) and no shapefile, use clip_raster_with_bbox "
    "directly — do not try to fabricate or guess a shapefile path.\n"
    "Spectral index tools (all take a local multi-band raster + 1-based band "
    "indices and write a single-band float32 GeoTIFF): you must already have "
    "that raster on disk (e.g. from the GEE sub-agent's download_gee_product) "
    "before calling any of these — none of them fetch imagery themselves.\n"
    "  - compute_ndvi(source, red_band_index, nir_band_index, destination): "
    "(NIR-Red)/(NIR+Red) — general vegetation condition. Sentinel-2: B8=NIR, B4=Red.\n"
    "  - compute_evi(source, red_band_index, nir_band_index, blue_band_index, "
    "destination, scale_factor=0.0001): 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1) — "
    "dense vegetation with atmospheric correction. Sentinel-2: B8=NIR, B4=Red, "
    "B2=Blue. Keep scale_factor=0.0001 for Sentinel-2 SR (DN 0-10000); use 1.0 "
    "only if the source raster is already reflectance (0-1).\n"
    "  - compute_ndwi(source, green_band_index, nir_band_index, destination): "
    "(Green-NIR)/(Green+NIR) — McFeeters water detection. Sentinel-2: B3=Green, B8=NIR.\n"
    "  - compute_ndmi(source, nir_band_index, swir1_band_index, destination): "
    "(NIR-SWIR1)/(NIR+SWIR1) — vegetation moisture / drought stress. Sentinel-2: "
    "B8=NIR, B11=SWIR1.\n"
    "  - compute_nbr(source, nir_band_index, swir2_band_index, destination): "
    "(NIR-SWIR2)/(NIR+SWIR2) — burn severity; compare pre-/post-fire scenes as "
    "dNBR (pre-NBR minus post-NBR). Sentinel-2: B8=NIR, B12=SWIR2."
)


def clip_raster_with_bbox(
    raster_path: str,
    bbox: list[float],
    destination: str,
    bbox_crs: str = "EPSG:4326",
) -> dict:
    """Clip a raster to a bounding box, without requiring a shapefile.

    Args:
        raster_path: Local path to the source raster.
        bbox: [west, south, east, north].
        destination: Local path to write the clipped raster to.
        bbox_crs: CRS of the bbox coordinates (default WGS-84 lon/lat).

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    src_path = os.path.expanduser(raster_path.replace("`", ""))
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Raster not found at '{src_path}'.")

    west, south, east, north = bbox

    with rasterio.open(src_path) as src:
        if src.crs and str(src.crs) != bbox_crs:
            west, south, east, north = transform_bounds(
                bbox_crs, src.crs, west, south, east, north
            )
        window = from_bounds(west, south, east, north, transform=src.transform)
        out_transform = src.window_transform(window)
        out_image = src.read(window=window)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    dst_path = Path(os.path.expanduser(destination.replace("`", "")))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(dst_path), "w", **out_meta) as dst:
        dst.write(out_image)

    return {
        "status": "success",
        "destination": str(dst_path),
        "message": f"Raster clipped to bbox and saved to '{dst_path}'.",
    }


def compute_evi(
    source: str,
    red_band_index: int,
    nir_band_index: int,
    blue_band_index: int,
    destination: str,
    scale_factor: float = 0.0001,
) -> dict:
    """Compute EVI (Enhanced Vegetation Index) and save to GeoTIFF.

    Improves on NDVI for dense vegetation canopies and corrects for
    atmospheric and soil-background effects using the blue band.

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B8, B4, B2).
        red_band_index: Index of the red band (1-based).
        nir_band_index: Index of the near-infrared band (1-based).
        blue_band_index: Index of the blue band (1-based).
        destination: Output EVI raster path.
        scale_factor: Multiplier applied to raw band values to convert them to
            reflectance (0-1) before applying the formula — EVI's "+1" constant
            is only valid in reflectance units. Sentinel-2 SR products store
            DN 0-10000, so the default 0.0001 is correct for them; pass 1.0 if
            the source raster is already scaled to reflectance.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        red = src.read(red_band_index).astype("float32") * scale_factor
        nir = src.read(nir_band_index).astype("float32") * scale_factor
        blue = src.read(blue_band_index).astype("float32") * scale_factor
        denom = nir + 6 * red - 7.5 * blue + 1
        evi = 2.5 * (nir - red) / np.where(denom == 0, np.nan, denom)
        evi = np.nan_to_num(evi, nan=0.0).astype("float32")

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(evi, 1)

    return {
        "status": "success",
        "destination": str(dst_path),
        "message": f"EVI calculated and saved to '{dst_path}'.",
    }


def compute_ndwi(
    source: str,
    green_band_index: int,
    nir_band_index: int,
    destination: str,
) -> dict:
    """Compute NDWI (McFeeters Normalized Difference Water Index) and save to GeoTIFF.

    Detects open water bodies; higher values indicate water.

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B3, B8).
        green_band_index: Index of the green band (1-based).
        nir_band_index: Index of the near-infrared band (1-based).
        destination: Output NDWI raster path.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        green = src.read(green_band_index).astype("float32")
        nir = src.read(nir_band_index).astype("float32")
        ndwi = (green - nir) / (green + nir + 1e-6)

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(ndwi, 1)

    return {
        "status": "success",
        "destination": str(dst_path),
        "message": f"NDWI calculated and saved to '{dst_path}'.",
    }


def compute_ndmi(
    source: str,
    nir_band_index: int,
    swir1_band_index: int,
    destination: str,
) -> dict:
    """Compute NDMI (Normalized Difference Moisture Index) and save to GeoTIFF.

    Indicates vegetation moisture content; useful for detecting drought stress.

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B8, B11).
        nir_band_index: Index of the near-infrared band (1-based).
        swir1_band_index: Index of the SWIR1 band (1-based, ~1.6um; Sentinel-2 B11).
        destination: Output NDMI raster path.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        nir = src.read(nir_band_index).astype("float32")
        swir1 = src.read(swir1_band_index).astype("float32")
        ndmi = (nir - swir1) / (nir + swir1 + 1e-6)

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(ndmi, 1)

    return {
        "status": "success",
        "destination": str(dst_path),
        "message": f"NDMI calculated and saved to '{dst_path}'.",
    }


def compute_nbr(
    source: str,
    nir_band_index: int,
    swir2_band_index: int,
    destination: str,
) -> dict:
    """Compute NBR (Normalized Burn Ratio) and save to GeoTIFF.

    Used to identify burned areas and assess burn severity; often compared
    pre-/post-fire as dNBR (pre-fire NBR minus post-fire NBR).

    Args:
        source: Input multi-band raster path (e.g. Sentinel-2 bands B8, B12).
        nir_band_index: Index of the near-infrared band (1-based).
        swir2_band_index: Index of the SWIR2 band (1-based, ~2.2um; Sentinel-2 B12).
        destination: Output NBR raster path.

    Returns:
        dict with status, destination path, and a message.
    """
    import rasterio
    import numpy as np

    src_path = os.path.expanduser(source.replace("`", ""))
    dst_path = os.path.expanduser(destination.replace("`", ""))

    with rasterio.open(src_path) as src:
        nir = src.read(nir_band_index).astype("float32")
        swir2 = src.read(swir2_band_index).astype("float32")
        nbr = (nir - swir2) / (nir + swir2 + 1e-6)

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(nbr, 1)

    return {
        "status": "success",
        "destination": str(dst_path),
        "message": f"NBR calculated and saved to '{dst_path}'.",
    }


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
                tools=toolset + [
                    clip_raster_with_bbox,
                    compute_evi,
                    compute_ndwi,
                    compute_ndmi,
                    compute_nbr,
                ],
                mode='single_turn'
            )
        else:
            logger.warning("GIS agent not initialized due to missing toolset.")
            return None


gis_agent = _fetch_agent(toolset)
