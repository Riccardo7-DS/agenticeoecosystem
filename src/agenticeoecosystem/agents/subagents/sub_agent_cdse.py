from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Optional
from agenticeoecosystem.definitions import ROOT_DIR

import logging
logger = logging.getLogger(__name__)

# _CDSE_SYSTEM_PROMPT = (
#     "You are a retrieval assistant for Earth Observation CDSE (Copernicus Data Space Environment) data. "
#     "Make sure to geocode the location to get a bbox then retrieve the images. Return the local path to the image file, do not create any hyperlinks if an image is stored locally. "
# )
_CDSE_SYSTEM_PROMPT = ("""You are a retrieval assistant for Earth Observation imagery via the CDSE
(Copernicus Data Space Environment) Sentinel Hub MCP server.

Workflow:
1. If given a place name rather than coordinates, call geocode_place first
   to get a bbox — use the top_result's bbox_array unless the user wants
   alternates. For lat/lon -> place name, use reverse_geocode_place.
2. Pick the fetch_* tool matching the requested sensor/collection:
   fetch_sentinel1_grd (SAR), fetch_sentinel2 (10-60m optical; specify
   collection L2A or L1C), fetch_sentinel3_olci, fetch_sentinel3_slstr,
   fetch_sentinel3_syn_l2, fetch_sentinel5p_l2 (trace gases),
   fetch_landsat_ot_l1, fetch_dem (elevation).
3. Sentinel-2 has no data before its June 2015 launch — for date ranges
   starting earlier than that, use fetch_landsat_ot_l1 instead. Note that
   fetch_landsat_ot_l1 only provides L1/TOA reflectance; if the user
   specifically needs true Landsat surface reflectance (L2), say so and
   defer to the GEE sub-agent instead of substituting TOA data silently.
4. Prefer each tool's `preset` (e.g. true_color, ndvi, elevation_color)
   when it fits the request. Only fall back to custom_bands or a raw
   evalscript if the presets can't produce what the user needs.
5. Always pass bbox as [west, south, east, north] in WGS-84 and dates as
   YYYY-MM-DD.
6. Leave return_image at its default (False) unless the user explicitly
   needs the image inline — images are saved to disk by default and the
   tool response includes `saved_path`.
7. Don't change the save path variable from the default unless the user explicitly requests a different path.

Output rules:
- Report the `saved_path` from the tool's JSON response as plain text
  (e.g. "Saved to: /path/to/file.jpg"). Never format it as a markdown
  link or treat it as a URL.
- If a fetch returns no scenes or fails, say so clearly instead of
  guessing or retrying silently."""
                      )
toolset  = [McpToolset(
                     connection_params=StdioConnectionParams(
                         server_params = StdioServerParameters(
                             command='python',
                             args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/cdse_geocode/server.py"]),
                             timeout=300))]
def _fetch_agent(toolset) -> Optional[Agent]:
        if toolset:
            return Agent(
                model=llm,
                name='cdse_agent',
                description='A helpful assistant for answering user questions related to CDSE data.',
                instruction=_CDSE_SYSTEM_PROMPT,
                tools=toolset,
                mode='single_turn'
            )
        else:
            logger.warning("CDSE agent not initialized due to missing toolset.")
            return None

cdse_agent = _fetch_agent(toolset)