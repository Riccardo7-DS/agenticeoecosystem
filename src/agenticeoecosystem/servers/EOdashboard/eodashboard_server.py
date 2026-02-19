import io
import re

import pyarrow.parquet as pq
import pystac
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EODashboard MCP Server")

# Configuration
NARRATIVES_URL = (
    "https://esa-eodashboards.github.io/eodashboard-narratives/narratives.json"
)
STAC_CATALOG_URL = (
    "https://esa-eodashboards.github.io/eodashboard-catalog/trilateral/catalog.json"
)


# Cache for narratives and STAC catalog data
class EODashboardCache:
    """Cache for narratives and STAC catalog data"""

    def __init__(self):
        self.narratives_index = None
        self.narrative_to_indicators = None
        self.code_to_collection = None
        self.root_catalog = None

    def load_narratives(self):
        """Load and cache narratives index with indicators"""
        if self.narratives_index is not None:
            return

        response = requests.get(NARRATIVES_URL)
        response.raise_for_status()
        self.narratives_index = response.json()

        # Build narrative → indicators mapping
        self.narrative_to_indicators = {}
        for narrative in self.narratives_index:
            try:
                content = self._fetch_narrative(narrative["file"])
                indicators = self._extract_indicator_codes(content)

                self.narrative_to_indicators[narrative.get("title", "Untitled")] = {
                    "indicators": indicators,
                    "theme": narrative.get("theme", "unknown"),
                    "tags": narrative.get("tags", ""),
                    "url": narrative["file"],
                    "date": narrative.get("date", ""),
                    "official": narrative.get("official", False),
                }
            except Exception:
                continue

    def load_catalog(self):
        """Load and cache STAC catalog"""
        if self.root_catalog is not None:
            return

        self.root_catalog = pystac.Catalog.from_file(STAC_CATALOG_URL)

        # Build code → collection index
        self.code_to_collection = {}

        # Iterate through catalog's child links
        for link in self.root_catalog.links:
            if link.rel == "child":
                code = link.extra_fields.get("code")

                if code:
                    self.code_to_collection[code] = {
                        "code": code,
                        "title": link.extra_fields.get("title", code),
                        "themes": link.extra_fields.get("themes", []),
                        "tags": link.extra_fields.get("tags", []),
                        "href": link.href,
                        "full_url": requests.compat.urljoin(
                            STAC_CATALOG_URL, link.href
                        ),
                        "self_href": requests.compat.urljoin(
                            STAC_CATALOG_URL, link.href
                        ),
                        "satellite": link.extra_fields.get("satellite"),
                        "sensor": link.extra_fields.get("sensor"),
                        "description": link.extra_fields.get("description", ""),
                    }

    @staticmethod
    def _fetch_narrative(url: str) -> str:
        """Fetch narrative markdown content"""
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _extract_indicator_codes(content: str) -> list:
        """Extract indicator codes from narrative content"""
        pattern = r"indicator=([A-Za-z0-9_-]+)"
        matches = re.findall(pattern, content)
        return list(set(matches))


# Initialize cache
cache = EODashboardCache()


@mcp.tool()
def list_narratives() -> str:
    """Get a list of all available EODashboard narratives"""
    cache.load_narratives()
    narratives = cache.narratives_index

    # Format the response
    result = f"Found {len(narratives)} narratives:\n\n"
    for narr in narratives[:10]:  # Show first 10
        result += f"- {narr.get('title', 'Untitled')}\n"
        result += f"  Theme: {narr.get('theme', 'unknown')}\n"
        result += f"  URL: {narr.get('file', '')}\n\n"

    if len(narratives) > 10:
        result += f"... and {len(narratives) - 10} more"

    return result


@mcp.tool()
def search_narratives(keyword: str = "", theme: str = "") -> str:
    """Search narratives by keyword in title or by theme (e.g., 'oceans', 'ice', 'atmosphere')"""
    cache.load_narratives()
    narratives = cache.narratives_index

    # Filter narratives
    filtered = []
    for narr in narratives:
        title = narr.get("title", "").lower()
        narr_theme = narr.get("theme", "").lower()

        if keyword and keyword.lower() not in title:
            continue
        if theme and theme.lower() not in narr_theme:
            continue

        filtered.append(narr)

    if not filtered:
        return f"No narratives found matching keyword='{keyword}' theme='{theme}'"

    # Format results with indicators
    result = f"Found {len(filtered)} matching narrative(s):\n\n"
    for narr in filtered[:15]:
        title = narr.get("title", "Untitled")
        indicators = cache.narrative_to_indicators.get(title, {}).get("indicators", [])

        result += f"- {title}\n"
        result += f"  Theme: {narr.get('theme', 'unknown')}\n"
        result += f"  Tags: {narr.get('tags', 'none')}\n"
        result += f"  Date: {narr.get('date', 'unknown')}\n"
        result += f"  Indicators: {', '.join(indicators) if indicators else 'None'}\n"
        result += f"  URL: {narr.get('file', '')}\n\n"

    if len(filtered) > 15:
        result += f"... and {len(filtered) - 15} more"

    return result


@mcp.tool()
def get_narrative_content(url: str) -> str:
    """Get the full content of a specific narrative by its URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text

        # Parse frontmatter and body
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                body = parts[2].strip()

                # Limit content length
                if len(body) > 4000:
                    body = (
                        body[:4000]
                        + "\n\n... (content truncated, showing first 4000 chars)"
                    )

                result = f"FRONTMATTER:\n{frontmatter}\n\n"
                result += f"CONTENT:\n{body}"
                return result

        # Limit content length for context
        if len(content) > 5000:
            content = (
                content[:5000] + "\n\n... (content truncated, showing first 5000 chars)"
            )

        return content
    except Exception as e:
        return f"Error fetching narrative: {str(e)}"


@mcp.tool()
def list_themes() -> str:
    """Get all available themes from narratives"""
    cache.load_narratives()
    narratives = cache.narratives_index

    # Collect unique themes
    themes = set()
    for narr in narratives:
        theme = narr.get("theme", "")
        if theme:
            themes.add(theme)

    result = f"Available themes ({len(themes)}):\n\n"
    for theme in sorted(themes):
        # Count narratives for this theme
        count = sum(1 for n in narratives if n.get("theme") == theme)
        result += f"- {theme} ({count} narratives)\n"

    return result


@mcp.tool()
def get_narrative_indicators(narrative_title: str) -> str:
    """Get the collection indicator codes referenced in a narrative. Returns the list of STAC collection codes that provide data for this story."""
    cache.load_narratives()
    cache.load_catalog()

    narrative_info = cache.narrative_to_indicators.get(narrative_title)

    if not narrative_info:
        # Try fuzzy match
        for nar_title in cache.narrative_to_indicators.keys():
            if narrative_title.lower() in nar_title.lower():
                narrative_info = cache.narrative_to_indicators[nar_title]
                narrative_title = nar_title
                break

    if not narrative_info:
        return f"Narrative '{narrative_title}' not found. Use search_narratives to find exact titles."

    indicators = narrative_info["indicators"]

    if not indicators:
        return f"Narrative '{narrative_title}' does not reference any collection indicators."

    # Get collection details for each indicator
    response = (
        f"Narrative '{narrative_title}' references {len(indicators)} collection(s):\n\n"
    )

    for code in indicators:
        collection = cache.code_to_collection.get(code)
        if collection:
            response += f"   {code}\n"
            response += f"   Title: {collection['title']}\n"
            response += f"   Themes: {', '.join(collection['themes'])}\n"
            if collection.get("satellite"):
                response += f"   Satellite: {collection['satellite']}\n"
            response += f"   URL: {collection['full_url']}\n\n"
        else:
            response += f"    {code} - Not found in catalog\n\n"

    return response


@mcp.tool()
def search_collections(code: str = "", theme: str = "", keyword: str = "") -> str:
    """Search STAC collections (datasets) in the EODashboard catalog. Filter by collection code, theme, or keyword in titles/descriptions."""
    cache.load_catalog()

    code = code.upper()
    theme = theme.lower()
    keyword = keyword.lower()

    results = []

    for coll_code, collection in cache.code_to_collection.items():
        # Apply filters
        if code and code != coll_code:
            continue

        if theme:
            coll_themes = [t.lower() for t in collection.get("themes", [])]
            if theme not in coll_themes:
                continue

        if keyword:
            searchable = (
                f"{collection['title']} {collection.get('description', '')}".lower()
            )
            if keyword not in searchable:
                continue

        results.append(collection)

    if not results:
        return "No collections found matching the criteria."

    # Format response
    response = f"Found {len(results)} collection(s):\n\n"
    for coll in results[:15]:  # Limit to 15
        response += f"   {coll['code']}\n"
        response += f"   Title: {coll['title']}\n"
        response += f"   Themes: {', '.join(coll['themes'])}\n"
        if coll.get("satellite"):
            response += f"   Satellite: {coll['satellite']}\n"
        if coll.get("sensor"):
            response += f"   Sensor: {coll['sensor']}\n"
        response += f"   URL: {coll['self_href']}\n\n"

    if len(results) > 15:
        response += f"... and {len(results) - 15} more results (showing first 15)"

    return response


@mcp.tool()
def get_collection_items(collection_code: str, limit: int = 10) -> str:
    """Get data items from a specific STAC collection. Items represent individual observations or time slices. Returns temporal data points with properties and geometries."""
    cache.load_catalog()

    # Try exact match first, then uppercase
    collection = cache.code_to_collection.get(collection_code)
    if not collection:
        collection = cache.code_to_collection.get(collection_code.upper())

    if not collection:
        return f"Collection '{collection_code}' not found. Use search_collections to find valid codes."

    try:
        # Load the collection from STAC
        pystac_collection = pystac.Collection.from_file(collection["self_href"])

        # Check for parquet asset
        geoparquet_link = None

        for link in pystac_collection.links:
            if link.rel == "geoparquet":
                geoparquet_link = link.target
                break

        if geoparquet_link:
            # Resolve relative URL
            if not geoparquet_link.startswith("http"):
                base_url = collection["self_href"].rsplit("/", 1)[0]
                geoparquet_link = f"{base_url}/{geoparquet_link}"

            response = requests.get(geoparquet_link)
            response.raise_for_status()

            table = pq.read_table(io.BytesIO(response.content))
            df = table.to_pandas()

            # Convert to items format
            response_text = (
                f"Collection '{collection['code']}' - {collection['title']}\n"
            )
            response_text += (
                f"Found {len(df)} item(s) (showing first {min(limit, len(df))}):\n\n"
            )

            for idx, row in df.head(limit).iterrows():
                response_text += f"Item {idx + 1}:\n"
                response_text += f"  Properties: {dict(row)}\n\n"

            return response_text

        else:
            return f"Collection '{collection['code']}' does not have a geoparquet asset. Items may be in a different format."

    except Exception as e:
        return f"Error retrieving items for collection '{collection['code']}': {str(e)}"


@mcp.tool()
def find_narratives_for_indicator(indicator_code: str) -> str:
    """Reverse lookup: find all narratives that reference a specific collection code. Useful to discover stories related to a particular dataset."""
    cache.load_narratives()

    code = indicator_code.upper()

    # Find matching narratives
    matching = []

    for title, info in cache.narrative_to_indicators.items():
        if code in [ind.upper() for ind in info["indicators"]]:
            matching.append(
                {
                    "title": title,
                    "theme": info["theme"],
                    "tags": info["tags"],
                    "url": info["url"],
                }
            )

    if not matching:
        return f"No narratives found referencing indicator '{code}'."

    # Format response
    response = f"Found {len(matching)} narrative(s) referencing '{code}':\n\n"
    for narr in matching:
        response += f"   {narr['title']}\n"
        response += f"   Theme: {narr['theme']} | Tags: {narr['tags']}\n"
        response += f"   URL: {narr['url']}\n\n"

    return response


if __name__ == "__main__":
    mcp.run()
