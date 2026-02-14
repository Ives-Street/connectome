import os
from typing import List, Optional, Union
import logging
import tempfile
import shutil

import pandas as pd
import geopandas as gpd
import pygris
import pygris.utils
import shapely

from constants import GEO_LEVELS, GEOID_LEN, WATER_THRESHOLD_PROPORTION

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BUFFER_DISTANCE = 10000  # meters
DEFAULT_WATER_THRESHOLD = 0.1


# ============================================================================
# Core Processing Functions (Internal)
# ============================================================================

def _fetch_tracts_for_states(
        states: List[str],
        year: int,
        subset_by: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """
    Fetch census tracts from multiple states with optional subsetting.

    Args:
        states: List of state abbreviations
        year: Census year
        subset_by: Optional dict for pygris subset_by parameter (e.g., {address: buffer})

    Returns:
        Combined GeoDataFrame of census tracts
    """
    logger.info(f"Fetching tracts for states: {states}")

    union_tracts_list = []
    for state in states:
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                logger.info(f"Fetching tracts for state: {state} (attempt {retry_count + 1}/{max_retries})")
                
                # Clear any temporary files that might be causing issues
                import tempfile
                import glob
                temp_dir = tempfile.gettempdir()
                old_files = glob.glob(os.path.join(temp_dir, f"cb_{year}_{state}_tract_*.zip"))
                for f in old_files:
                    try:
                        os.remove(f)
                        logger.debug(f"Removed old temp file: {f}")
                    except Exception:
                        pass
                
                # Fetch with cache disabled to avoid corruption issues
                tracts = pygris.tracts(
                    cb=True,
                    state=state,
                    year=year,
                    subset_by=subset_by,
                    cache=False
                )
                
                # Verify the data is valid
                if tracts is None or len(tracts) == 0:
                    raise ValueError(f"No tracts returned for state {state}")
                
                union_tracts_list.append(tracts)
                logger.info(f"Successfully fetched {len(tracts)} tracts for state: {state}")
                success = True
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count} failed for state {state}: {e}")
                
                if retry_count >= max_retries:
                    logger.error(f"Failed to fetch tracts for state {state} after {max_retries} attempts: {e}")
                    raise
                
                # Wait before retrying
                import time
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

    if not union_tracts_list:
        raise ValueError("No tracts were successfully fetched from any state")

    union_tracts = pd.concat(union_tracts_list, ignore_index=True)
    logger.info(f"Combined {len(union_tracts)} tracts from {len(states)} states")

    return union_tracts


def _fetch_block_groups_for_states(
        states: List[str],
        year: int,
) -> gpd.GeoDataFrame:
    """Fetch census block groups from multiple states.

    Returns:
        Combined GeoDataFrame of block groups with GEOID column.
    """
    logger.info(f"Fetching block groups for states: {states}")
    bg_list = []
    for state in states:
        bg = pygris.block_groups(state=state, cb=True, year=year, cache=False)
        bg_list.append(bg)
        logger.info(f"Fetched {len(bg)} block groups for state: {state}")

    if not bg_list:
        raise ValueError("No block groups fetched from any state")

    return pd.concat(bg_list, ignore_index=True)


def _fetch_blocks_for_states(
        states: List[str],
        year: int = 2020,
) -> gpd.GeoDataFrame:
    """Fetch census blocks from multiple states (decennial, includes POP20).

    Args:
        states: List of state abbreviations
        year: Decennial census year (default 2020)

    Returns:
        Combined GeoDataFrame of blocks with GEOID20 and POP20 columns.
    """
    logger.info(f"Fetching blocks for states: {states}")
    block_list = []
    for state in states:
        blocks = pygris.blocks(state=state, year=year, cache=False)
        block_list.append(blocks)
        logger.info(f"Fetched {len(blocks)} blocks for state: {state}")

    if not block_list:
        raise ValueError("No blocks fetched from any state")

    return pd.concat(block_list, ignore_index=True)


def resolve_center_geom(
        center_geom=None,
        address: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Convert various center_geom inputs to a unified GeoDataFrame in WGS84.

    Exactly one of *center_geom* or *address* must be provided.

    Args:
        center_geom: GeoDataFrame, Shapely geometry, or (lon, lat) tuple.
        address: Street address to geocode via Nominatim.

    Returns:
        GeoDataFrame with a single geometry in EPSG:4326.
    """
    if address is not None:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

        logger.info("Geocoding address for center geometry: %s", address)
        geolocator = Nominatim(user_agent="connectome", timeout=5)
        try:
            location = geolocator.geocode(address)
        except (GeocoderUnavailable, GeocoderTimedOut) as e:
            raise RuntimeError(
                f"Failed to geocode address '{address}': {e}. "
                "Pass center_geom=(lon, lat) to avoid external geocoding."
            ) from e
        if location is None:
            raise ValueError(f"Could not geocode address: {address}")

        point = shapely.Point(float(location.longitude), float(location.latitude))
        logger.info("Geocoded '%s' to lon=%s, lat=%s", address, location.longitude, location.latitude)
        return gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")

    if isinstance(center_geom, tuple) and len(center_geom) == 2:
        lon, lat = center_geom
        point = shapely.Point(float(lon), float(lat))
        return gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")

    # Delegate to existing helper for GeoDataFrame / Shapely / file path
    return _load_and_unify_polygon(center_geom, crs="EPSG:4326")


def _load_and_unify_polygon(
        polygon: Union[gpd.GeoDataFrame, gpd.GeoSeries, str, shapely.Geometry],
        crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load polygon from various input formats and unify into single geometry.

    Handles:
    - File path (str) -> loads with geopandas
    - GeoDataFrame -> uses as-is
    - GeoSeries -> converts to GeoDataFrame
    - shapely Geometry -> converts to GeoDataFrame

    All geometries are unified using union_all() into a single polygon.

    Args:
        polygon: Polygon input in various formats
        crs: CRS to set if polygon doesn't have one

    Returns:
        GeoDataFrame with a single unified geometry
    """
    # Load from file if string path
    if isinstance(polygon, str):
        if not os.path.exists(polygon):
            raise FileNotFoundError(f"Polygon file not found: {polygon}")
        gdf = gpd.read_file(polygon)
        logger.info(f"Loaded polygon from file: {polygon}")

    # Convert GeoSeries to GeoDataFrame
    elif isinstance(polygon, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=polygon)
        logger.info("Converted GeoSeries to GeoDataFrame")

    # Convert shapely geometry to GeoDataFrame
    elif isinstance(polygon, shapely.Geometry):
        gdf = gpd.GeoDataFrame(geometry=[polygon])
        logger.info("Converted shapely Geometry to GeoDataFrame")

    # Use GeoDataFrame as-is
    elif isinstance(polygon, gpd.GeoDataFrame):
        gdf = polygon.copy()
        logger.info("Using provided GeoDataFrame")

    else:
        raise TypeError(
            f"Unsupported polygon type: {type(polygon)}. "
            "Expected GeoDataFrame, GeoSeries, shapely Geometry, or file path (str)"
        )

    # Set CRS if provided and not already set
    if crs and gdf.crs is None:
        gdf = gdf.set_crs(crs)
        logger.info(f"Set CRS to: {crs}")

    # Ensure CRS is set
    if gdf.crs is None:
        raise ValueError(
            "Polygon must have a CRS. Either provide a polygon with CRS already set, "
            "or pass the 'crs' parameter."
        )

    # Unify all geometries into one
    unified_geom = gdf.union_all()
    unified_gdf = gpd.GeoDataFrame(geometry=[unified_geom], crs=gdf.crs)

    logger.info(f"Unified {len(gdf)} geometries into single polygon")

    return unified_gdf


def _filter_tracts_by_geometry(tracts: gpd.GeoDataFrame, filter_geom: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter tracts that intersect with the given geometry.
    
    Args:
        tracts: GeoDataFrame of census tracts (any CRS)
        filter_geom: GeoDataFrame to use as filter (any CRS)
        
    Returns:
        GeoDataFrame of filtered tracts in WGS84 (EPSG:4326)
    """
    # Ensure both are in the same CRS for intersection
    filter_geom = filter_geom.to_crs(tracts.crs)
    
    intersecting = tracts[tracts.intersects(filter_geom.union_all())]
    
    # Convert to WGS84 before returning
    if intersecting.crs != "EPSG:4326":
        intersecting = intersecting.to_crs("EPSG:4326")
    
    return intersecting


def _process_geographies(gdf: gpd.GeoDataFrame,
                         water_threshold=DEFAULT_WATER_THRESHOLD,
                         year=2022,
                         save_to: str = "") -> gpd.GeoDataFrame:
    """Process geographies: add sequential geom_ids, remove water, compute area, save.

    Works for tracts, block groups, or blocks.

    Args:
        gdf: GeoDataFrame of census geographies (any CRS)
        water_threshold: Minimum area in km² to keep
        year: Census year
        save_to: Optional path to save the result

    Returns:
        GeoDataFrame with geom_id and area_sqm, in WGS84 (EPSG:4326)
    """
    gdf = gdf.copy()
    gdf['geom_id'] = gdf.index.astype(str)

    # Remove water areas (need projected CRS for area calculation)
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf.copy()

    gdf_proj['area_sqm'] = gdf_proj.geometry.area
    gdf_proj['area_km2'] = gdf_proj['area_sqm'] / 1_000_000
    mask = gdf_proj['area_km2'] >= water_threshold
    filtered = gdf[mask].copy()
    filtered['area_sqm'] = gdf_proj[mask]['area_sqm']

    # Convert to WGS84
    if filtered.crs != "EPSG:4326":
        filtered = filtered.to_crs("EPSG:4326")

    if save_to:
        filtered.to_file(save_to)
        logger.info(f"Saved geographies to {save_to}")

    return filtered


# ============================================================================
# Public API Functions
# ============================================================================

def get_usa_tracts_from_location(
    states: list[str],
    center_geom,
    buffer: float = DEFAULT_BUFFER_DISTANCE,
    year: int = 2022,
    save_to: str = "",
) -> gpd.GeoDataFrame:
    """Get census tracts within a buffer distance of a center geometry.

    Args:
        states: List of state abbreviations
        center_geom: Center of the analysis area. Accepts a GeoDataFrame,
            Shapely geometry, or ``(lon, lat)`` tuple.
        buffer: Buffer distance in meters
        year: Census year for tract boundaries
        save_to: Optional path to save the result

    Returns:
        GeoDataFrame: Processed census tracts with water bodies removed in
        WGS84 (EPSG:4326)
    """
    center_gdf = resolve_center_geom(center_geom)

    # Project to UTM for accurate buffering
    center_proj = center_gdf.to_crs(center_gdf.estimate_utm_crs())
    buffered_proj = center_proj.buffer(buffer)

    # Convert back to WGS84
    buffered_geom = gpd.GeoDataFrame(geometry=buffered_proj, crs=center_proj.crs)
    buffered_geom = buffered_geom.to_crs("EPSG:4326")

    # Get tracts and filter
    tracts = _fetch_tracts_for_states(states, year=year)
    filtered = _filter_tracts_by_geometry(tracts, buffered_geom)

    return _process_geographies(
        filtered,
        year=year,
        save_to=save_to,
    )


def get_usa_tracts_from_polygon(
        states: List[str],
        polygon: Union[gpd.GeoDataFrame, gpd.GeoSeries, str, shapely.Geometry],
        crs: Optional[str] = None,
        save_to: Optional[str] = None,
        water_threshold: float = DEFAULT_WATER_THRESHOLD,
        year = 2022,
) -> gpd.GeoDataFrame:
    """Get census tracts that intersect with a polygon.
    
    Args:
        states: List of state abbreviations
        polygon_path: Path to polygon file, GeoDataFrame, GeoSeries, or Shapely geometry
        save_to: Optional path to save the result
        
    Returns:
        GeoDataFrame of processed census tracts in WGS84 (EPSG:4326)

    """
    logger.info("Fetching tracts based on polygon input")

    # Load and unify polygon
    unified_polygon = _load_and_unify_polygon(polygon, crs)

    # Fetch all tracts for the states (no subsetting yet)
    all_tracts = _fetch_tracts_for_states(states, year, subset_by=None)

    # Filter by polygon intersection
    tracts = _filter_tracts_by_geometry(all_tracts, unified_polygon)

    # Process and return
    tracts = _process_geographies(tracts, water_threshold, year, save_to=save_to)

    return tracts


def get_usa_tracts_from_state(
        states: Union[str, List[str]],
        save_to: Optional[str] = None,
        water_threshold: float = DEFAULT_WATER_THRESHOLD,
        year: int = 2022,
) -> gpd.GeoDataFrame:
    """Get all census tracts for entire state(s).
    
    Args:
        states: Single state abbreviation or list of state abbreviations (e.g., 'RI' or ['RI', 'MA'])
        save_to: Optional path to save the result
        water_threshold: Minimum area in km² to keep (filters out small water-only tracts)
        year: Census year for tract boundaries
        
    Returns:
        GeoDataFrame of all census tracts in the state(s), in WGS84 (EPSG:4326)
        
    Examples:
        >>> # Single state
        >>> ri_tracts = get_usa_tracts_from_state(
        ...     states='RI',
        ...     save_to='rhode_island_tracts.gpkg'
        ... )
        
        >>> # Multiple states
        >>> ne_tracts = get_usa_tracts_from_state(
        ...     states=['RI', 'MA', 'CT'],
        ...     save_to='new_england_tracts.gpkg'
        ... )
    """
    logger.info(f"Fetching all tracts for state(s): {states}")
    
    # Handle single state as string
    if isinstance(states, str):
        states = [states]
    
    # Fetch all tracts for the state(s) - no subsetting
    all_tracts = _fetch_tracts_for_states(states, year, subset_by=None)
    
    # Process and return (adds geom_id, removes water, converts to WGS84)
    tracts = _process_geographies(all_tracts, water_threshold, year, save_to=save_to)
    
    logger.info(f"Returned {len(tracts)} tracts for state(s): {states}")

    return tracts


def get_multi_resolution_geometries(
    states: List[str],
    center_geom,
    block_buffer: Optional[float] = None,
    bg_buffer: Optional[float] = None,
    tract_buffer: float = 30000,
    year: int = 2022,
    save_to: str = "",
) -> gpd.GeoDataFrame:
    """Build a multi-resolution set of analysis areas using concentric buffers.

    Tracts whose centroid falls within ``bg_buffer`` are replaced by their
    constituent block groups.  Block groups whose centroid falls within
    ``block_buffer`` are further replaced by constituent blocks.  The result
    is a single GeoDataFrame with no gaps (child geographies tile perfectly
    within parents).

    Args:
        states: State abbreviations to fetch geographies for.
        center_geom: Center of the analysis area (GeoDataFrame, Shapely
            geometry, or ``(lon, lat)`` tuple).
        block_buffer: Distance in meters for the innermost (block) ring.
            ``None`` means no block-level detail.
        bg_buffer: Distance in meters for the middle (block group) ring.
            ``None`` means no block-group-level detail.
        tract_buffer: Distance in meters for the outermost (tract) ring.
        year: Census year for tract / block-group boundaries.
        save_to: Optional path to save the result.

    Returns:
        GeoDataFrame in EPSG:4326 with columns ``geom_id`` (str),
        ``geo_level`` (``'tract'``, ``'block_group'``, or ``'block'``),
        ``parent_tract_GEOID``, and ``area_sqm``.
    """
    # --- resolve center geometry and build buffer rings --------------------
    center_gdf = resolve_center_geom(center_geom)
    utm_crs = center_gdf.estimate_utm_crs()
    center_proj = center_gdf.to_crs(utm_crs)
    center_union = center_proj.union_all()

    def _buffer_to_wgs84(dist):
        ring = gpd.GeoDataFrame(geometry=[center_union.buffer(dist)], crs=utm_crs)
        return ring.to_crs("EPSG:4326")

    tract_ring = _buffer_to_wgs84(tract_buffer)
    bg_ring = _buffer_to_wgs84(bg_buffer) if bg_buffer is not None else None
    block_ring = _buffer_to_wgs84(block_buffer) if block_buffer is not None else None

    # --- fetch tracts and classify ----------------------------------------
    all_tracts = _fetch_tracts_for_states(states, year=year)
    tracts = _filter_tracts_by_geometry(all_tracts, tract_ring)

    # Project centroids to UTM for distance classification
    tracts_proj = tracts.to_crs(utm_crs)
    tracts['centroid_dist'] = tracts_proj.centroid.distance(center_union)

    # Determine which tracts get exploded to BGs
    if bg_buffer is not None:
        near_bg_mask = tracts['centroid_dist'] <= bg_buffer
    else:
        near_bg_mask = pd.Series(False, index=tracts.index)

    far_tracts = tracts[~near_bg_mask].copy()
    near_tract_geoids = set(tracts.loc[near_bg_mask, 'GEOID'].values)

    # --- fetch block groups for near tracts --------------------------------
    remaining_bgs = gpd.GeoDataFrame()
    near_bg_geoids = set()

    # Always fetch blocks for near tracts so we can compute POP20 for BGs
    all_blocks = gpd.GeoDataFrame()
    if near_tract_geoids:
        all_bgs = _fetch_block_groups_for_states(states, year=year)
        # Filter to BGs whose parent tract is in the near set (GEOID[:11])
        all_bgs['parent_tract_GEOID'] = all_bgs['GEOID'].str[:GEOID_LEN['tract']]
        all_bgs = all_bgs[all_bgs['parent_tract_GEOID'].isin(near_tract_geoids)].copy()

        # Fetch blocks to get POP20 (needed for proration of BGs and blocks)
        all_blocks = _fetch_blocks_for_states(states, year=2020)
        if 'GEOID20' in all_blocks.columns and 'GEOID' not in all_blocks.columns:
            all_blocks = all_blocks.rename(columns={'GEOID20': 'GEOID'})
        if 'POP20' not in all_blocks.columns:
            all_blocks['POP20'] = 0
        all_blocks['POP20'] = pd.to_numeric(all_blocks['POP20'], errors='coerce').fillna(0)
        all_blocks['parent_bg_GEOID'] = all_blocks['GEOID'].str[:GEOID_LEN['block_group']]
        all_blocks['parent_tract_GEOID'] = all_blocks['GEOID'].str[:GEOID_LEN['tract']]
        # Keep only blocks within near tracts
        all_blocks = all_blocks[all_blocks['parent_tract_GEOID'].isin(near_tract_geoids)].copy()

        # Compute POP20 for each BG by summing its constituent blocks
        bg_pop = all_blocks.groupby('parent_bg_GEOID')['POP20'].sum()
        all_bgs['POP20'] = all_bgs['GEOID'].map(bg_pop).fillna(0)

        if block_buffer is not None and len(all_bgs) > 0:
            bgs_proj = all_bgs.to_crs(utm_crs)
            all_bgs['centroid_dist'] = bgs_proj.centroid.distance(center_union)
            near_block_mask = all_bgs['centroid_dist'] <= block_buffer
            near_bg_geoids = set(all_bgs.loc[near_block_mask, 'GEOID'].values)
            remaining_bgs = all_bgs[~near_block_mask].copy()
        else:
            remaining_bgs = all_bgs.copy()

    # --- filter blocks for near BGs (only included as geometries when block_buffer is set)
    blocks = gpd.GeoDataFrame()
    if near_bg_geoids and len(all_blocks) > 0:
        blocks = all_blocks[all_blocks['parent_bg_GEOID'].isin(near_bg_geoids)].copy()

    # --- tag geo_level and parent_tract_GEOID -----------------------------
    far_tracts['geo_level'] = 'tract'
    far_tracts['parent_tract_GEOID'] = far_tracts['GEOID']

    if len(remaining_bgs) > 0:
        remaining_bgs['geo_level'] = 'block_group'
        # parent_tract_GEOID already set above

    if len(blocks) > 0:
        blocks['geo_level'] = 'block'
        # parent_tract_GEOID already set above

    # --- normalize CRS and combine ----------------------------------------
    pieces = []
    for piece in [far_tracts, remaining_bgs, blocks]:
        if len(piece) > 0:
            if piece.crs is not None and piece.crs != "EPSG:4326":
                piece = piece.to_crs("EPSG:4326")
            pieces.append(piece)

    if not pieces:
        raise ValueError("No geographies found within the specified buffers")

    combined = pd.concat(pieces, ignore_index=True)

    # Remove water / tiny areas
    if combined.crs is None:
        combined = combined.set_crs("EPSG:4326")
    if combined.crs.is_geographic:
        comb_proj = combined.to_crs(combined.estimate_utm_crs())
    else:
        comb_proj = combined.copy()
    combined['area_sqm'] = comb_proj.geometry.area
    median_area = combined.groupby('geo_level')['area_sqm'].transform('median')
    combined = combined[combined['area_sqm'] >= WATER_THRESHOLD_PROPORTION * median_area].copy()

    # Sequential string geom_id
    combined = combined.reset_index(drop=True)
    combined['geom_id'] = combined.index.astype(str)

    # Ensure WGS84
    if combined.crs != "EPSG:4326":
        combined = combined.to_crs("EPSG:4326")

    logger.info(
        "Multi-resolution geometries: %d tracts, %d block groups, %d blocks (%d total)",
        (combined['geo_level'] == 'tract').sum(),
        (combined['geo_level'] == 'block_group').sum(),
        (combined['geo_level'] == 'block').sum(),
        len(combined),
    )

    if save_to:
        combined.to_file(save_to)
        logger.info(f"Saved multi-resolution geometries to {save_to}")

    return combined


# TODO: functionality for hexagons

