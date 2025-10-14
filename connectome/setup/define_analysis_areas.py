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

# Configure logging
logging.basicConfig(level=logging.INFO)
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


def _process_tracts(tracts: gpd.GeoDataFrame,
                    water_threshold = DEFAULT_WATER_THRESHOLD,
                    year = 2022,
                    save_to: str = "") -> gpd.GeoDataFrame:
    """Process tracts: add IDs, remove water, and save.
    
    Args:
        tracts: GeoDataFrame of census tracts (any CRS)
        save_to: Optional path to save the processed tracts
        
    Returns:
        GeoDataFrame of processed tracts in WGS84 (EPSG:4326)
    """
    tracts['geom_id'] = tracts.index.astype(str)
    
    # Remove water areas (need projected CRS for area calculation)
    if tracts.crs.is_geographic:
        tracts_proj = tracts.to_crs(tracts.estimate_utm_crs())
    else:
        tracts_proj = tracts.copy()

    tracts_proj['area_sqm'] = tracts_proj.geometry.area
    tracts_proj['area_km2'] = tracts_proj['area_sqm'] / 1_000_000
    mask = tracts_proj['area_km2'] >= DEFAULT_WATER_THRESHOLD
    tracts_filtered = tracts[mask].copy()
    tracts_filtered['area_sqm'] = tracts_proj[mask]['area_sqm']

    # Convert to WGS84
    if tracts_filtered.crs != "EPSG:4326":
        tracts_filtered = tracts_filtered.to_crs("EPSG:4326")
    
    if save_to:
        tracts_filtered.to_file(save_to)
    
    return tracts_filtered


# ============================================================================
# Public API Functions
# ============================================================================

def get_usa_tracts_from_address(
    states: list[str],
    address: str,
    buffer: float = DEFAULT_BUFFER_DISTANCE,
    year = 2022,
    save_to: str = ""
) -> gpd.GeoDataFrame:
    """Get census tracts within a buffer distance of an address.
    
    Args:
        states: List of state abbreviations
        address: Address to center the analysis area
        buffer: Buffer distance in meters
        save_to: Optional path to save the result

    Returns:
        GeoDataFrame: Processed census tracts with water bodies removed in WGS84 (EPSG:4326)
    """
    # Geocode address
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="connectome")
    location = geolocator.geocode(address)
    
    if location is None:
        raise ValueError(f"Could not geocode address: {address}")
    
    # Create point in WGS84 and buffer in projected CRS
    point = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([location.longitude], [location.latitude]),
        crs="EPSG:4326"
    )
    
    # Project to UTM for accurate buffering
    point_proj = point.to_crs(point.estimate_utm_crs())
    buffered_proj = point_proj.buffer(buffer)
    
    # Convert back to WGS84
    buffered_geom = gpd.GeoDataFrame(geometry=buffered_proj, crs=point_proj.crs)
    buffered_geom = buffered_geom.to_crs("EPSG:4326")
    
    # Get tracts and filter
    tracts = _fetch_tracts_for_states(states, year=year)
    filtered = _filter_tracts_by_geometry(tracts, buffered_geom)
    
    return _process_tracts(filtered, save_to)


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
    tracts = _process_tracts(tracts, water_threshold, year, save_to)

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
    tracts = _process_tracts(all_tracts, water_threshold, year, save_to)
    
    logger.info(f"Returned {len(tracts)} tracts for state(s): {states}")
    
    return tracts

# TODO: functionality for hexagons

