import pandas as pd
import geopandas as gpd
import pygris.utils
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BUFFER_DISTANCE = 10000  # meters
DEFAULT_WATER_THRESHOLD = 0.1


def usa_tracts_from_address(
    states: List[str],
    address: str,
    buffer: float = DEFAULT_BUFFER_DISTANCE,
    save_to: Optional[str] = None,
    water_threshold: float = DEFAULT_WATER_THRESHOLD,
) -> gpd.GeoDataFrame:
    """
    Fetch and process US census tracts based on an address and buffer distance.
    
    This function retrieves census tracts from multiple states within a specified
    buffer distance from a given address, removes water bodies, and optionally
    saves the result to a file.
    
    Args:
        states: List of state abbreviations (e.g., ['VT', 'NY'])
        address: Address string to use as the center point
        buffer: Buffer distance in meters around the address (default: 10000)
        save_to: Optional file path to save the resulting GeoDataFrame (default: None)
        water_threshold: Threshold for water body removal (default: 0.1)
    
    Returns:
        GeoDataFrame: Processed census tracts with water bodies removed
        
    Example:
        >>> tracts = usa_tracts_from_address(
        ...     states=['VT', 'NY'],
        ...     address="26 University Pl, Burlington, VT 05405",
        ...     buffer=3000,
        ...     save_to="analysis_geometry.gpkg"
        ... )
    """
    logger.info(f"Fetching tracts for states: {states}")
    logger.info(f"Using address: {address} with buffer: {buffer}m")
    
    # Fetch tracts from each state
    union_tracts_list = []
    for state in states:
        try:
            tracts = pygris.tracts(
                cb=True,
                state=state,
                subset_by={address: buffer}
            )
            union_tracts_list.append(tracts)
            logger.info(f"Successfully fetched tracts for state: {state}")
        except Exception as e:
            logger.error(f"Failed to fetch tracts for state {state}: {e}")
            raise
    
    # Combine all tracts
    if not union_tracts_list:
        raise ValueError("No tracts were successfully fetched from any state")
    
    union_tracts = pd.concat(union_tracts_list, ignore_index=True) # Do you want an actual unioned shape here? Or just concated?
    logger.info(f"Combined {len(union_tracts)} tracts from {len(states)} states")
    
    # Remove water bodies
    original_count = len(union_tracts)
    union_tracts = pygris.utils.erase_water(union_tracts, area_threshold=water_threshold)
    final_count = len(union_tracts)
    logger.info(f"Removed water bodies: {original_count - final_count} tracts removed")
    logger.info(f"Final tract count: {final_count}")
    
    # Save to file if specified
    if save_to:
        try:
            union_tracts.to_file(save_to, driver='GPKG')
            logger.info(f"Successfully saved GeoDataFrame to {save_to}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to save file to {save_to}: {e}")
    
    return union_tracts

#TODO functionality for hexagons

