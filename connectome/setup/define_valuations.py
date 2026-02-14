import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#maybe eventually this will have some fancy means of dynamically generating functions for various pairs
#for now let's keep it simple

# Definition of valuation relationships between people and destinations (gravity model & exponent, dual/n-th access, etc) 
# by destination class and potentially by origin subgroup, also by time of day


#assume the average Place is visited 10x/day, the average job 1x/day, and the average person 0.05x/day
# what categories does Overture Places include? all retail, food, entertainment, etc?
VISIT_FREQ = {
    "overture_places":5,
    "lodes_jobs":1,
    "total_pop":0.01,
}

# read in overture places categories and assign overture json to appropriate category
# ^^ probably not necessary since json includes categories
# How many categories we thinking?

# TODO: move this to populate_destinations.py?
def generalize_destination_units(geoms_with_dests,
                                 visit_freqs=VISIT_FREQ,
                                 warn_missing=True):
    """Calculate total number of visits to each geometry based on destination types.
    
    Args:
        geoms_with_dests: GeoDataFrame containing geometries and their destinations
        visit_freqs: Dictionary mapping destination types to visit frequencies
        warn_missing: If True, warn when destination types are not found in columns
    
    Returns:
        GeoDataFrame with total visits per geometry
    
    Note:
        Destination types from visit_freqs that are not present in geoms_with_dests
        columns will be ignored with an optional warning.
    """
    general_dests = pd.Series(0, index=geoms_with_dests.index)

    for dest_type, freq in visit_freqs.items():
        if dest_type in geoms_with_dests.columns:
            general_dests += geoms_with_dests[dest_type] * freq
        elif warn_missing:
            logger.warning(f"destination type '{dest_type}' not found in input data")

    geoms_with_dests['general_destinations'] = general_dests
    return geoms_with_dests

