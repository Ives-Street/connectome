import geopandas as gpd
import osmnx as ox
import pandas as pd
import rasterstats
from census import Census
from pygris.data import get_lodes
from pandas import DataFrame
from tqdm import tqdm
from overturemaps import core
import PyQt6

def populate_destinations_overture_places(geographies: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Populate destinations for each geography.
    
    Args:
        geographies: GeoDataFrame in any CRS
        
    Returns:
        GeoDataFrame with overture_places column, in WGS84 (EPSG:4326)
    """
    # Ensure WGS84 for bbox and joining
    geographies_ll = geographies.to_crs(4326)
    bbox = list(geographies_ll.total_bounds)

    # Download Places from Overture Maps within the bounding box
    try:
        places_gdf = core.geodataframe("place", bbox)
    except FileNotFoundError:
        print("ERROR in Overture download: NOT adding overture destinations")
        # Return in WGS84
        if geographies.crs != "EPSG:4326":
            return geographies.to_crs("EPSG:4326")
        return geographies
    
    # Ensure CRS is set to WGS84 for joining
    places_gdf = places_gdf.set_crs(4326, allow_override=True)

    # Find number of Places within each input analysis area
    joined = gpd.sjoin(places_gdf, geographies_ll, predicate="within")
    counts = joined.groupby("index_right").size()
    geographies_ll["overture_places"] = counts.reindex(geographies_ll.index, fill_value=0).values

    return geographies_ll


def get_state_lodes(state: str) -> pd.DataFrame:
    return get_lodes(state, year=2022, lodes_type="wac", agg_level="tract")

def populate_destinations_LODES_jobs(geographies: gpd.GeoDataFrame,
                                     states: list[str],
                                     already_tracts=False) -> gpd.GeoDataFrame:
    """Populate LODES job counts for each geography.
    
    Args:
        geographies: GeoDataFrame in any CRS
        states: List of state abbreviations
        already_tracts: Whether geographies are already census tracts
        
    Returns:
        GeoDataFrame with lodes_jobs column, in WGS84 (EPSG:4326)
    """
    # Ensure WGS84 for consistency
    if geographies.crs != "EPSG:4326":
        geographies = geographies.to_crs("EPSG:4326")
    
    if already_tracts:
        # Get LODES data for all states and concatenate
        lodes_dfs = []
        for state in states:
            lodes_dfs.append(get_lodes(state, year=2022, lodes_type="wac", agg_level="tract"))
        lodes_wac = pd.concat(lodes_dfs)

        geographies['lodes_jobs'] = geographies['GEOID'].map(lodes_wac.set_index("w_geocode")["C000"])
        geographies['lodes_jobs_per_sqkm'] = geographies['lodes_jobs'] / (geographies['area_sqm'] / 1000000)

        return geographies

    else:  # calculate based on block groups then interpolate to analysis geometry
        # Get LODES data at block group level for all states and concatenate
        lodes_dfs = []
        for state in states:
            lodes_dfs.append(get_lodes(state, year=2022, lodes_type="wac", agg_level="bg"))
        lodes_wac = pd.concat(lodes_dfs)
        
        # Get block group geometries from pygris
        from pygris import block_groups
        
        # Fetch block group geometries for all states
        bg_gdfs = []
        for state in states:
            bg_gdf = block_groups(state=state, year=2022, cache=True)
            bg_gdfs.append(bg_gdf)
        bg_geoms = pd.concat(bg_gdfs)
        
        # Ensure WGS84
        if bg_geoms.crs != "EPSG:4326":
            bg_geoms = bg_geoms.to_crs("EPSG:4326")
        
        # Merge LODES data with block group geometries
        bg_geoms = bg_geoms.merge(lodes_wac, left_on="GEOID", right_on="w_geocode", how="inner")
        
        # Use projected CRS for area calculations
        target_crs = geographies.estimate_utm_crs()
            
        geographies_proj = geographies.to_crs(target_crs)
        bg_geoms_proj = bg_geoms.to_crs(target_crs)
        
        # Calculate total jobs before interpolation (for assertion)
        total_jobs_before = bg_geoms["C000"].sum()
        
        # Perform overlay to find intersections
        overlay = gpd.overlay(geographies_proj, bg_geoms_proj, how='intersection', keep_geom_type=False)
        
        # Calculate area of each intersection
        overlay['intersection_area'] = overlay.geometry.area
        
        # Calculate area of each original block group in the projected CRS
        bg_geoms_proj['bg_area'] = bg_geoms_proj.geometry.area
        bg_area_dict = bg_geoms_proj.set_index('GEOID')['bg_area'].to_dict()
        overlay['bg_area'] = overlay['GEOID_2'].map(bg_area_dict)
        
        # Calculate proportion of block group area that overlaps with each geography
        overlay['area_proportion'] = overlay['intersection_area'] / overlay['bg_area']
        
        # Calculate interpolated jobs
        overlay['interpolated_jobs'] = overlay['C000'] * overlay['area_proportion']
        
        # Sum interpolated jobs for each input geography
        jobs_by_geography = overlay.groupby(overlay.index)['interpolated_jobs'].sum()
        
        # Add the results back to the original geographies (in WGS84)
        geographies['lodes_jobs'] = jobs_by_geography.reindex(geographies.index, fill_value=0)
        geographies['lodes_jobs_per_sqkm'] = geographies['lodes_jobs'] / (geographies['area_sqm'] / 1000000)
        
        # Assert that total jobs are conserved (within a small tolerance for rounding errors)
        total_jobs_after = geographies['lodes_jobs'].sum()
        assert abs(total_jobs_before - total_jobs_after) < 1.0, \
            f"Jobs not conserved: before={total_jobs_before:.2f}, after={total_jobs_after:.2f}, diff={abs(total_jobs_before - total_jobs_after):.2f}"


        return geographies


def populate_all_dests_USA(geographies: gpd.GeoDataFrame,
                           states: str,
                           already_tracts = False,
                           save_to: str = "",
                            ) -> gpd.GeoDataFrame:
    """Populate all destination types for USA geographies.
    
    Args:
        geographies: GeoDataFrame in any CRS
        states: List of state abbreviations
        already_tracts: Whether geographies are already census tracts
        save_to: Optional path to save results
        
    Returns:
        GeoDataFrame with all destination columns, in WGS84 (EPSG:4326)
    """
    with_overture = populate_destinations_overture_places(geographies)
    with_overture_and_lodes = populate_destinations_LODES_jobs(with_overture, states, already_tracts)
    
    # Ensure WGS84 before saving
    if with_overture_and_lodes.crs != "EPSG:4326":
        with_overture_and_lodes = with_overture_and_lodes.to_crs("EPSG:4326")
    
    # Only save if a non-empty path is provided
    with_overture_and_lodes.index = with_overture_and_lodes['geom_id'].values
    if save_to:
        columns_to_drop = [col for col in with_overture_and_lodes.columns if col in ['level_0', 'index']]
        if columns_to_drop:
            with_overture_and_lodes = with_overture_and_lodes.drop(columns=columns_to_drop)
        with_overture_and_lodes.to_file(save_to)

    return with_overture_and_lodes