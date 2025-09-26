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
    '''
    populate destinations for each tract
    '''
    # Get the bounding box of the geographies
    bbox = list(geographies.total_bounds)  # [minx, miny, maxx, maxy]

    # Download Places from Overture Maps within the bounding box
    # Try default release first; fall back to "latest" if that path doesn't exist
    try:
        places_gdf = core.geodataframe("place", bbox)
    except FileNotFoundError:
        print("ERROR in Overture download: NOT adding overture destinations")
        return geographies
    # Ensure CRS is set to WGS84 for joining
    places_gdf = places_gdf.set_crs(4326, allow_override=True)

    #find number of Places within each input analysis area
    geographies_ll = geographies.to_crs(4326)
    joined = gpd.sjoin(places_gdf, geographies_ll, predicate="within")
    counts = joined.groupby("index_right").size()
    geographies["overture_places"] = counts.reindex(geographies_ll.index, fill_value=0).values

    #TODO include building size?

    return geographies

def populate_destinations_LODES_jobs(geographies: gpd.GeoDataFrame,
                                       state: str, #todo enable list-of-states as in populate_people_usa
                                       already_tracts = False) -> gpd.GeoDataFrame:
    if already_tracts: #our input analysis areas are already census tracts
        #use pygris to download
        lodes_wac = get_lodes(state, year=2022, lodes_type="wac", agg_level="tract")
        geographies['lodes_jobs'] = geographies['GEOID'].map(lodes_wac.set_index("w_geocode")["C000"])
        return geographies
    else: #calculate based on blocks then aggregate up to analysis geometry TODO: TEST THIS!
        raise RuntimeError("not yet implemented")
        # geographies_ll = geographies.to_crs(4326)
        # lodes_wac = get_lodes(state, year=2022, lodes_type="wac", agg_level="block", return_lonlat=True)
        # lodes_wac = gpd.GeoDataFrame(
        #     lodes_wac,
        #     geometry=gpd.points_from_xy(Y.w_lon, Y.w_lat),
        #     crs=4326
        # )
        # #count sum of all C000 jobs within each geography poly
        # #todo this could be more rigorous - it's possible to slip through the cracks,
        # # like if a block centroid is in a body of water
        # joined = gpd.sjoin(lodes_wac, geographies_ll, how="inner", predicate="within")
        # sums = joined.groupby(joined.index_right)["C000"].sum()
        # geographies_ll["C000_sum"] = geographies_ll.index.map(sums).fillna(0)
        # geographies_ll["C000_sum"] = geographies_ll.index.map(sums).fillna(0)
        # return geographies_ll.to_crs(geographies.crs)


def populate_all_dests_USA(geographies: gpd.GeoDataFrame,
                           state: str,
                           already_tracts = False,
                           save_to: str = "",  #'existing_conditions/geographies_with_dests.csv'
                            ) -> gpd.GeoDataFrame:
    with_overture = populate_destinations_overture_places(geographies)
    with_lodes = populate_destinations_LODES_jobs(with_overture, state, already_tracts)
    # Only save if a non-empty path is provided
    if save_to:
        with_lodes.to_file(save_to)

    return with_lodes