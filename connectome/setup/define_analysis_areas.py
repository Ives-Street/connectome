import pandas as pd
import geopandas as gpd
import pygris.utils


def usa_tracts_from_address(states: list,
                        address: str,
                        buffer: float = 10000,
                        save_to: str = "",  #'existing_conditions/analysis_geometry.gpkg'
                        ) -> gpd.GeoDataFrame:
    union_tracts_list = [pygris.tracts(
        cb = True,
        state = x,
        subset_by = {address: buffer})
        for x in states]
    union_tracts = pd.concat(union_tracts_list)
    union_tracts = pygris.utils.erase_water(union_tracts, area_threshold=0.1)
    if not save_to == "":
        union_tracts.to_file(save_to, driver = 'GPKG')
    return union_tracts

