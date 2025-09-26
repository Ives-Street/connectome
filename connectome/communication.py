import os
import pandas as pd
import geopandas as gpd
import folium


def visualize_access_to_zone(geoms_with_dests,
                             ttm,
                             target_zone_id = None,
                             out_path = "communication/access_to_zone.html"):
    """Visualize access to a specific zone or to the zone with most destinations.
    The target zone will be highlighted in red on the map.

    Args:
        geoms_with_dests: GeoDataFrame containing geometries and their destinations
        ttm: Travel time matrix (wide, with from_id index and to_id columns) or a CSV path to it
        target_zone_id: Target zone ID. If None, selects zone with most destinations
    """
    if target_zone_id is None:
        #   right now only using lodes_jobs
        total_dests = geoms_with_dests['lodes_jobs']
        # Get the zone with maximum destinations
        target_zone_id = total_dests.idxmax()

    # Load wide TTM if a path is provided
    if isinstance(ttm, str):
        ttm = pd.read_csv(ttm, index_col=0)

    # Ensure ids are strings for alignment with TTM
    target_zone_id = str(target_zone_id)
    # Some GeoDataFrames have geom_id as index; ensure it's also a column
    if "geom_id" not in geoms_with_dests.columns:
        geoms_with_dests = geoms_with_dests.copy()
        geoms_with_dests["geom_id"] = geoms_with_dests.index.astype(str)
    else:
        geoms_with_dests = geoms_with_dests.copy()
        geoms_with_dests["geom_id"] = geoms_with_dests["geom_id"].astype(str)

    # Extract travel times to target zone (a Series indexed by from_id/geom_id)
    if target_zone_id not in ttm.columns:
        raise KeyError(f"Target zone id {target_zone_id} not found in travel time matrix columns.")
    travel_times_to_dest = ttm[target_zone_id].copy()
    travel_times_to_dest.index.name = 'geom_id'
    travel_times_to_dest.name = 'travel_time_to_dest'

    # Merge travel times into the GeoDataFrame
    to_visualize = geoms_with_dests.merge(
        travel_times_to_dest,
        left_on="geom_id",
        right_index=True,
        how="left"
    )

    # Keep only the necessary columns
    to_visualize = to_visualize[['geom_id', 'geometry', 'travel_time_to_dest']]

    def style_function(feature):
        # Highlight the target zone in blue
        if str(feature['properties']['geom_id']) == target_zone_id:
            return {'fillColor': 'blue', 'color': 'blue', 'weight': 2}
        return {}

    m = to_visualize.explore(
        column="travel_time_to_dest",
        cmap="YlOrBr",
        tiles="CartoDB.Positron",
        legend=True,
        legend_kwds={"caption": "Travel Time to Selected Destination (minutes)"},
        style_kwds={"style_function":style_function}
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    print(
        f"Saved visualization of access to zone {target_zone_id} to {out_path}"
    )

