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
        ttm: Travel time matrix
        scenario_dir: Directory for scenario data
        target_zone_id: Target zone ID. If None, selects zone with most destinations
    """
    if target_zone_id is None:
        # Sum up all destination columns (overture_places and lodes_jobs)
        total_dests = geoms_with_dests['overture_places'] + geoms_with_dests['lodes_jobs']
        # Get the zone with maximum destinations
        target_zone_id = total_dests.idxmax()

    if type(ttm) == str:
        ttm = pd.read_csv(ttm)


    # Extract travel times to target zone
    travel_times_to_dest = ttm[target_zone_id]
    travel_times_to_dest.index.name = 'geom_id'

    to_visualize = geoms_with_dests.join(travel_times_to_dest, on='geom_id')
    to_visualize.rename(columns={target_zone_id: 'travel_time_to_dest'}, inplace=True)

    to_visualize = pd.merge(geoms_with_dests, to_visualize, on='geom_id')

    to_visualize = to_visualize[['geom_id', 'geometry', 'travel_time_to_dest']]

    def style_function(feature):
        if feature['properties']['geom_id'] == target_zone_id:
            return {'fillColor': 'blue', 'color': 'blue', 'weight': 2}
        return {}

    m = to_visualize.explore(
        column="travel_time_to_dest",
        cmap="YlOrBr",
        tiles="CartoDB.Positron",
        legend=True,
        legend_kwds={"caption": "Travel Time to Selected Destination (minutes)"},
        style_kwds={'style_function':style_function}
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    print(
        f"Saved visualization of access to zone {target_zone_id} to {out_path}"
    )

