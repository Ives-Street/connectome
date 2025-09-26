import os
import json

import pandas as pd
import r5py
import datetime
import geopandas as gpd
from tqdm import tqdm

import communication


MODES = [ #todo - make this universal for the whole codebase
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
]

def route_for_all_envs(routing_dir: str,
                       geoms_with_dests, #maybe this shouldn't require dests? Only used for communication visualization
                       user_classes: pd.DataFrame,
                       departure_time = datetime.datetime(2025,9,23,9,0,0),
                       visualize_all = True,
                       ):
    rep_points = gpd.GeoDataFrame(crs=geoms_with_dests.crs, geometry=geoms_with_dests.representative_point())
    rep_points['id'] = geoms_with_dests['geom_id']
    rep_points.to_file(f"{routing_dir}/DEBUG: rep_points.geojson", driver="GeoJSON")
    for mode in MODES:
        print(f"identifying routeenvs for {mode}")
        route_envs_for_mode = set()
        route_envs_for_mode.update(user_classes[f'routeenv_{mode}'].unique())
        for routeenv in route_envs_for_mode:
            print(f"routing {mode} for {routeenv}")
            gtfs_files = os.listdir(f"{routing_dir}/{routeenv}/gtfs_files")
            gtfs_fullpaths = [f"{routing_dir}/{routeenv}/gtfs_files/{filename}" for filename in gtfs_files]

            # load r5 params
            with open(f"{routing_dir}/{routeenv}/r5_params.json") as f:
                r5_params = json.load(f)

            # create network
            network = r5py.TransportNetwork(
                osm_pbf=f"{routing_dir}/{routeenv}/osm_file.pbf",
                gtfs=gtfs_fullpaths,
            )

            # route on network to create travel time matrix
            ttm = r5py.TravelTimeMatrix(network,
                                        rep_points,
                                        rep_points,
                                        transport_modes=[mode],
                                        departure=departure_time,
                                        **r5_params
                                        )
            ttm_df = pd.DataFrame(ttm) #shouldn't be necessary, but I was getting a weird r5py error

            # Ensure ids are strings for consistent indexing/column matching
            ttm_df["from_id"] = ttm_df["from_id"].astype(str)
            ttm_df["to_id"] = ttm_df["to_id"].astype(str)

            # Pivot to wide format: rows=from_id, columns=to_id
            ttm_df_wide = ttm_df.pivot(index="from_id", columns="to_id", values="travel_time")
            ttm_df_wide.to_csv(f"{routing_dir}/{routeenv}/ttm_{mode}.csv")
            if visualize_all:
                communication.visualize_access_to_zone(geoms_with_dests,
                                                       ttm_df_wide,
                                                       target_zone_id = None,
                                                       out_path=f"{routing_dir}/{routeenv}/route_{mode}_viz.html")

