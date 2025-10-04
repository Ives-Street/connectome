import os
import json
import zipfile
import shutil
import tempfile

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

DEPARTURE_TIME = datetime.datetime(2025,9,23,9,0,0)

def route_environment(routeenv, mode, routing_dir, rep_points, departure_time):
    print(f"routing {mode} for {routeenv}")
    gtfs_files = os.listdir(f"{routing_dir}/{routeenv}/gtfs_files")
    gtfs_fullpaths = [f"{routing_dir}/{routeenv}/gtfs_files/{filename}" for filename in gtfs_files]

    # load r5 params
    with open(f"{routing_dir}/{routeenv}/r5_params.json") as f:
        r5_params = json.load(f)

    # create network
    print("creating network")
    try:
        network = r5py.TransportNetwork(
            osm_pbf=f"{routing_dir}/{routeenv}/osm_file.pbf",
            gtfs=gtfs_fullpaths,
        )
    except ValueError as e:
        print(f"Error loading network for {routeenv}: {e}")
        print("TRY EMPTYING YOUR CACHE /home/user/.cache/r5py/*")
        raise e

    # route on network to create travel time matrix
    print("calculating travel time matrix")
    ttm = r5py.TravelTimeMatrix(network,
                                rep_points,
                                rep_points,
                                transport_modes=[mode],
                                departure=departure_time,
                                **r5_params
                                )
    ttm_df = pd.DataFrame(ttm)  # shouldn't be necessary, but I was getting a weird r5py error

    # Ensure ids are strings for consistent indexing/column matching
    ttm_df["from_id"] = ttm_df["from_id"].astype(str)
    ttm_df["to_id"] = ttm_df["to_id"].astype(str)

    # Pivot to wide format: rows=from_id, columns=to_id
    ttm_df_wide = ttm_df.pivot(index="from_id", columns="to_id", values="travel_time")
    return ttm_df_wide

def post_process(ttm_df_wide, mode):
    if mode == "WALK":
        # Subtract 1 from travel times with minimum of 0
        # this is a way of ensuring that WALK will be selected over TRANSIT
        # consider this an approximation of the 'hassle factor' of taking transit
        # TODO: either add transit cost, or represent this with a 'travel convenience matrix'
        # Either way, document it better
        ttm_df_wide = (ttm_df_wide - 1).clip(lower=0)
    return ttm_df_wide

def route_for_all_envs(routing_dir: str,
                       geoms_with_dests, #maybe this shouldn't require dests? Only used for communication visualization
                       user_classes: pd.DataFrame,
                       departure_time = DEPARTURE_TIME,
                       visualize_all = True,
                       ):
    rep_points = gpd.GeoDataFrame(crs=geoms_with_dests.crs, geometry=geoms_with_dests.representative_point())
    rep_points['id'] = geoms_with_dests['geom_id']
    rep_points.to_file(f"{routing_dir}/DEBUG: rep_points.geojson", driver="GeoJSON")
    user_classes.fillna("", inplace=True)
    for mode in MODES:
        print(f"identifying routeenvs for {mode}")
        route_envs_for_mode = set()
        route_envs_for_mode.update(user_classes[f'routeenv_{mode}'].unique())
        route_envs_for_mode.discard("")
        print(f"routeenvs for {mode}: {route_envs_for_mode}")
        for routeenv in route_envs_for_mode:
            ttm_df_wide = route_environment(routeenv, mode, routing_dir, rep_points, departure_time)
            ttm_df_wide = post_process(ttm_df_wide, mode)
            ttm_df_wide.to_csv(f"{routing_dir}/{routeenv}/ttm_{mode}.csv")
            if visualize_all:
                geoms_with_dests.index = geoms_with_dests['geom_id'].values
                communication.visualize_access_to_zone(geoms_with_dests,
                                                       ttm_df_wide,
                                                       target_zone_id = None,
                                                       out_path=f"{routing_dir}/{routeenv}/route_{mode}_viz.html")

