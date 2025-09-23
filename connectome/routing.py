import os

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
                       subdemo_categories: pd.DataFrame,
                       departure_time = datetime.datetime(2025,9,23,9,0,0),
                       visualize_all = True,
                       ):
    rep_points = gpd.GeoDataFrame(crs=geoms.crs, geometry=geoms.representative_point())
    rep_points['id'] = geoms['geom_id']
    rep_points.to_file(f"{routing_dir}/DEBUG: rep_points.geojson", driver="GeoJSON")
    for mode in MODES:
        print(f"identifying routeenvs for {mode}")
        route_envs_for_mode = set()
        route_envs_for_mode.update(subdemo_categories[f'routeenv_{mode}'].unique())
        for routeenv in route_envs_for_mode:
            print(f"routing {mode} for {routeenv}")
            gtfs_files = os.listdir(f"{routing_dir}/{routeenv}/gtfs_files")
            gtfs_fullpaths = [f"{routing_dir}/{routeenv}/gtfs_files/{filename}" for filename in gtfs_files]
            network = r5py.TransportNetwork(
                osm_pbf=f"{routing_dir}/{routeenv}/osm_file.pbf",
                gtfs=gtfs_fullpaths,
            )
            ttm = r5py.TravelTimeMatrix(network,
                                        rep_points,
                                        rep_points,
                                        transport_modes=[r5py.TransportMode.WALK],
                                        departure=departure_time,
                                        )
            ttm_df = pd.DataFrame(ttm) #shouldn't be necessary, but I was getting a weird r5py error
            ttm_df = ttm_df.set_index(ttm.from_id)
            ttm_df.index = ttm_df.index.astype(str)
            ttm_df_wide = ttm_df.pivot(index="from_id", columns="to_id", values="travel_time")
            ttm_df_wide.to_csv(f"{routing_dir}/{routeenv}/ttm_{mode}.csv")
            if visualize_all:
                communication.visualize_route_for_mode(geoms_with_dests,
                                                       ttm_df_wide,
                                                       f"{routing_dir}/{routeenv}/route_{mode}_viz.html")

