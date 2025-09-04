import os

import pandas as pd
from r5py import TransportNetwork
from tqdm import tqdm



MODES = [ #todo - make this universal for the whole codebase
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
]


def route_for_all_envs(routeenv_dir: str,
                       destination_dir: str,
                       subdemo_categories: pd.DataFrame,
                       ):
    routeenvs = set()
    for mode in MODES:
        routeenvs.update(subdemo_categories[f'routeenv_{mode}'].unique())
    for routeenv in tqdm(routenvs):
        gtfs_files = os.listdir(f"{routeenv_dir}/{routeenv}/gtfs_files")
        gtfs_fullpaths = [f"{routeenv_dir}/{routeenv}/gtfs_files/{filename}" for filename in gtfs_files]
        network = TransportNetwork(
            osm_pbf=f"{routeenv_dir}/{routeenv}/osm_file.pbf",
            gtfs=gtfs_fullpaths,
        )

