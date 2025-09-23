import os
import shutil
import pandas as pd

from setup.prep_bike_osm import add_lts_tags

MODES = [
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
    #"RIDESHARE",
    #"SHARED_BICYCLE",
]


def define_experiences(input_osm_filename: str,
                       input_gtfs_dir: str,
                       subdemo_categories: pd.DataFrame,
                       destination_dir: str,
                       save_subdemos_to: str,
) -> pd.DataFrame:
    """
    Creates a new directory "routing environment" with OSM/GTFS files reflecting conditions as experienced
    by each combination of subdemographic group and mode
    Also assign each subdemo/mode combination to a routing environment.
    Args:
        input_osm_filename:
        input_gtfs_dir:
        subdemo_categories:
        destination_dir:
    """
    os.makedirs(destination_dir, exist_ok=True)

    # for now, we're going to assume the most basic version possible:
    # all subgroups experience the city the same way for CAR, TRANSIT, and WALK,
    # and there are only four different options for BIKE
    universal_routing_env_id = "universal_re"
    # first let's do CAR, TRANSIT, and WALK: 1) make the directory
    os.makedirs(f"{destination_dir}/{universal_routing_env_id}", exist_ok=True)
    # 2) copy the GTFS and OSM
    shutil.copy(input_osm_filename,f"{destination_dir}/{universal_routing_env_id}/osm_file.pbf")
    shutil.copytree(input_gtfs_dir, f"{destination_dir}/{universal_routing_env_id}/gtfs_files/")
    # 3) assign that ID to the subdemo_categories dataframe
    subdemo_categories.loc[:, "routeenv_CAR"] = universal_routing_env_id
    subdemo_categories.loc[:, "routeenv_TRANSIT"] = universal_routing_env_id
    subdemo_categories.loc[:, "routeenv_WALK"] = universal_routing_env_id

    # next let's do the bike options
    # we're ALSO going to take LTS into account in representation.py
    LTSs = [int(item[-1:]) for item in subdemo_categories['max_bicycle'].unique()]
    for lts in LTSs:
        bike_env_id = f"bike_re_lts{lts}"
        os.makedirs(f"{destination_dir}/{bike_env_id}/", exist_ok=True)
        # copy OSM, adding LTS tags
        add_lts_tags(input_osm_filename,
                     f"{destination_dir}/{bike_env_id}/osm_file.pbf",
                     lts)
        # copy GTFS, just for the heck of it, we won't actually use it for routing
        # maybe someday we'll route bike-to-transit options
        shutil.copytree(input_gtfs_dir, f"{destination_dir}/{bike_env_id}/gtfs_files/")
        #assign IDs
        selector = subdemo_categories['max_bicycle'] == f'bike_lts{lts}'
        subdemo_categories.loc[selector,"routeenv_BICYCLE"] = bike_env_id
    if not save_subdemos_to == "":
        subdemo_categories.to_csv(save_subdemos_to)
    return subdemo_categories