import os
import pandas as pd
import geopandas as gpd
import PyQt6
from pathlib import Path

#assume we're running this from connectome/connectome/
#in case we're just in connectome/ :
if not os.path.exists('setup'):
    os.chdir('connectome/')


from setup.define_analysis_areas import usa_tracts_from_address
from setup.populate_people_usa import (
    create_subdemo_categories, create_subdemo_statistics)
from setup.populate_destinations import populate_all_dests_USA
from setup.physical_conditions import (
    download_osm, make_osm_editable, get_GTFS_from_mobility_database
)
from routing import route_for_all_envs
from representation import apply_experience_defintions


if __name__=='__main__':
    run_name = "test_run_burlington"
    os.makedirs(run_name,exist_ok=True)
    os.makedirs("existing_conditions", exist_ok=True)
    os.makedirs("existing_conditions/input_data", exist_ok=True)
    scenario_dir = f'{run_name}/existing_conditions'
    input_dir = f'{run_name}/existing_conditions/input_data'

    #get geometries (tracts)
    if not os.path.exists(f"{input_dir}/analysis_geometry.gpkg"):
        print("fetching census tracts")
        states = ["VT"]
        address = "26 University Pl, Burlington, VT 05405"
        buffer = 3000 #up to 10000
        burlington_tracts = usa_tracts_from_address(
            states,
            address,
            buffer=3000,
            save_to = f"{input_dir}/analysis_geometry.gpkg"
        )
    else:
        print("loading tracts from disk")
        burlington_tracts = gpd.read_file(f"{input_dir}/analysis_geometry.gpkg")

    #get tract info
    if not os.path.exists(f"{input_dir}/subdemo_categories.csv"):
        num_income_bins = 4
        print("establishing subdemo categories")
        subdemo_categories = create_subdemo_categories(
            burlington_tracts,
            num_income_bins,
            save_to = f"{input_dir}/subdemo_categories.csv"
        )
    else:
        print("loading subdemo categories from disk")
        subdemo_categories = pd.read_csv(f"{input_dir}/subdemo_categories.csv")

    if not os.path.exists(f"{input_dir}/subdemo_statistics.csv"):
        print("calculating subdemo membership statistics")
        subdemo_statistics = create_subdemo_statistics(
            burlington_tracts,
            subdemo_categories,
            save_to=f"{input_dir}/subdemo_statistics.csv"
        )
    else:
        print("loading subdemo membership statistics from disk")
        subdemo_statistics = pd.read_csv(f"{input_dir}/subdemo_statistics.csv")

    if not os.path.exists(f"{input_dir}/destination_statistics.gpkg"):
        print("populating destinations")
        burlington_tracts_with_dests = populate_all_dests_USA(
            burlington_tracts,
            "VT",
            True,
            save_to=f"{input_dir}/destination_statistics.gpkg"
        )
    else:
        print("loading destinations from disk")
        burlington_tracts_with_dests = gpd.read_file(f"{input_dir}/destination_statistics.gpkg")

    if not os.path.exists(f"{input_dir}/osm_study_area.pbf"):
        print("downloading osm data")
        download_osm(
            burlington_tracts,
            f"{input_dir}/osm_large_file.pbf",
            f"{input_dir}/osm_study_area.pbf",
            buffer_dist=500,
        )
        make_osm_editable(f"{input_dir}/osm_study_area.pbf", f"{input_dir}/osm_study_area_editable.osm")


    if (
            not os.path.exists(f"{input_dir}/GTFS/") or
            not any(p.suffix == ".zip" for p in Path(f"{input_dir}/GTFS/").iterdir())
    ):
        print("downloading GTFS data")
        get_GTFS_from_mobility_database(burlington_tracts,
                                        f"{input_dir}/GTFS/",
                                        0.2)

    if not os.path.exists(f"{scenario_dir}/subdemo_categories_with_routeenvs.csv"):
        print("defining experiences")
        subdemo_categories_with_routeenvs = apply_experience_defintions(f"{input_dir}/osm_study_area.pbf",
                                    f"{input_dir}/GTFS/",
                                   subdemo_categories,
                                   f"{scenario_dir}/routing/",
                                   f"{scenario_dir}/subdemo_categories_with_routeenvs.csv"
                                   )
    else:
        print("loading experiences")
        subdemo_categories_with_routeenvs = pd.read_csv(f"{scenario_dir}/subdemo_categories_with_routeenvs.csv")

    if not os.path.exists(f"{scenario_dir}/routing/universal_re/ttm_WALK.csv"):
        print("routing")
        route_for_all_envs(f"{scenario_dir}/routing",
                           burlington_tracts_with_dests,
                           subdemo_categories_with_routeenvs
                           )
    else:
        print("loading routing results")
        subdemo_categories_with_routeenvs = pd.read_csv(f"{scenario_dir}/subdemo_categories_with_routeenvs.csv")

