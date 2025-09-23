import os
import pandas as pd
import geopandas as gpd
import PyQt6
from pathlib import Path
import argparse

#assume we're running this from connectome/connectome/
#in case we're just in connectome/ :
if not os.path.exists('setup'):
    os.chdir('connectome/')


from setup.define_analysis_areas import usa_tracts_from_address
from setup.populate_people_usa import (
    create_subdemo_categories, create_subdemo_statistics)
from setup.populate_destinations import populate_all_dests_USA, populate_destinations_overture_places
from setup.physical_conditions import (
    download_osm, make_osm_editable, get_GTFS_from_mobility_database
)
from setup.define_valuations import value
from routing import route_for_all_envs
from representation import apply_experience_defintions


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Set up arguments for testing each setup script - should be organized logically
    # i.e. populate destinations + populate people should be run before valuations
    parser.add_argument('--address', type=str, default="26 University Pl, Burlington, VT 05405",
                        help='Address for analysis area')
    parser.add_argument('--state', type=str, default="VT",
                        help='State for analysis area')

    parser.add_argument('--run_dir', type=str, default="test_run_burlington",
                        help = 'directory to store scenario files')

    parser.add_argument('--areas', help = 'run define_analysis_areas', action='store_true')
    parser.add_argument('--people', help = 'run populate_people_usa', action='store_true')
    parser.add_argument('--dests', help = 'run populate_destinations', action='store_true')
    parser.add_argument('--conditions', help = 'run physical_conditions', action='store_true')
    parser.add_argument('--valuations', help = 'run define_valuations', action='store_true')
    #parser.add_argument('--all', help = 'run all three setup steps', action='store_true')
    args = parser.parse_args()

    no_args = not any([args.areas, args.people, args.dests, args.conditions, args.valuations])


    run_name = args.run_dir
    os.makedirs(run_name,exist_ok=True)
    os.makedirs(f"{run_name}/existing_conditions", exist_ok=True)
    scenario_dir = f"{run_name}/existing_conditions"
    os.makedirs(f"{run_name}/existing_conditions/input_data", exist_ok=True)
    input_dir = f'{run_name}/existing_conditions/input_data'

    # Think this through a bit better...
    study_area_tracts = gpd.read_file(f"{input_dir}/analysis_geometry.gpkg")


    if args.areas or no_args:
        # get analysis area
        if not os.path.exists(f"{input_dir}/analysis_geometry.gpkg"):
            print(f"fetching census tracts from {args.state}")
            states = [args.state]
            address = args.address
            buffer = 3000 #up to 10000
            study_area_tracts = usa_tracts_from_address(
                states,
                address,
                buffer=buffer,
                save_to = f"{input_dir}/analysis_geometry.gpkg"
            )
        else:
            print("loading tracts from disk")
            study_area_tracts = gpd.read_file(f"{input_dir}/analysis_geometry.gpkg")

    if args.people or no_args:
        #get tract info
        if not os.path.exists(f"{input_dir}/subdemo_categories.csv"):
            num_income_bins = 4
            print("establishing subdemo categories")
            subdemo_categories = create_subdemo_categories(
                study_area_tracts,
                num_income_bins,
                save_to = f"{input_dir}/subdemo_categories.csv"
            )
        else:
            print("loading subdemo categories from disk")
            subdemo_categories = pd.read_csv(f"{input_dir}/subdemo_categories.csv")

        if not os.path.exists(f"{input_dir}/subdemo_statistics.csv"):
            print("calculating subdemo membership statistics")
            subdemo_statistics = create_subdemo_statistics(
                study_area_tracts,
                subdemo_categories,
                save_to=f"{input_dir}/subdemo_statistics.csv"
            )
        else:
            print("loading subdemo membership statistics from disk")
            subdemo_statistics = pd.read_csv(f"{input_dir}/subdemo_statistics.csv")

    if args.dests or no_args:
        #get destinations
        if not os.path.exists(f"{input_dir}/destination_statistics.gpkg"):
            print("populating overture destinations")
            study_area_tracts_with_dests = populate_all_dests_USA(
                study_area_tracts,
                "VT",
                True,
                save_to=f"{input_dir}/destination_statistics.gpkg"
            )
        else:
            print("loading destinations from disk")
            study_area_tracts_with_dests = gpd.read_file(f"{input_dir}/destination_statistics.gpkg")

    if args.conditions or no_args:
        #get physical conditions
        if not os.path.exists(f"{input_dir}/osm_study_area.pbf"):
            print("downloading osm data")
            download_osm(
                study_area_tracts,
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
            get_GTFS_from_mobility_database(study_area_tracts,
                                            f"{input_dir}/GTFS/",
                                            0.2)

    if args.valuations or no_args:
        study_area_tracts_with_dests = gpd.read_file(f"{input_dir}/destination_statistics.gpkg")
        print("example valuation:", value("low_income", "overture_places", 15, "morning"))

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
                           study_area_tracts_with_dests,
                           subdemo_categories_with_routeenvs
                           )
    else:
        print("loading routing results")
        subdemo_categories_with_routeenvs = pd.read_csv(f"{scenario_dir}/subdemo_categories_with_routeenvs.csv")

