import os
import pandas as pd
import geopandas as gpd
import PyQt6
from pathlib import Path
import argparse
from pygris import states

#assume we're running this from connectome/connectome/
#in case we're just in connectome/ :
if not os.path.exists('setup'):
    os.chdir('connectome/')


from setup.define_analysis_areas import (
    get_usa_tracts_from_address,
    get_usa_tracts_from_state,
)
from setup.populate_people_usa import (
    create_userclasses, create_userclass_statistics)
from setup.populate_destinations import populate_all_dests_USA, populate_destinations_overture_places
from setup.physical_conditions import (
    download_osm, make_osm_editable, get_GTFS_from_mobility_database
)
from setup.define_valuations import generalize_destination_units
from routing_and_impedance import route_for_all_envs
from representation import apply_experience_defintions
from evaluation import evaluate_scenario

import communication

# I removed the usage of argparse becuase it was preventing me from running this in the pycharm interactive shell
# Can add it back later, but honestly I don't see anyone using this from the commandline for a while.

    #parser = argparse.ArgumentParser()
    # Set up arguments for testing each setup script - should be organized logically
    # i.e. populate destinations + populate people should be run before valuations
    # parser.add_argument('--address', type=str, default="26 University Pl, Burlington, VT 05405",
    #                     help='Address for analysis area')
    # parser.add_argument('--state', type=str, default="VT",
    #                     help='State for analysis area')
    #
    # parser.add_argument('--run_dir', type=str, default="test_run_burlington",
    #                     help = 'directory to store scenario files')
    #
    # parser.add_argument('--areas', help = 'run define_analysis_areas', action='store_true')
    # parser.add_argument('--people', help = 'run populate_people_usa', action='store_true')
    # parser.add_argument('--dests', help = 'run populate_destinations', action='store_true')
    # parser.add_argument('--conditions', help = 'run physical_conditions', action='store_true')
    # parser.add_argument('--valuations', help = 'run define_valuations', action='store_true')
    # #parser.add_argument('--all', help = 'run all three setup steps', action='store_true')
    # args = parser.parse_args()
    #
    # no_args = not any([args.areas, args.people, args.dests, args.conditions, args.valuations])

#Parameters for testing
# run_name = "burlington_test"
# states = ["VT"]
# address = "26 University Pl, Burlington, VT 05405"
# buffer = 3000


def run_scenario(scenario_dir,
                 states,
                 address = None,
                 buffer = 10000, #m
                ):

    os.makedirs(f"{scenario_dir}/input_data", exist_ok=True)
    input_dir = f'{scenario_dir}/input_data'

    # get analysis area
    if not os.path.exists(f"{input_dir}/analysis_geometry.gpkg"):

        if address is not None:
            print(f"fetching census tracts from {address} with buffer {buffer}m")
            study_area_tracts = get_usa_tracts_from_address(
                states=states,
                address=address,
                buffer_dist=buffer,
                save_to = f"{input_dir}/analysis_geometry.gpkg"
            )
        else:
            print(f"fetching census tracts from {states} at the state level")

            study_area_tracts = get_usa_tracts_from_state(
                states=['RI'],
                save_to = f"{input_dir}/analysis_geometry.gpkg"
            )
    else:
        # print("loading tracts from disk")
        study_area_tracts = gpd.read_file(f"{input_dir}/analysis_geometry.gpkg",index_col=0)

    #get tract info
    if not os.path.exists(f"{input_dir}/user_classes.csv"):
        num_income_bins = 4
        print("establishing user classes")
        user_classes = create_userclasses(
            study_area_tracts,
            num_income_bins,
            save_to = f"{input_dir}/user_classes.csv"
        )
    else:
        print("loading user classes from disk")
        user_classes = pd.read_csv(f"{input_dir}/user_classes.csv")
        user_classes.index = user_classes.user_class_id.values
        user_classes.fillna("",inplace=True)

    if not os.path.exists(f"{input_dir}/userclass_statistics.csv"):
        print("calculating user class statistics")
        userclass_statistics = create_userclass_statistics(
            study_area_tracts,
            user_classes,
            save_to=f"{input_dir}/userclass_statistics.csv"
        )
    else:
        print("loading user class statistics from disk")
        userclass_statistics = pd.read_csv(f"{input_dir}/userclass_statistics.csv")

    #get destinations
    if not os.path.exists(f"{input_dir}/destination_statistics.gpkg"):
        print("populating overture destinations")
        study_area_tracts_with_dests = populate_all_dests_USA(
            geographies = study_area_tracts,
            states = states,
            already_tracts = True,
            save_to = f"{input_dir}/destination_statistics.gpkg"
        )
    else:
        print("loading destinations from disk")
        study_area_tracts_with_dests = gpd.read_file(f"{input_dir}/destination_statistics.gpkg")
        study_area_tracts_with_dests.index = study_area_tracts_with_dests['geom_id'].values

    #get physical conditions
    if not os.path.exists(f"{input_dir}/osm_study_area.pbf"):
        print("downloading osm data")
        download_osm(
            study_area_tracts,
            f"{input_dir}/osm_large_file.pbf",
            f"{input_dir}/osm_large_file_filtered.pbf",
            f"{input_dir}/osm_study_area.pbf",
            buffer_dist=500,
        )
        make_osm_editable(f"{input_dir}/osm_study_area.pbf", f"{input_dir}/osm_study_area_editable.osm")


    if (
            not os.path.exists(f"{input_dir}/GTFS/")# or
            #not any(p.suffix == ".zip" for p in Path(f"{input_dir}/GTFS/").iterdir())
    ):
        print("downloading GTFS data")
        get_GTFS_from_mobility_database(study_area_tracts,
                                        f"{input_dir}/GTFS/",
                                        0.2)

    if not os.path.exists(f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"):
        print("defining experiences")
        user_classes_w_routeenvs = apply_experience_defintions(f"{input_dir}/osm_study_area.pbf",
                                    f"{input_dir}/GTFS/",
                                                                 user_classes,
                                   f"{scenario_dir}/routing/",
                                   f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"
                                                                 )
    else:
        print("loading experiences")
        user_classes_w_routeenvs = pd.read_csv(f"{scenario_dir}/routing/user_classes_with_routeenvs.csv")
        user_classes_w_routeenvs.index = user_classes_w_routeenvs.user_class_id.values
        user_classes_w_routeenvs.fillna("", inplace=True)

    if not os.path.exists(f"{scenario_dir}/impedances"):
        print("routing")
        route_for_all_envs(f"{scenario_dir}",
                           study_area_tracts_with_dests,
                           user_classes_w_routeenvs
                           )
    else:
        print("routing already done. skipping.")

    #structure ttms and create cost matrices

    if not os.path.exists(f"{scenario_dir}/results/geometry_results.geojson"):
        print("evaluating")
        evaluate_scenario(scenario_dir, user_classes_w_routeenvs, userclass_statistics, study_area_tracts_with_dests)
        communication.make_radio_choropleth_map(
            scenario_dir=scenario_dir,
            in_data="results/geometry_results.gpkg",
            outfile="results/geometry_results.html"
        )
    else:
        print("scenario has already been run")


if __name__ == "__main__":
    run_scenario(
        scenario_dir = "testing/ri_test/regional_rail",
        states = ["RI"],
        address = None
    )
    communication.compare_scenarios("testing/ri_test", "existing_conditions", "regional_rail")