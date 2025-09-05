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
from representation import apply_experience_defintions


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Set up arguments for testing each setup script - should be organized logically
    # i.e. populate destinations + populate people should be run before valuations
    parser.add_argument('--address', type=str, default="26 University Pl, Burlington, VT 05405",
                        help='Address for analysis area')
    parser.add_argument('--state', type=str, default="VT",
                        help='State for analysis area')
    parser.add_argument('--areas', help = 'run define_analysis_areas', action='store_true')
    parser.add_argument('--people', help = 'run populate_people_usa', action='store_true')
    parser.add_argument('--dests', help = 'run populate_destinations', action='store_true')
    parser.add_argument('--conditions', help = 'run physical_conditions', action='store_true')
    parser.add_argument('--valuations', help = 'run define_valuations', action='store_true')
    #parser.add_argument('--all', help = 'run all three setup steps', action='store_true')
    args = parser.parse_args()

    no_args = not any([args.areas, args.people, args.dests, args.conditions, args.valuations])


    run_name = "test_run"
    os.makedirs(run_name,exist_ok=True)
    os.makedirs(f"{run_name}/existing_conditions", exist_ok=True)
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
            subdemo_categories = pd.read_csv(f"{input_dir}/subdemo_categories.csv")

        if not os.path.exists(f"{input_dir}/subdemo_statistics.csv"):
            print("calculating subdemo membership statistics")
            subdemo_statistics = create_subdemo_statistics(
                study_area_tracts,
                subdemo_categories,
                save_to=f"{input_dir}/subdemo_statistics.csv"
            )
        else:
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
            study_area_tracts_with_dests = gpd.read_file(f"{input_dir}/destination_statistics.gpkg")

    if args.conditions or no_args:
        #get physical conditions
        if not os.path.exists(f"{input_dir}/osm_study_area.pbf"):
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
        subdemo_categories_with_routeenvs = pd.read_csv(f"{scenario_dir}/subdemo_categories_with_routeenvs.csv")
#
# routeenv_dir = "test_run_burlington/existing_conditions/routing"
# routeenv = 'universal_re'
# from r5py import TransportNetwork
# gtfs_files = os.listdir(f"{routeenv_dir}/{routeenv}/gtfs_files")
# gtfs_fullpaths = [f"{routeenv_dir}/{routeenv}/gtfs_files/{filename}" for filename in gtfs_files]
# network = TransportNetwork(
#     osm_pbf=f"{routeenv_dir}/{routeenv}/osm_file.pbf",
#     gtfs=gtfs_fullpaths,
# )
