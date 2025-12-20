import os
import pandas as pd
import geopandas as gpd
import PyQt6
from pygris import states
import shutil

import osmnx as ox

from connectome.traffic_utils.tmas_volume_utils import demand_to_volume_ratio

#assume we're running this from connectome/connectome/
#in case we're just in connectome/ :
if not os.path.exists('setup'):
    os.chdir('connectome/')

from traffic_utils.assignment_utils import assign_relative_demand, back_calculate_freeflow_speeds_from_obs
from traffic_utils.tmas_volume_utils import apply_tmas_obs_volumes_to_graph, infer_volumes_from_relative_demand

from setup.physical_conditions import physical_conditions
from setup.populate_people_usa import populate_people_usa

from setup.census_geometry_usa import (
    get_usa_tracts_from_location,
    get_usa_tracts_from_state,
    get_usa_tracts_from_polygon,
)

from setup.geography_utils import (
    interpolate_tracts_to_tazs,
    calculate_population_per_sqkm,
)

from setup.populate_destinations import populate_all_dests_USA, populate_destinations_overture_places

from setup.define_valuations import generalize_destination_units
from routing_and_impedance import route_for_all_envs
from representation import apply_experience_defintions
from evaluation import evaluate_scenario

import communication


def run_scenario(scenario_dir,
                 volume_datasource = None):
    print("loading experiences")

    analysis_areas = gpd.read_file(f"{scenario_dir}/input_data/analysis_areas.gpkg")
    analysis_areas.index = analysis_areas['geom_id'].values

    if not os.path.exists(f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"):
        print("defining experiences")
        user_classes_w_routeenvs = apply_experience_defintions(f"{scenario_dir}/input_data", scenario_dir)
        user_classes_w_routeenvs.index = user_classes_w_routeenvs.user_class_id.values
    else:
        print("loading experiences from disk")
        user_classes_w_routeenvs = pd.read_csv(f"{scenario_dir}/routing/user_classes_with_routeenvs.csv")
        user_classes_w_routeenvs.index = user_classes_w_routeenvs.user_class_id.values
        user_classes_w_routeenvs.fillna("", inplace=True)


    if not os.path.exists(f"{scenario_dir}/impedances"):
        print("routing")
        route_for_all_envs(f"{scenario_dir}",
                           analysis_areas,
                           user_classes_w_routeenvs
                           )
    else:
        print("routing already done. skipping.")

    #structure ttms and create cost matrices
    if not os.path.exists(f"{scenario_dir}/results/geometry_results.gpkg"):
        print("evaluating")
        evaluate_scenario(scenario_dir,
                          user_classes_w_routeenvs,
                          analysis_areas,
                          )
    if not os.path.exists(f"{scenario_dir}/results/geometry_results.html"):
        communication.make_radio_choropleth_map(
            scenario_dir=scenario_dir,
            in_data="results/geometry_results.gpkg",
            outfile="results/geometry_results.html"
        )

    else:
        print("scenario has already been run")

    # traffic benchmarking analysis
    if bool(volume_datasource):
        if not os.path.exists(f"{scenario_dir}/input_data/traffic/post_benchmark/graph_with_relative_demands.graphml"):
            print("assigning relative demands")
            assign_relative_demand(scenario_dir)
        G, matched_edges = None, None
        if not os.path.exists(f"{scenario_dir}/input_data/traffic/post_benchmark/edges_with_demands_and_TMAS_vols.gpkg"):
            print("applying TMAS volumes to graph")
            G, matched_edges = apply_tmas_obs_volumes_to_graph(scenario_dir,
                                            date_start="2024-08-05",
                                            date_end="2024-08-09",
                                            hours=[8],
                                            )
        if not os.path.exists(f"{scenario_dir}/input_data/traffic/post_benchmark/graph_with_demands_and_modeled_vols.graphml"):
            print("establishing demand_to_volume_ratio")
            infer_volumes_from_relative_demand(scenario_dir, G, matched_edges)
        if not os.path.exists(f"{scenario_dir}/input_data/traffic/post_benchmark/routing_graph.graphml"):
            back_calculate_freeflow_speeds_from_obs(scenario_dir)

def initialize_existing_conditions(scenario_dir,
                                   states,
                                   address = None,
                                   lat=None,
                                   lon=None,
                                   buffer = 30000,  #m
                                   taz_file = None,
                                   traffic_datasource = None,
                                   volume_datasource = None,
                                   ):

    os.makedirs(f"{scenario_dir}/input_data", exist_ok=True)
    input_dir = f'{scenario_dir}/input_data'

    # get census tracts
    os.makedirs(f"{input_dir}/census", exist_ok=True)
    if not os.path.exists(f"{input_dir}/census/census_tracts.gpkg"):
        if address is not None:
            print(f"fetching census tracts from {address} with buffer {buffer}m")
            census_tracts = get_usa_tracts_from_location(
                states=states,
                buffer=buffer,
                save_to = f"{input_dir}/census/census_tracts.gpkg",
                address=address,
            )
        if (lat is not None) and (lon is not None):
            print(f"fetching census tracts from {address} with buffer {buffer}m")
            census_tracts = get_usa_tracts_from_location(
                states=states,
                buffer=buffer,
                save_to = f"{input_dir}/census/census_tracts.gpkg",
                lat=lat,
                lon=lon,
            )
        elif taz_file is not None:
            print(f"fetching census tracts from polygon {taz_file}")
            census_tracts = get_usa_tracts_from_polygon(
                states=states,
                polygon=taz_file,
                save_to=f"{input_dir}/census/census_tracts.gpkg",
            )
        else:
            print(f"fetching census tracts from {states} at the state level")

            census_tracts = get_usa_tracts_from_state(
                states=['RI'],
                save_to = f"{input_dir}/census/census_tracts.gpkg"
            )
    else:
        # print("loading tracts from disk")
        census_tracts = gpd.read_file(f"{input_dir}/census/census_tracts.gpkg",index_col=0)

    #get tract info
    populate_people_usa(scenario_dir)

    # get destinations
    # For now, we're going to get destinations at the tract level BEFORE we interpolate to TAZs,
    # because LODES jobs are at the tract level,
    # so that we can use the same interpolation for destinations

    if not os.path.exists(f"{input_dir}/census/census_tracts_with_dests.gpkg"):
        print("populating overture destinations")
        tracts_with_dests = populate_all_dests_USA(
            geographies=census_tracts,
            states=states,
            already_tracts=True,
            save_to=f"{input_dir}/census/census_tracts_with_dests.gpkg"
        )
    else:
        print("loading destinations from disk")
        analysis_areas = gpd.read_file(f"{input_dir}/census/census_tracts_with_dests.gpkg")
        analysis_areas.index = analysis_areas['geom_id'].values

    # determine analysis areas
    # if we're using census tracts as the TAZs, copy them directly
    # if we have TAZs already, MAUP interpolate to them
    ### TODO: Base ratios off of tracts, but population numbers off of blocks

    if taz_file is None: #assume we're using tracts
        shutil.copyfile(f"{input_dir}/census/census_tracts_with_dests.gpkg", f"{input_dir}/analysis_areas.gpkg")
        shutil.copyfile(f"{input_dir}/census/tract_userclass_statistics.csv", f"{input_dir}/userclass_statistics.csv")
    else: #we're using some kind of TAZs
        #check if we've already done this interpolation
        if os.path.exists(f"{input_dir}/analysis_areas.gpkg") and os.path.exists(
                f"{input_dir}/userclass_statistics.csv"):
            print("loading analysis areas from disk (assuming they're already TAZs)")
        else:
            print("interpolating tracts to TAZs")
            interpolate_tracts_to_tazs(
                tracts=f"{input_dir}/census/census_tracts_with_dests.gpkg",
                tazs=taz_file,
                userclass_statistics=f"{input_dir}/census/tract_userclass_statistics.csv",
                taz_id_col="geom_id",
                create_taz_id_col=True,
                source_geom_id_col="geom_id",
                save_userclass_csv_to=f"{input_dir}/userclass_statistics.csv",
                save_analysis_areas_gpkg_to=f"{input_dir}/analysis_areas.gpkg",
                interpolate_tract_cols=['overture_places','lodes_jobs']
            )

    calculate_population_per_sqkm(input_dir)

    # get physical conditions
    # includes its own has-this-already-been-run checks
    physical_conditions(scenario_dir,
                        traffic_datasource = traffic_datasource,
                        volume_datasource = volume_datasource,)





def create_lewes_pricing_scenario(scenario_dir, new_scenario_dir):
    shutil.copytree(f'{scenario_dir}/input_data/', f"{new_scenario_dir}/input_data/", dirs_exist_ok=True)


    G = ox.load_graphml(f"{new_scenario_dir}/input_data/traffic/post_benchmark/routing_graph.graphml")
    # Set toll attribute for Coastal Highway edges
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get('name') == 'Coastal Highway':
            G.edges[u, v, k]['toll'] = 'yes'

    # Save modified graph
    ox.save_graphml(G, f"{new_scenario_dir}/input_data/traffic/post_benchmark/routing_graph.graphml")


if __name__ == "__main__":
    traffic_datasource = "tomtom"
    volume_datasource = "tmas"
    initialize_existing_conditions(
        scenario_dir = "testing/lewes/existing_conditions/",
        states = ["DE"],
        lat=38.7710985,
        lon=-75.141997,
        buffer = 9000,
        traffic_datasource = traffic_datasource,
        volume_datasource = volume_datasource,

        #scenario_dir = "testing/denver/existing_conditions",
        #states = ["CO"],
        #taz_file = "testing/denver/taz.geojson"
    )
    run_scenario("testing/lewes/existing_conditions/",
                 volume_datasource=volume_datasource)

    create_lewes_pricing_scenario("testing/lewes/existing_conditions/", "testing/lewes/lewes_pricing/")

    run_scenario("testing/lewes/lewes_pricing",)

    communication.compare_scenarios("testing/lewes", "existing_conditions", "lewes_pricing")