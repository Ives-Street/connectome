import os
import shutil
import geopandas as gpd
import pandas as pd
import logging

from pathlib import Path


#assume we're running this from connectome/connectome/
#in case we're just in connectome/ :
if not os.path.exists('setup'):
    os.chdir('connectome/')


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

from setup.populate_destinations import populate_all_dests_USA

from routing_and_impedance import route_for_all_envs
from representation import apply_experience_defintions
from evaluation import evaluate_scenario
from traffic_utils.volume_utils import add_and_calibrate_volume_attributes

import communication
from constants import TRAFFIC_PARAMS_PATH, load_traffic_params

logger = logging.getLogger(__name__)

def run_scenario(scenario_dir,
                 track_volumes: bool | dict = False):
    """

    Args:
        scenario_dir:
        track_volumes:
            If false, collect no info that can be used to determine the relative or absolute volume of traffic by link/userclass/etc
            If dict, collect info based only on trips that move through a specific set of edges.
                dict['checkpoint_node_ids'], dict['checkpoint_edge_attr'] and dict['checkpoint_edge_values']
                 represent inputs to car_routing.od_matrix_times_with_checkpoints
                 eg.: checkpoint_node_ids=Null, checkpoint_edge_attr='ref', checkpoint_edge_values=['DE 1', 'US 9;DE 1']
                A new graph will be saved to results, with a new attribute that indicates:
                "Of total VMT on all trips that pass through any of the checkpoint edges, what percentage of that VMT
                is on this particular edge?"
                This will be used when allocating induced demand from highway widenings.

    """
    logger.info(f"loading experiences for {scenario_dir}")

    analysis_areas = gpd.read_file(f"{scenario_dir}/input_data/analysis_areas.gpkg")
    analysis_areas.index = analysis_areas['geom_id'].values

    if not os.path.exists(f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"):
        logger.info(f"defining experiences for {scenario_dir}")
        user_classes_w_routeenvs = apply_experience_defintions(f"{scenario_dir}/input_data", scenario_dir)
        user_classes_w_routeenvs.index = user_classes_w_routeenvs.user_class_id.values
    else:
        logger.info("loading experiences from disk")
        user_classes_w_routeenvs = pd.read_csv(f"{scenario_dir}/routing/user_classes_with_routeenvs.csv")
        user_classes_w_routeenvs.index = user_classes_w_routeenvs.user_class_id.values
        user_classes_w_routeenvs.fillna("", inplace=True)


    #if not os.path.exists(f"{scenario_dir}/impedances"):
    logger.info("routing")
    route_for_all_envs(f"{scenario_dir}",
                       analysis_areas,
                       user_classes_w_routeenvs,
                       track_volumes = track_volumes,
                       )
    # else:
    #     logger.info("routing already done. skipping.")

    #structure ttms and create cost matrices
    if not os.path.exists(f"{scenario_dir}/results/geometry_results.gpkg"):
        logger.info(f"evaluating {scenario_dir}")
        evaluate_scenario(scenario_dir,
                          user_classes_w_routeenvs,
                          analysis_areas,
                          add_relative_for_induced_demand = bool(track_volumes)
                          )

    else:
        logger.info("scenario has already been run")
    if not os.path.exists(f"{scenario_dir}/results/geometry_results.html"):
        communication.make_radio_choropleth_map(
            scenario_dir=scenario_dir,
            in_data="results/geometry_results.gpkg",
            outfile="results/geometry_results.html"
        )
    else:
        logger.info("visualization has already been run")



def initialize_existing_conditions(scenario_dir,
                                   states,
                                   address = None,
                                   lat=None,
                                   lon=None,
                                   buffer = 30000,  #m
                                   taz_file = None,
                                   traffic_datasource = None,
                                   volume_datasource = None,
                                   transcad_source = None,
                                   ):

    os.makedirs(f"{scenario_dir}/input_data", exist_ok=True)
    input_dir = f'{scenario_dir}/input_data'

    # get census tracts
    os.makedirs(f"{input_dir}/census", exist_ok=True)
    if not os.path.exists(f"{input_dir}/census/census_tracts.gpkg"):
        if address is not None:
            logger.info(f"fetching census tracts from {address} with buffer {buffer}m")
            census_tracts = get_usa_tracts_from_location(
                states=states,
                buffer=buffer,
                save_to = f"{input_dir}/census/census_tracts.gpkg",
                address=address,
            )
        elif (lat is not None) and (lon is not None):
            logger.info(f"fetching census tracts from ({lat}, {lon}) with buffer {buffer}m")
            census_tracts = get_usa_tracts_from_location(
                states=states,
                buffer=buffer,
                save_to = f"{input_dir}/census/census_tracts.gpkg",
                lat=lat,
                lon=lon,
            )
        elif taz_file is not None:
            logger.info(f"fetching census tracts from polygon {taz_file}")
            census_tracts = get_usa_tracts_from_polygon(
                states=states,
                polygon=taz_file,
                save_to=f"{input_dir}/census/census_tracts.gpkg",
            )
        else:
            logger.info(f"fetching census tracts from {states} at the state level")
            census_tracts = get_usa_tracts_from_state(
                states=states,
                save_to = f"{input_dir}/census/census_tracts.gpkg"
            )
    else:
        # logger.info("loading tracts from disk")
        census_tracts = gpd.read_file(f"{input_dir}/census/census_tracts.gpkg",index_col=0)

    #get tract info
    populate_people_usa(scenario_dir)

    # get destinations
    # For now, we're going to get destinations at the tract level BEFORE we interpolate to TAZs,
    # because LODES jobs are at the tract level,
    # so that we can use the same interpolation for destinations

    if not os.path.exists(f"{input_dir}/census/census_tracts_with_dests.gpkg"):
        logger.info("populating overture destinations")
        populate_all_dests_USA(
            geographies=census_tracts,
            states=states,
            already_tracts=True,
            save_to=f"{input_dir}/census/census_tracts_with_dests.gpkg"
        )
    else:
        logger.info("loading destinations from disk")

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
            logger.info("loading analysis areas from disk (assuming they're already TAZs)")
        else:
            logger.info("interpolating tracts to TAZs")
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
                        volume_datasource = volume_datasource,
                        transcad_source = transcad_source)

    #todo maybe move this into physical_conditions?
    if volume_datasource == "tmas":
        os.makedirs(f"{input_dir}/traffic/", exist_ok=True)
        if not (os.path.exists(f"{input_dir}/tmas/tmas.VOL") and
                os.path.exists(f"{input_dir}/tmas/tmas.STA")):
            raise FileNotFoundError(
                f"TMAS input data not found in {input_dir}/tmas/tmas.VOL and {input_dir}/tmas/tmas.STA. Please download."
            )
        add_and_calibrate_volume_attributes(scenario_dir)

    else:
        raise ValueError(f"volume_datasource must be 'tmas', none others enabled")
        #TODO: download from https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/#y24  automatically
        # TODO also figure out how to handle multi-state areas



def copy_ttms(from_scenario_dir,
              to_scenario_dir,
              modes_to_copy = ['']):
    # note, this only copies the raw_ttms folder, not the processed impedances (cost matrices, gtms)
    # so impedance processing still happens.
    # theoretically, we should copy the processed impedances,
    # but that's such a small performance change that I won't bother for now
    for mode in modes_to_copy:
        routeenvs = os.listdir(f"{from_scenario_dir}/routing/")
        for routeenv in routeenvs:
            if os.path.exists(f"{from_scenario_dir}/routing/{routeenv}/raw_ttms"):
                ttms = os.listdir(f"{from_scenario_dir}/routing/{routeenv}/raw_ttms")
                for ttm in ttms:
                    if mode in ttm:
                        logger.info(f"copying {ttm} from {from_scenario_dir} to {to_scenario_dir}")
                        os.makedirs(f"{to_scenario_dir}/routing/{routeenv}/raw_ttms", exist_ok=True)
                        shutil.copyfile(f"{from_scenario_dir}/routing/{routeenv}/raw_ttms/{ttm}",
                                        f"{to_scenario_dir}/routing/{routeenv}/raw_ttms/{ttm}")


if __name__ == "__main__":
    from project_scripts.denver_i270_scenarios import create_denver_cdot_scenario, create_denver_hcnw_scenario

    study_name = "denver20"
    # communication.compare_scenarios(f"testing/{study_name}", "existing_conditions", "cdot_scenario")
    # compared_prepare_results(f"testing/{study_name}/cdot_scenario/")
    #
    # communication.compare_scenarios(f"testing/{study_name}", "existing_conditions", "hcnw_scenario")
    # compared_prepare_results(f"testing/{study_name}/hcnw_scenario/")

# def hold():
    traffic_datasource = "tomtom"
    volume_datasource = "tmas"
    study_name = "denver20"
    initialize_existing_conditions(
        scenario_dir=f"testing/{study_name}/existing_conditions/",
        states=["CO"],
        lat=39.805084,
        lon=-104.940186,
        buffer=20000,
        traffic_datasource=traffic_datasource,
        volume_datasource=volume_datasource,
    )
    run_scenario(f"testing/{study_name}/existing_conditions/",
                 track_volumes={
                                "checkpoint_node_ids": None,
                                "checkpoint_edge_attr": "ref",
                                "checkpoint_edge_values": ['I 270']}
                 )
    create_denver_cdot_scenario(
        f"testing/{study_name}/existing_conditions/",
        f"testing/{study_name}/cdot_scenario/",
        )
    run_scenario(f"testing/{study_name}/cdot_scenario/")

    create_denver_hcnw_scenario(
        f"testing/{study_name}/existing_conditions/",
        f"testing/{study_name}/hcnw_scenario/",
    )
    run_scenario(f"testing/{study_name}/hcnw_scenario/")

    communication.compare_scenarios(f"testing/{study_name}", "existing_conditions", "cdot_scenario")
    communication.summarize_compared_results(f"testing/{study_name}/cdot_scenario/")

    communication.compare_scenarios(f"testing/{study_name}", "existing_conditions", "hcnw_scenario")
    communication.summarize_compared_results(f"testing/{study_name}/hcnw_scenario/")

    # create_denver_hcnw_scenario("testing/denver/existing_conditions/",
    #                             "testing/denver/hcnw_scenario/")
    # run_scenario("testing/denver/hcnw_scenario/")
    # communication.compare_scenarios("testing/denver",
    #                                 "existing_conditions",
    #                                 "cdot_scenario")
    # communication.compare_scenarios("testing/denver",
    #                                 "existing_conditions",
    #                                 "hcnw_scenario")

