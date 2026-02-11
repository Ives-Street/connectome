import os
import shutil
import geopandas as gpd
import pandas as pd
import logging
import json

from tqdm import tqdm
import osmnx as ox
from shapely.geometry import Point, Polygon
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
from traffic_utils.assignment import recalculate_speeds, relative_to_absolute_induced_demand, save_traffic_stats
from traffic_utils.volume_utils import add_and_calibrate_volume_attributes

import communication

logger = logging.getLogger(__name__)

TRAFFIC_PARAMS_PATH = Path(__file__).parent / "traffic_utils" / "traffic_analysis_parameters.json"

def load_traffic_params(path: str | Path = TRAFFIC_PARAMS_PATH):
    """Load traffic analysis parameters (functional classes + clamps)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

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


def bearings_aligned(b1, b2, margin_deg):
    diff = abs((b1 - b2 + 180) % 360 - 180)
    return diff <= margin_deg

def nodes_within_polygon(G,u_id,v_id,polygon):
    u = Point(G.nodes[u_id]['x'], G.nodes[u_id]['y'])
    v = Point(G.nodes[v_id]['x'], G.nodes[v_id]['y'])
    return polygon.contains(u) and polygon.contains(v)

def redistribute_traffic_hncw(G, fraction_to_redistribute = 0.8):
    #TODO make this run fast if I care later

    capacity_factor = 0.9

    i270_target_bearing = 135

    G = ox.add_edge_bearings(G)

    total_to_redistribute = 0

    #compute total to redistribute
    logger.info("computing total to redistribute")
    for u, v, k, data in tqdm(list(G.edges(keys=True, data=True))):
        if "I270_HCNWA" in data.get('ref',""):
            if bearings_aligned(data.get('bearing'), i270_target_bearing, 90):
                capacity = float(data.get('capacity_vph'))
                volume = float(data.get('post_induction_vol_vph'))
                if volume > (capacity*capacity_factor):
                    total_to_redistribute = max(total_to_redistribute, volume - (capacity*capacity_factor))

    total_to_remove = total_to_redistribute
    total_to_add = total_to_redistribute * fraction_to_redistribute

    try:
        assert total_to_redistribute > 0, "no vehicles to redistribute"
    except AssertionError:
        import pdb; pdb.set_trace()

    redist_area_bounds = [
        (-104.99001, 39.77860),
        (-104.87193, 39.77468),
        (-104.87094, 39.83383),
        (-104.98368, 39.82842),
        (-104.99001, 39.77860),
        ]

    redist_area_geom = Polygon(redist_area_bounds)

    alt_route_osmids = {
        627902176, 627902175,
        627902174,
        605513707,
        628056983,
        605513700, 605513701, 16966246, 35306825, 521695385, 1342037754,
        427814288, 903407217, 37356331, 427814287,
        37356330, 177242477, 16966214,
        35883125,
        1341994240, 1341994241, 628376843, 35883125,
        1340514008, 628376844,
        88175761, 35883146,
        307640401, 307640402, 35797660, 24817799,
        88471266, 88471267, 88471268, 35882293,
        628433089,
        88471337, 88471274,
        1342042682, 49230950,
        628433096,
        628433120, 1079584167, 967150904, 967150906, 967150908,
        1413223021, 1395930286,
        967150912, 967150915, 1079584168, 1079584170, 1079584172, 1079584174, 1395930287, 967150911,
        1079608805, 1079608806, 1341992840, 1079608811, 1079608812, 1079608816, 1079608817, 1161301363, 1038603231,
        1111534208, 967150903, 1078079865, 1078079866, 1078079867, 1078079868, 967150909, 967150910,
        1341998770, 24851315, 24851316, 955926647,
        967781611,
        1078082908, 24873383,
        1132446921,
        24873384, 24873491, 24873492,
        860521539, 89177389, 89177390, 24874518, 24874519,
        628728355, 24874214, 24874215,
        800506261, 800506262, 372294119,
    }

    logger.info("redistributing traffic")
    links_redistributed = 0
    for u, v, k, data in tqdm(list(G.edges(keys=True, data=True))):
        if "I270_HCNWA" in data.get('ref', ""):
            if bearings_aligned(data.get('bearing'), i270_target_bearing, 90):
                if nodes_within_polygon(G,u,v,redist_area_geom):
                    current_volume = float(data.get('post_induction_vol_vph'))
                    new_volume = current_volume - total_to_remove
                    data['post_induction_vol_vph'] = new_volume
                    data['redistributed_vol'] = total_to_remove * -1
                    links_redistributed += 1
        try:
            link_osmids = set(data['osmid'])
        except TypeError:
            link_osmids = {data['osmid']}
        if bool(link_osmids & alt_route_osmids):
            current_volume = float(data.get('post_induction_vol_vph'))
            new_volume = current_volume + total_to_add
            data['post_induction_vol_vph'] = new_volume
            data['redistributed_vol'] = total_to_add
            links_redistributed += 1

    logger.info(f"redistributed traffic on {links_redistributed} links")
    return G

# TODO generalize to 'create widening scenario'?
def create_denver_cdot_scenario(
        scenario_dir,
        new_scenario_dir,
        induced_demand_annual_vmt = 104000000,
):
    params = load_traffic_params()

    daily_induced_demand_vmt = induced_demand_annual_vmt / 252  # TODO add this to params
    K_factor = 0.12  # TODO add this to params
    peak_hour_induced_demand_vmt = daily_induced_demand_vmt * K_factor
    improved_lane_capacity_per_lane = params['custom_capacities_perlane']['improved_lanes']

    if os.path.exists(f"{new_scenario_dir}/input_data/traffic/routing_graph.graphml"):
        logger.info("scenario already exists, skipping")
    else:
        logger.info("creating denver cdot scenario")
        os.makedirs(f"{new_scenario_dir}/input_data/", exist_ok=True)
        for file in ['analysis_areas.gpkg', 'userclass_statistics.csv', 'user_classes.csv','osm_study_area.pbf']:
            shutil.copyfile(f"{scenario_dir}/input_data/{file}", f"{new_scenario_dir}/input_data/{file}")
        shutil.copytree(f'{scenario_dir}/input_data/GTFS', f"{new_scenario_dir}/input_data/GTFS", dirs_exist_ok=True)

        G = ox.load_graphml(f"{scenario_dir}/results/traffic/routing_graph_with_relative_demand.graphml")

        aux_lane_osmids = {37355553, 82574884, 82574885, 82574887, 37320107,
                           82574882, 37355557, 82574886, 37320103, 628120912, 628120916, 628120917,
                           44769380, 628649541, 16974022, 82574884, 16974002, 16966131,
                           44769376, 82574881, 82574882, 628649543, 628649544, 16966126
                           }


        added_aux_lanes = 0

        for u, v, k, data in tqdm(list(G.edges(keys=True, data=True))):
            if "I 270" in data.get('ref',""):
                data['toll'] = 'yes'
                data['ref'] = 'I270_CDOT'
                lanes = float(data.get('lanes'))
                capacity = float(data.get('capacity_vph'))

                new_capacity_per_lane = improved_lane_capacity_per_lane

                try:
                    link_osmids = set(data['osmid'])
                except TypeError:
                    link_osmids = {data['osmid']}
                if bool(link_osmids & aux_lane_osmids):
                    new_lanes = lanes + 2
                    added_aux_lanes += 1
                else:
                    new_lanes = lanes + 1
                new_capacity = new_capacity_per_lane * new_lanes
                data['lanes'] = new_lanes
                data['capacity_vph'] = new_capacity
                data['calibration_factor'] = 1 #reset calibration, pure modeling
                data['recalculate_speed'] = True
                logger.info(f"setting lanes and capacity for edge {u}-{v} to {new_lanes} and {new_capacity}, from {lanes} and {capacity}")

        try:
            assert added_aux_lanes >= 2
        except AssertionError:
            import pdb
            pdb.set_trace()

        G = relative_to_absolute_induced_demand(new_scenario_dir, G, peak_hour_induced_demand_vmt, save=False)

        G = recalculate_speeds(G)

        # Save modified graph
        save_traffic_stats(G, new_scenario_dir)

    logger.info("copying ttms")
    copy_ttms(scenario_dir, new_scenario_dir, modes_to_copy = ['WALK',"BICYCLE","TRANSIT"])

def create_denver_hcnw_scenario(
        scenario_dir,
        new_scenario_dir,
        induced_demand_annual_vmt = 22000000,
):
    params = load_traffic_params()

    daily_induced_demand_vmt = induced_demand_annual_vmt / 260  # TODO add this to params
    K_factor = 0.12  # TODO add this to params
    peak_hour_induced_demand_vmt = daily_induced_demand_vmt * K_factor
    improved_lane_capacity_per_lane = params['custom_capacities_perlane']['improved_lanes']

    if os.path.exists(f"{new_scenario_dir}/input_data/traffic/routing_graph.graphml"):
        logger.info("scenario already exists, skipping")
    else:
        logger.info("creating denver hcnw scenario")
        os.makedirs(f"{new_scenario_dir}/input_data/", exist_ok=True)
        for file in ['analysis_areas.gpkg', 'userclass_statistics.csv', 'user_classes.csv','osm_study_area.pbf']:
            shutil.copyfile(f"{scenario_dir}/input_data/{file}", f"{new_scenario_dir}/input_data/{file}")
        shutil.copytree(f'{scenario_dir}/input_data/GTFS', f"{new_scenario_dir}/input_data/GTFS", dirs_exist_ok=True)

        G = ox.load_graphml(f"{scenario_dir}/results/traffic/routing_graph_with_relative_demand.graphml")

        aux_lane_osmids = {37355553, 82574884, 82574885, 82574887, 37320107,
                           82574882, 37355557, 82574886, 37320103, 628120912, 628120916, 628120917,
                           44769380, 628649541, 16974022, 82574884, 16974002, 16966131,
                           44769376, 82574881, 82574882, 628649543, 628649544, 16966126
                           }


        added_aux_lanes = 0

        for u, v, k, data in tqdm(list(G.edges(keys=True, data=True))):
            if "I 270" in data.get('ref',""):
                data['toll'] = 'yes'
                data['ref'] = 'I270_HCNWA'
                lanes = float(data.get('lanes'))
                capacity = float(data.get('capacity_vph'))

                new_capacity_per_lane = improved_lane_capacity_per_lane + 200

                try:
                    link_osmids = set(data['osmid'])
                except TypeError:
                    link_osmids = {data['osmid']}
                if bool(link_osmids & aux_lane_osmids):
                    new_lanes = lanes + 1
                    added_aux_lanes += 1
                else:
                    new_lanes = lanes

                new_capacity = new_capacity_per_lane * new_lanes
                data['lanes'] = new_lanes
                data['capacity_vph'] = new_capacity
                data['calibration_factor'] = 1 #reset calibration, pure modeling
                data['recalculate_speed'] = True
                logger.info(f"setting lanes and capacity for edge {u}-{v} to {new_lanes} and {new_capacity}, from {lanes} and {capacity}")

        try:
            assert added_aux_lanes >= 2
        except AssertionError:
            import pdb
            pdb.set_trace()

        relative_to_absolute_induced_demand(new_scenario_dir, G, peak_hour_induced_demand_vmt)

        G = redistribute_traffic_hncw(G)

        G = recalculate_speeds(G)

        # Save modified graph
        save_traffic_stats(G, new_scenario_dir)

    logger.info("copying ttms")
    copy_ttms(scenario_dir, new_scenario_dir, modes_to_copy = ['WALK',"BICYCLE","TRANSIT"])

if __name__ == "__main__":
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

