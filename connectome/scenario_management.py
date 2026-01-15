import os
import shutil
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import logging

from tqdm import tqdm
import osmnx as ox


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

from setup.populate_destinations import populate_all_dests_USA, populate_destinations_overture_places

from setup.define_valuations import generalize_destination_units
from routing_and_impedance import route_for_all_envs
from representation import apply_experience_defintions
from evaluation import evaluate_scenario
from traffic_utils.assignment import recalculate_speeds, relative_to_absolute_induced_demand
from traffic_utils.volume_utils import add_and_calibrate_volume_attributes
from traffic_utils.speed_utils import compute_bearing

import communication

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
    if not os.path.exists(f"{scenario_dir}/results/geometry_results.html"):
        communication.make_radio_choropleth_map(
            scenario_dir=scenario_dir,
            in_data="results/geometry_results.gpkg",
            outfile="results/geometry_results.html"
        )

    else:
        logger.info("scenario has already been run")



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

    # ---------------------------------------------------------
    # Define study area polygon (EPSG:4326) if lat/lon/buffer given
    # ---------------------------------------------------------
    study_area_geom_4326 = None
    if (lat is not None) and (lon is not None) and (buffer is not None):
        # Start from a point in EPSG:4326
        center_ll = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        # Project to a metric CRS (WebMercator) for buffering in meters
        center_m = center_ll.to_crs(3857)
        # Buffer by the requested distance (meters)
        study_area_m = center_m.buffer(buffer)
        # Project buffered polygon back to EPSG:4326
        study_area_ll = study_area_m.to_crs(4326)
        study_area_geom_4326 = study_area_ll.unary_union

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
        if (lat is not None) and (lon is not None):
            logger.info(f"fetching census tracts from {address} with buffer {buffer}m")
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
                states=['RI'],
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
        tracts_with_dests = populate_all_dests_USA(
            geographies=census_tracts,
            states=states,
            already_tracts=True,
            save_to=f"{input_dir}/census/census_tracts_with_dests.gpkg"
        )
    else:
        logger.info("loading destinations from disk")
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

def redistribute_traffic_hncw(G, fraction_to_redistribute = 0.8):
    #TODO make this run fast if I care later

    i270_target_bearing = 135
    i70_target_bearing = 90
    i25_target_bearing = 180

    G = ox.add_edge_bearings(G)

    total_to_redistribute = 0

    #compute total to redistribute
    logger.info("computing total to redistribute")
    for u, v, k, data in tqdm(list(G.edges(keys=True, data=True))):
        if "I270_HCNWA" in data.get('ref',""):
            if bearings_aligned(data.get('bearing'), i270_target_bearing, 90):
                capacity = float(data.get('capacity_vph'))
                volume = float(data.get('post_induction_vol_vph'))
                if volume > capacity:
                    total_to_redistribute = max(total_to_redistribute, volume - capacity)
                import pdb; pdb.set_trace()

    total_to_remove = total_to_redistribute
    total_to_add = total_to_redistribute * fraction_to_redistribute
    assert total_to_redistribute > 0, "no vehicles to redistribute"

    logger.info("redistributing traffic")
    for u, v, k, data in tqdm(list(G.edges(keys=True, data=True))):
        if "I270_HCNWA" in data.get('ref', ""):
            if bearings_aligned(data.get('bearing'), i270_target_bearing, 90):
                current_volume = float(data.get('post_induction_vol_vph'))
                new_volume = current_volume - total_to_remove
                data['post_induction_vol_vph'] = new_volume
                data['redistributed_vol'] = total_to_remove * -1
        if "I 25" in str(data.get('ref', "")):
            if bearings_aligned(data.get('bearing'), i25_target_bearing, 90):
                current_volume = float(data.get('post_induction_vol_vph'))
                new_volume = current_volume + total_to_add
                data['post_induction_vol_vph'] = new_volume
                data['redistributed_vol'] = total_to_add
        if "I 70" in str(data.get('ref', "")):
            if bearings_aligned(data.get('bearing'), i25_target_bearing, 90):
                current_volume = float(data.get('post_induction_vol_vph'))
                new_volume = current_volume + total_to_add
                data['post_induction_vol_vph'] = new_volume
                data['redistributed_vol'] = total_to_add

    return G

# TODO generalize to 'create widening scenario'?
def create_denver_cdot_scenario(
        scenario_dir,
        new_scenario_dir,
        induced_demand_annual_vmt = 100000000,
):
    daily_induced_demand_vmt = induced_demand_annual_vmt / 260  # TODO add this to params
    K_factor = 0.12  # TODO add this to params
    peak_hour_induced_demand_vmt = daily_induced_demand_vmt * K_factor

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

                new_capacity_per_lane = 1900 #TODO add this to traffic_params? or some other params?

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

        G = relative_to_absolute_induced_demand(new_scenario_dir, G, peak_hour_induced_demand_vmt)

        G = recalculate_speeds(G)

        # Save modified graph
        logger.info("saving modified road graph")
        G.graph['has_toll'] = True
        os.makedirs(f"{new_scenario_dir}/input_data/traffic", exist_ok=True)
        ox.save_graphml(G, f"{new_scenario_dir}/input_data/traffic/routing_graph.graphml")
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        edges.to_file(f"{new_scenario_dir}/input_data/traffic/routing_edges.gpkg", driver="GPKG")

    logger.info("copying ttms")
    copy_ttms(scenario_dir, new_scenario_dir, modes_to_copy = ['WALK',"BICYCLE","TRANSIT"])

def create_denver_hcnw_scenario(
        scenario_dir,
        new_scenario_dir,
        induced_demand_annual_vmt = 22000000,
):
    daily_induced_demand_vmt = induced_demand_annual_vmt / 260  # TODO add this to params
    K_factor = 0.12  # TODO add this to params
    peak_hour_induced_demand_vmt = daily_induced_demand_vmt * K_factor

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
                data['ref'] = 'I270_HCNWA'
                lanes = float(data.get('lanes'))
                capacity = float(data.get('capacity_vph'))

                new_capacity_per_lane = 1900 #TODO add this to traffic_params? or some other params?

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
        logger.info("saving modified road graph")
        G.graph['has_toll'] = True
        os.makedirs(f"{new_scenario_dir}/input_data/traffic", exist_ok=True)
        ox.save_graphml(G, f"{new_scenario_dir}/input_data/traffic/routing_graph.graphml")
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        edges.to_file(f"{new_scenario_dir}/input_data/traffic/scenario_edges.gpkg", driver="GPKG")

    logger.info("copying ttms")
    copy_ttms(scenario_dir, new_scenario_dir, modes_to_copy = ['WALK',"BICYCLE","TRANSIT"])

def compared_prepare_results(scenario_dir):
    #relies on scenarios and comparison with EC having been run
    comparison_results = gpd.read_file(f"{scenario_dir}/comparison/geometry_comparison.gpkg")
    traffic_edges = gpd.read_file(f"{scenario_dir}/input_data/traffic/scenario_edges.gpkg")

    save_cols_as_numeric = ['ff_speed_kph', 'obs_speed_kph', 'tomtom_sample_size',
                            'tomtom_night_speed_kph', 'tomtom_sample_size_night', 'peak_speed_kph', 'modeled_vol_vph',
                            'forecast_speed_kph', "peak_traversal_time_sec", 'post_induction_vol_vph',
                            'calibration_factor', 'relative_demand', 'induced_volume', 'speed_diff_percent',
                            'previous_peak_speed_kph', 'previous_traversal_time']

    for col in save_cols_as_numeric:
        traffic_edges[col] = traffic_edges[col].astype(float)

    stats = {}

    traffic_edges['scenario_delay_veh_min_per_mile'] = traffic_edges['scenario_delay_veh_min'] / (traffic_edges['length'] / 1609)
    traffic_edges['scenario_delay_veh_hr_per_mile'] = traffic_edges['scenario_delay_veh_min_per_mile'] / 60

    traffic_edges['induced_vmt'] = traffic_edges['induced_volume'] * (traffic_edges['length'] / 1609)
    stats['induced_vmt'] = traffic_edges['induced_vmt'].sum()

    os.makedirs(f"{scenario_dir}/prepared_results/", exist_ok=True)
    traffic_edges.to_file(f"{scenario_dir}/prepared_results/scenario_edges.gpkg", driver="GPKG")

    print(stats)

if __name__ == "__main__":
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
    compared_prepare_results(f"testing/{study_name}/cdot_scenario/")

    communication.compare_scenarios(f"testing/{study_name}", "existing_conditions", "hcnw_scenario")
    compared_prepare_results(f"testing/{study_name}/hcnw_scenario/")

    # create_denver_hcnw_scenario("testing/denver/existing_conditions/",
    #                             "testing/denver/hcnw_scenario/")
    # run_scenario("testing/denver/hcnw_scenario/")
    # communication.compare_scenarios("testing/denver",
    #                                 "existing_conditions",
    #                                 "cdot_scenario")
    # communication.compare_scenarios("testing/denver",
    #                                 "existing_conditions",
    #                                 "hcnw_scenario")

def pl3():
    traffic_datasource = "tomtom"
    volume_datasource = "tmas"
    initialize_existing_conditions(
        scenario_dir="testing/minidenver/existing_conditions/",
        states=["CO"],
        lat=39.805084,
        lon=-104.940186,
        buffer=10000,
        traffic_datasource=traffic_datasource,
        volume_datasource=volume_datasource,
    )
    run_scenario("testing/minidenver/existing_conditions/",
                 track_volumes={
                                "checkpoint_node_ids": None,
                                "checkpoint_edge_attr": "ref",
                                "checkpoint_edge_values": ['I 270']},
                 induced_demand_annual_vmt = 100000000,)
    create_denver_cdot_scenario("testing/minidenver/existing_conditions/", "testing/minidenver/cdot_scenario/")
    run_scenario("testing/minidenver/cdot_scenario/")
    communication.compare_scenarios("testing/minidenver", "existing_conditions", "cdot_scenario")
    # create_denver_cdot_scenario("testing/denver/existing_conditions/",
    #                             "testing/denver/cdot_scenario/")
    # create_denver_hcnw_scenario("testing/denver/existing_conditions/",
    #                             "testing/denver/hcnw_scenario/")
    # run_scenario("testing/denver/cdot_scenario/")
    # run_scenario("testing/denver/hcnw_scenario/")
    # communication.compare_scenarios("testing/denver",
    #                                 "existing_conditions",
    #                                 "cdot_scenario")
    # communication.compare_scenarios("testing/denver",
    #                                 "existing_conditions",
    #                                 "hcnw_scenario")




def test():
    traffic_datasource = "tomtom"
    volume_datasource = "tmas"
    initialize_existing_conditions(
        scenario_dir="testing/lewes/existing_conditions/",
        states=["DE"],
        lat=38.770,
        lon=-75.1444,
        buffer=9000,
        traffic_datasource=traffic_datasource,
        volume_datasource=volume_datasource,
        transcad_source=False,

    )
    run_scenario("testing/lewes/existing_conditions/",
                 track_volumes={
                                "checkpoint_node_ids": None,
                                "checkpoint_edge_attr": "ref",
                                "checkpoint_edge_values": ['DE 1', 'US 9;DE 1']},
                 induced_demand_annual_vmt = 33000000,
                 )
    create_lewes_cdot_scenario("testing/lewes/existing_conditions/", "testing/lewes/cdot_scenario/")
    run_scenario("testing/lewes/cdot_scenario/")
    communication.compare_scenarios("testing/lewes", "existing_conditions", "cdot_scenario")