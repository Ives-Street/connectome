import os
import datetime

import sys
sys.argv.append(["--max-memory", "80%"])

import pandas as pd
import geopandas as gpd
from r5py import TransportNetwork, TravelTimeMatrixComputer, DetailedItinerariesComputer
from r5py import TransitMode, LegMode
import osmnx as ox


'''
todo

one day, I'll have to run this many times, 
probably even once per subdemo category,
in order to properly account for things like road pricing, transit fares,
physical ability, willingness to cycle, etc

actually, I should do that soon, for physical ability and willingness to cycle,
just in order to get the logic set up

'''


def prepare_mode_settings(**kwargs):
    mode_settings = {}
    general_settings = {
        'departure': datetime.datetime(2025,1,29,8,30),
        #'departure_time_window': datetime.timedelta(hours=1), #this is the default
        #'percentiles': [50], #this is the default
        'max_time':datetime.timedelta(hours=2),
        'max_time_walking':datetime.timedelta(hours=2),
        'max_time_cycling':datetime.timedelta(hours=2),
        'max_time_driving':datetime.timedelta(hours=2),
        'speed_walking':4.7,
        'speed_cycling':12.0,
        'max_public_transport_rides':4,
    }
    general_settings.update(kwargs)
    
    walk_settings = general_settings.copy()
    walk_settings.update({
        'transport_modes':[LegMode.WALK],
        'access_modes':[LegMode.WALK],
        })
    mode_settings['WALK'] = walk_settings
    
    transit_settings = general_settings.copy()
    transit_settings.update({
        'transport_modes':[TransitMode.TRANSIT],
        'access_modes':[LegMode.WALK],
         })
    mode_settings['TRANSIT'] = transit_settings
    
    bike_lts1_settings = general_settings.copy()
    bike_lts1_settings.update({
        'transport_modes':[LegMode.WALK, LegMode.BICYCLE],
        'access_modes':[LegMode.WALK, LegMode.BICYCLE],
        'max_time_walking':datetime.timedelta(minutes=10),
        'speed_walking':4,
        'max_bicycle_traffic_stress':1
        })
    mode_settings['BIKE_LTS1'] = bike_lts1_settings
    
    bike_lts2_settings = general_settings.copy()
    bike_lts2_settings.update({
         'transport_modes':[LegMode.WALK, LegMode.BICYCLE],
         'access_modes':[LegMode.WALK, LegMode.BICYCLE],
         'max_time_walking':datetime.timedelta(minutes=10),
         'speed_walking':4,
         'max_bicycle_traffic_stress':2
         })
    mode_settings['BIKE_LTS2'] = bike_lts2_settings
    
    bike_lts4_settings = general_settings.copy()
    bike_lts4_settings.update({
        'transport_modes':[LegMode.WALK, LegMode.BICYCLE],
        'access_modes':[LegMode.WALK, LegMode.BICYCLE],
        'max_time_walking':datetime.timedelta(minutes=10),
        'speed_walking':4,
        'max_bicycle_traffic_stress':4
        })
    mode_settings['BIKE_LTS4'] = bike_lts4_settings
    
    car_settings = general_settings.copy()
    car_settings.update({
        'transport_modes':[LegMode.CAR],
        'access_modes':[LegMode.CAR],
        })
    mode_settings['CAR'] = car_settings
    
    return mode_settings



def route_scenario(folder_name, 
                   mode_settings=prepare_mode_settings(),
                   modes = None,
                   ):
    start_time = datetime.datetime.now()
    if not folder_name[:10] == 'scenarios/':
        folder_name = 'scenarios/'+folder_name
    if not folder_name[-1:] == '/':
        folder_name = folder_name +'/'
    print('routing', folder_name)
    files = os.listdir(folder_name)
    gtfs_files = []
    osm_files = []
    for filename in files:
        if filename[-4:] == '.pbf':
            osm_files.append(folder_name+filename)
        if filename[-4:] == '.zip':
            gtfs_files.append(folder_name+filename)
    if len(osm_files) < 1:
        print(f'osm .pbf file not found in {folder_name}')
        raise ValueError
    if len(osm_files) > 1:
        print(f'more than one osm .pbf file found in {folder_name}')
        raise ValueError
    if len(gtfs_files) < 1:
        print('WARNING: no GTFS files found')
    pop_polys = gpd.read_file(folder_name+'population_with_dests.geojson')
    polys_utm = ox.project_gdf(pop_polys)
    pop_points_utm = gpd.GeoDataFrame(
        {'id':polys_utm.id,
         'population':polys_utm.population},
        geometry = polys_utm.centroid,
        crs=polys_utm.crs)
    pop_points = pop_points_utm.to_crs(4326)
    print('building network')
    scenario_transport_network = TransportNetwork(
        osm_files[0],
        gtfs_files
        )
    #detailed transit
    # print('computing detailed transit')
    # detailed_ttm_computer = DetailedItinerariesComputer(
    #     scenario_transport_network, 
    #     origins = pop_points,
    #     destinations = pop_points,
    #     **mode_settings['TRANSIT'],
    #     )
    # travel_time_matrix_detailed = detailed_ttm_computer.compute_travel_times()
    # travel_time_matrix_detailed.to_pickle(f'{folder_name}/results/ttmd_transit.pkl')
    if modes == None:
        modes = list(mode_settings.keys())
    for mode in modes:
        print(f'computing for {mode}')
        ttm_computer = TravelTimeMatrixComputer(
            scenario_transport_network, 
            pop_points,
            **mode_settings[mode]
            )
        ttm_long = ttm_computer.compute_travel_times()
        ttm = pd.pivot(ttm_long, index='from_id', columns='to_id', values='travel_time')
        ttm.to_csv(f'{folder_name}/results/{mode}_ttm.csv')
    
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print(elapsed,'elapsed')
    

def clean_scenarios():
    for sc in os.listdir('scenarios/'):
        for file in os.listdir(f'scenarios/{sc}/results/'):
            os.remove(f'scenarios/{sc}/results/{file}')
            
            