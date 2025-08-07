import geopandas as gpd
from tqdm import tqdm
import geojson
import json
import os
import subprocess
import prep_bike_osm
import requests
import shapely.geometry
import string



def prepare_osm(polygon, scenario="existing_conditions"):
    print('preparing OSM')
    geom_in_geojson_forosm = geojson.Feature(geometry=polygon, properties={})
    with open(scenario+'/boundaries_forosm.geojson', 'w') as out:
        out.write(json.dumps(geom_in_geojson_forosm))
    osmfiles = []
    files = os.listdir(f"{scenario}/")
    for file in files:
        if file[-4:] == '.pbf':
            osmfiles.append(file)
    if len(osmfiles) == 0:
        print(f'''
              ERROR: No .osm.pbf file found.
              To measure a connectome, we need data from OpenStreetMap
              that covers the entire geographic analysis area.
              Download a .osm.pbf file from https://download.geofabrik.de/
              (preferably the smallest one that covers your area)
              And put that file in {scenario}.
              ''')
        raise ValueError
    elif len (osmfiles) > 1:
        print(f'''
              ERROR: More than one .osm.pbf file found in {scenario}.
              There should only be one file there from which to 
              extract the data for the connectome measurement.
              ''')
        raise ValueError
    #crop OSM
    osmfile = scenario + '/' + osmfiles[0]
    command = f"osmium extract {osmfile} -p {scenario}/boundaries_forosm.geojson -s complete_ways -v -o {scenario}/study_area.pbf"
    subprocess.check_call(command.split(' '))
    if os.path.getsize(f'{scenario}/study_area.pbf') < 300:
        print(f'''
              ERROR: The OSM file you provided does not seem to 
              include your study area.
              ''')
        raise ValueError
    #add LTS tags for biking
    #TODO -- move to pre-scenario-run?
    print('adding bike LTS tags')
    prep_bike_osm.add_lts_tags(scenario + "/study_area.pbf",
                               scenario +"/study_area_LTS.pbf")
    #make a JOSM-editable .osm file for users to make new scenarios
    command = f"osmconvert {scenario}/study_area_LTS.pbf -o={scenario}/study_area_LTS_editable.osm"
    subprocess.check_call(command.split(' '))
    
def get_GTFS_from_mobility_database(polygon, scenario):
    files = os.listdir(scenario)
    if not 'sources.csv' in files:
        url = 'https://bit.ly/catalogs-csv'
        r = requests.get(url, allow_redirects=True)  # to get content after redirection
        with open(f'{scenario}/sources.csv', 'wb') as f:
            f.write(r.content)
    sources = gpd.read_file(f'{scenario}/sources.csv')
    filenames = []
    for idx in tqdm(list(sources.index)):
        if not sources.loc[idx,'location.bounding_box.minimum_longitude'] == '':
            sources.loc[idx,'geometry'] = shapely.geometry.box(
                float(sources.loc[idx,'location.bounding_box.minimum_longitude']),
                float(sources.loc[idx,'location.bounding_box.minimum_latitude']),
                float(sources.loc[idx,'location.bounding_box.maximum_longitude']),
                float(sources.loc[idx,'location.bounding_box.maximum_latitude']),
                )
            if sources.loc[idx,'geometry'].intersects(polygon):
                overlap = sources.loc[idx,'geometry'].intersection(polygon)
                if overlap.area * 1000 > sources.loc[idx,'geometry'].area:
                    url = sources.loc[idx,'urls.latest']
                    name = sources.loc[idx,'provider']
                    if sources.loc[idx,'name'] != '':
                        name = name+'_'+ sources.loc[idx,'name']
                    name = name.translate(str.maketrans('', '', string.punctuation))
                    name = name.replace(' ','_')
                    if name != '' and url != '':
                        filename=name+'.zip'
                        filenames.append(filename)
                        r = requests.get(url, allow_redirects=True)  # to get content after redirection
                        with open(f'{scenario}/{filename}', 'wb') as f:
                            f.write(r.content)
    return filenames