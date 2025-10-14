
import os
import osmnx as ox
import subprocess
import shapely
import requests
import shapely.geometry
import geopandas as gpd
from mobility_db_api import MobilityAPI
from pyrosm import OSM

import zipfile
import tempfile
import shutil


os.environ['MOBILITY_API_REFRESH_TOKEN'] = open("mobility_db_refresh_token.txt").read().rstrip("\n")

def download_osm(geometry: gpd.GeoDataFrame, #unbuffered
                save_to_unclipped: str,
                save_to_unclipped_filtered:str,
                save_to_clipped: str,
                buffer_dist: float = 2000, #meters
                approach = "geofabrik",
                ):
    print('preparing OSM')
    polygon_unbuffered = geometry.union_all()
    poly_unbuff_utm, utm_crs = ox.projection.project_geometry(polygon_unbuffered)
    geom_buffered_utm = shapely.buffer(poly_unbuff_utm, buffer_dist)
    geom_buffered_latlon = ox.projection.project_geometry(geom_buffered_utm, crs=utm_crs, to_latlong=True)[0]
    print(geom_buffered_latlon.bounds)
    minx, miny, maxx, maxy = geom_buffered_latlon.bounds
    if approach == "pyrosm":
        osm = OSM(save_to_unclipped)
        custom_filter = {
            'highway': [
                'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                'residential', 'living_street', 'service', 'unclassified',
                'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
                'path', 'footway', 'cycleway', 'bridleway', 'steps', 'track'
            ]
        }
        network = OSM.get_network(custom_filter=custom_filter)
        network = network[network.geometry.intersects(geom_buffered_latlon)]
        network.to_file(save_to_clipped)
        #TODO check that this works

    if approach == "geofabrik": 
        if not os.path.exists(save_to_unclipped): #download from Geofabrik
            print("attempting to download from Germany")
            bbox_poly = shapely.geometry.box(minx, miny, maxx, maxy)
    
            # Load Geofabrik index with geometries
            index_url = "https://download.geofabrik.de/index-v1.json"
            resp = requests.get(index_url)
            resp.raise_for_status()
            data = resp.json()
    
            # Convert to GeoDataFrame
            features = []
            for feat in data["features"]:
                poly = shapely.geometry.shape(feat["geometry"])
                props = feat["properties"]
                features.append({"geometry": poly, **props})
    
            geofabrik_gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    
            # Filter regions that contain the bbox
            candidates = geofabrik_gdf[geofabrik_gdf.contains(bbox_poly)]
    
            if candidates.empty:
                raise ValueError("No Geofabrik region fully contains the bounding box!")
    
            # Pick the smallest by polygon area
            smallest = candidates.iloc[candidates.area.argmin()]
    
            region_url = smallest["urls"]["pbf"]  # direct .osm.pbf link
            region_name = smallest["id"]
            print(f"Selected region: {region_name}")
            print(f"Download URL: {region_url}")
    
            # Download the .osm.pbf file
            if not os.path.exists(save_to_unclipped):
                print(f"Downloading {region_url} ...")
                r = requests.get(region_url, stream=True)
                r.raise_for_status()
                with open(save_to_unclipped, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                print(f"Saved to {save_to_unclipped}")
    
    #    geom_in_geojson_forosm = geojson.Feature(geometry=geom_buffered_latlon, properties={})
    #    with open(scenario+'/boundaries_forosm.geojson', 'w') as out:
    #        out.write(json.dumps(geom_in_geojson_forosm))
    
        if not os.path.exists(save_to_unclipped):
            print(f'''
                  ERROR: No .osm.pbf file found nor autodownloaded.
                  To measure a connectome, we need data from OpenStreetMap
                  that covers the entire geographic analysis area.
                  Download a .osm.pbf file from https://download.geofabrik.de/
                  (preferably the smallest one that covers your area)
                  And put that file in {save_to_unclipped}.
                  ''')
            raise ValueError
    
        #crop OSM
        cmd = [
            "osmium",
            "tags-filter",
            str(save_to_unclipped),
            "w/highway",
            "r/highway",
            "-o", str(save_to_unclipped_filtered),
            "--overwrite"
        ]

        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Filtered file saved to: {save_to_unclipped_filtered}")
        except subprocess.CalledProcessError as e:
            print(f"❌ osmium command failed: {e}")
    
        cmd = [
            "osmium", "extract",
            f"--bbox={minx},{miny},{maxx},{maxy}",
            "--overwrite",
            f"--output={save_to_clipped}",
            f"{save_to_unclipped_filtered}"
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    
        if os.path.getsize(save_to_clipped) < 300:
            print(f'''
                  ERROR: The OSM file you provided does not seem to 
                  include your study area.
                  ''')
            raise ValueError
            #add LTS tags for biking
    
    
    
    # print('adding bike LTS tags')
    # prep_bike_osm.add_lts_tags(scenario + "/study_area.pbf",
    #                            scenario +"/study_area_LTS.pbf")
    #make a JOSM-editable .osm file for users to make new scenarios

def make_osm_editable(pbf_file, osm_file):
    command = f"osmconvert {pbf_file} -o={osm_file}"
    subprocess.check_call(command.split(' '))


def sanitize_gtfs_file(gtfs_path):
    """
    Remove empty optional GTFS tables that cause r5py to fail.

    Args:
        gtfs_path: Path to the GTFS .zip file

    Returns:
        Path to the sanitized GTFS file (same as input, modified in place)
    """
    # Tables that can be empty and cause issues
    optional_tables = [
        'fare_attributes.txt',
        'fare_rules.txt',
        'frequencies.txt',
        'transfers.txt'
    ]

    print(f"Sanitizing GTFS file: {os.path.basename(gtfs_path)}")

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip = os.path.join(temp_dir, 'temp.zip')

        # Track which files to keep
        files_to_remove = []

        # Check which optional tables are empty
        with zipfile.ZipFile(gtfs_path, 'r') as zip_read:
            for table in optional_tables:
                if table in zip_read.namelist():
                    # Read the file
                    with zip_read.open(table) as f:
                        content = f.read().decode('utf-8').strip()
                        lines = content.split('\n')

                        # Check if file only contains header or is empty
                        if len(lines) <= 1 or (len(lines) == 2 and lines[1].strip() == ''):
                            files_to_remove.append(table)
                            print(f"  - Removing empty table: {table}")

        # If there are files to remove, create a new zip without them
        if files_to_remove:
            with zipfile.ZipFile(gtfs_path, 'r') as zip_read:
                with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zip_write:
                    for item in zip_read.namelist():
                        if item not in files_to_remove:
                            data = zip_read.read(item)
                            zip_write.writestr(item, data)

            # Replace original file with sanitized version
            shutil.move(temp_zip, gtfs_path)
            print(f"  ✓ Sanitized {len(files_to_remove)} empty table(s)")
        else:
            print(f"  ✓ No empty tables found")

    return gtfs_path


def get_GTFS_from_mobility_database(
        geometry: gpd.GeoDataFrame, #unbuffered
        save_to_dir: str,
        min_overlap: float = 0.001,
        include_no_bbox: bool = False,
        sanitize_for_r5py: bool = True
        ):
    """
    Downloads GTFS feeds from the Mobility Database API and saves them to the specified directory.
    Relies on loading an API refresh key from mobility_db_refresh_token.txt
    Args:
        geometry:
        save_to_dir:
        min_overlap: Minimum overlap between feed bbox and geometry to justify inclusion. Default 0.1%.
    """
    os.makedirs(save_to_dir, exist_ok=True)

    geometry_ll = geometry.to_crs(4326)
    study_poly = geometry_ll.union_all()  # merge into single polygon
    minx, miny, maxx, maxy = study_poly.bounds

    api = MobilityAPI()
    access_token = api.get_access_token()

    # build query
    # it seems like the MobilityAPI python library doesn't support queries by geometry :(

    url = "https://api.mobilitydatabase.org/v1/gtfs_feeds"
    params = {
        "limit": 100,
        "offset": 0,
        "dataset_latitudes": f"{miny},{maxy}",
        "dataset_longitudes": f"{minx},{maxx}",
        "bounding_filter_method": "partially_enclosed"
    }
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, params=params, headers=headers)

    # Check status and parse JSON
    if response.status_code == 200:
        data = response.json()
        print("called list of GTFS providers")
    else:
        print(f"Error in MobilityDatabase call {response.status_code}: {response.text}")
        print("Exiting without GTFS download")
        return

    # Filter by overlap
    selected_feeds = []
    feeds_without_bbox = []
    for feed in data:
        print(f"checking overlap with {feed['provider']}")
        bbox = feed['latest_dataset']['bounding_box']
        if bbox is None:
            print("no bbox found")
            feeds_without_bbox.append(feed)
            continue

        # Build feed polygon from bbox
        min_lat, max_lat = bbox["minimum_latitude"], bbox["maximum_latitude"]
        min_lon, max_lon = bbox["minimum_longitude"], bbox["maximum_longitude"]
        feed_poly = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)

        # Compute overlap

        intersection = study_poly.intersection(feed_poly)
        if not intersection.is_empty:
            overlap_ratio = intersection.area / feed_poly.area
            print('intersection is not empty. Overlap_ratio:', overlap_ratio)
            if overlap_ratio >= min_overlap:
                print("overlap is sufficient. including feed")
                selected_feeds.append(feed)
            else:
                print("overlap is insufficient. skipping feed")


    if len(selected_feeds) > 0:
        print(f"{len(selected_feeds)} feeds overlap ≥{min_overlap}")
    else:
        print("no overlapping GTFS files found!")
    if len(feeds_without_bbox) > 0:
        print(f"bbox not found for {[feed['provider'] for feed in feeds_without_bbox]} -- you might want to check if they're relevant")
        if include_no_bbox:
            print ("including those feeds without bbox")
            selected_feeds += feeds_without_bbox

    # Download selected feeds

    for feed in selected_feeds:
        print("feed for", feed['provider'])
        feed_id = feed['latest_dataset']["id"]
        download_url = feed['latest_dataset']['hosted_url']
        if not download_url:
            print('not download_url')
            continue

        outfile = f"{save_to_dir}{feed_id}_{feed['provider']}.zip"
        if os.path.exists(outfile):
            print(f"Already downloaded {feed_id}")
            continue

        print(f"Downloading {feed_id} from {download_url}")
        r = requests.get(download_url, stream=True)
        r.raise_for_status()
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        if sanitize_for_r5py == True:
            sanitize_gtfs_file(outfile)

    print("Download complete")
