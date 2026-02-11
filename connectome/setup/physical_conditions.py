
import os
from pathlib import Path
import osmnx as ox
import subprocess
import shapely
import requests
import shapely.geometry
import geopandas as gpd
import pandas as pd
from mobility_db_api import MobilityAPI
from pyrosm import OSM
from datetime import datetime, timedelta
import math
import time
import numpy as np

import zipfile
import tempfile
import shutil

from traffic_utils.speed_utils import conflate_tomtom_to_osm
from traffic_utils.volume_utils import match_tmas_stations_to_graph

from connectome.traffic_utils.volume_utils import infer_volume

_api_keys_dir = Path(__file__).resolve().parent.parent / "api_keys"
os.environ['MOBILITY_API_REFRESH_TOKEN'] = (_api_keys_dir / "mobility_db_refresh_token.txt").read_text().rstrip("\n")
os.environ['MAPBOX_TOKEN'] = (_api_keys_dir / "mapbox_token.txt").read_text().rstrip("\n")

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
    Relies on loading an API refresh key from api_keys/mobility_db_refresh_token.txt
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


### Google Maps based traffic tools ###

def groupings_for_traffic_estimates(analysis_areas: gpd.GeoDataFrame, number_groups: int) -> gpd.GeoDataFrame:
    """Creates population-balanced groupings of analysis areas based on density and contiguity.
    Any remaining areas that could not be grouped by population will be assigned to the closest group.

    Args:
        analysis_areas: GeoDataFrame containing polygons with population data
        number_groups: Number of groups to create

    Returns:
        GeoDataFrame with new 'traffic_group' column indicating group assignment
    """
    # Calculate initial target population per group
    total_population = analysis_areas['census_total_pop'].sum()
    target_pop = total_population / number_groups
    print(f"Target population per group: {target_pop:.0f}")

    # Initialize tracking variables
    remaining_areas = set(analysis_areas.index)
    too_small_areas = set()
    analysis_areas['traffic_group'] = -1  # -1 indicates unassigned
    current_group = 0

    while len(remaining_areas) > 0 and current_group < number_groups:
        # Find area with highest density that isn't grouped yet
        # 1) Restrict to remaining areas
        densities = analysis_areas.loc[list(remaining_areas), 'population_per_sqkm']

        # 2) Coerce to numeric and drop NULL/non-numeric values from contention
        densities_numeric = pd.to_numeric(densities, errors='coerce').dropna()

        if densities_numeric.empty:
            raise ValueError(
                "No remaining areas with valid numeric 'population_per_sqkm' to use as a seed "
                "(e.g., only NULL-density tracts like parks/airports are left)."
            )

        seed_area = densities_numeric.idxmax()

        # Start new group with this seed area
        analysis_areas.loc[seed_area, 'traffic_group'] = current_group
        current_pop = analysis_areas.loc[seed_area, 'census_total_pop']
        remaining_areas.remove(seed_area)

        create_group = True

        # Keep adding nearest neighbors until target population reached or no more contiguous areas
        while current_pop < target_pop and len(remaining_areas) > 0:
            # Find contiguous areas not yet in a group
            contiguous = set()
            for area_idx in analysis_areas[analysis_areas['traffic_group'] == current_group].index:
                area_geom = analysis_areas.loc[area_idx, 'geometry']
                touching = analysis_areas.loc[list(remaining_areas)].touches(area_geom)
                contiguous.update(touching[touching].index)

            if not contiguous:
                print(f"No more contiguous areas for traffic_group {current_group}")

                # Check if current group is too small (less than 30% of target)

                if current_pop < (target_pop * 0.3):
                    print(f"Group {current_group} is too small ({current_pop:.0f} < {target_pop * 0.3:.0f})")

                    # Reset areas in current group to unassigned
                    small_group_areas = analysis_areas[analysis_areas['traffic_group'] == current_group].index
                    analysis_areas.loc[small_group_areas, 'traffic_group'] = -1
                    too_small_areas.update(small_group_areas)

                    create_group = False
                break

            # Add nearest contiguous area - find centroid of current group
            current_group_geoms = analysis_areas[analysis_areas['traffic_group'] == current_group]
            group_centroid = current_group_geoms.geometry.unary_union.centroid
            
            next_area = min(contiguous,
                            key=lambda x: group_centroid.distance(
                                analysis_areas.loc[x, 'geometry'].centroid))

            analysis_areas.loc[next_area, 'traffic_group'] = current_group
            current_pop += analysis_areas.loc[next_area, 'census_total_pop']
            remaining_areas.remove(next_area)

        if create_group:
            print(
                f"Created group {current_group} with {len(analysis_areas[analysis_areas['traffic_group'] == current_group])} areas and population {current_pop}")

            # Only increment group and recalculate if we didn't reset
            current_group += 1

        # Recalculate target for remaining groups
        if len(remaining_areas) > 0:
            remaining_pop = analysis_areas.loc[list(remaining_areas), 'census_total_pop'].sum()
            remaining_groups = number_groups - current_group
            if remaining_groups > 0:  # Avoid division by zero
                target_pop = remaining_pop / remaining_groups
                print(f"New target population per traffic_group: {target_pop:.0f}")
            else:
                print("No more groups available, will assign remaining areas to nearest groups")

    remaining_areas = remaining_areas.union(too_small_areas)
    if len(remaining_areas) > 0:
        print(f"Assigning {len(remaining_areas)} remaining areas to closest groups")
        for area in remaining_areas:
            # Get centroid of current area
            area_centroid = analysis_areas.loc[area, 'geometry'].centroid

            # Find centroids of all existing groups
            group_centroids = {}
            for group in range(current_group):
                group_geoms = analysis_areas[analysis_areas['traffic_group'] == group]
                group_centroids[group] = group_geoms.geometry.union_all().centroid

            # Find closest group by checking adjacent polygons
            current_area_geom = analysis_areas.loc[area, 'geometry']
            adjacent_areas = []

            # Find all adjacent polygons that are already grouped
            for other_area in analysis_areas[analysis_areas['traffic_group'] != -1].index:
                if current_area_geom.touches(analysis_areas.loc[other_area, 'geometry']):
                    adjacent_areas.append(other_area)

            if adjacent_areas:
                # Find the adjacent polygon with closest centroid
                closest_adjacent = min(adjacent_areas,
                                       key=lambda x: area_centroid.distance(
                                           analysis_areas.loc[x, 'geometry'].centroid))
                closest_group = analysis_areas.loc[closest_adjacent, 'traffic_group']
            else:
                # If no adjacent polygons, fall back to centroid distance to group
                closest_group = min(group_centroids.keys(),
                                    key=lambda g: area_centroid.distance(group_centroids[g]))

            # Assign to closest group
            analysis_areas.loc[area, 'traffic_group'] = closest_group

    return analysis_areas


def select_traffic_samples_by_group(analysis_areas: gpd.GeoDataFrame,
                                    spots_per_group: int) -> gpd.GeoDataFrame:
    """Select traffic sample locations for each traffic group.

    Args:
        analysis_areas: GeoDataFrame with traffic_group column
        spots_per_group: Number of sample locations to select per group

    Returns:
        GeoDataFrame with traffic_sample_selection column
    """
    for group in analysis_areas['traffic_group'].unique():
        group_areas = analysis_areas[analysis_areas['traffic_group'] == group].copy()
        selected = select_traffic_sample_polygons(group_areas,
                                                  spots_per_group)
        analysis_areas.loc[selected.index, 'traffic_sample_selection'] = selected['traffic_sample_selection']
    return analysis_areas


def select_traffic_sample_polygons(group_gdf: gpd.GeoDataFrame,
                                   num_to_select: int,
                                   interior_bonus = 2,) -> gpd.GeoDataFrame:
    """Select sample polygons from a contiguous group, prioritizing interior locations.

    First selects the highest-population interior polygon, then selects additional
    polygons to maximize distance from already-selected ones, preferring interior polygons.

    Args:
        group_gdf: GeoDataFrame of contiguous polygons
        num_to_select: Number of polygons to select

    Returns:
        GeoDataFrame with added 'traffic_sample_selection' column
    """
    group_gdf['traffic_sample_selection'] = False

    # Identify interior polygons (surrounded by others in the group)
    group_boundary = group_gdf.geometry.unary_union.boundary
    interior_mask = ~group_gdf.geometry.touches(group_boundary)
    interior_indices = group_gdf[interior_mask].index

    # First selection: highest population interior polygon
    if len(interior_indices) > 0:
        first_selection = group_gdf.loc[interior_indices, 'census_total_pop'].idxmax()
    else:
        # Fallback if no interior polygons exist
        first_selection = group_gdf['census_total_pop'].idxmax()

    group_gdf.loc[first_selection, 'traffic_sample_selection'] = True
    selected_indices = [first_selection]

    # Subsequent selections: maximize distance from selected, prefer interior
    for _ in range(num_to_select - 1):
        remaining = group_gdf[~group_gdf['traffic_sample_selection']].index
        if len(remaining) == 0:
            break

        # Calculate minimum distance to any selected polygon
        selected_centroids = group_gdf.loc[selected_indices, 'geometry'].centroid
        min_distances = {}

        for idx in remaining:
            candidate_centroid = group_gdf.loc[idx, 'geometry'].centroid
            distances = [candidate_centroid.distance(sc) for sc in selected_centroids]
            min_distances[idx] = min(distances)

        # Prefer interior polygons by boosting their effective distance
        for idx in remaining:
            if idx in interior_indices:
                min_distances[idx] *= interior_bonus  # 50% bonus for interior locations

        # Select polygon with maximum distance
        next_selection = max(min_distances, key=min_distances.get)
        group_gdf.loc[next_selection, 'traffic_sample_selection'] = True
        selected_indices.append(next_selection)

    return group_gdf


def get_next_wednesday_9am():
    """Get the datetime for next Wednesday at 9am local time.
    
    Returns:
        datetime: Next Wednesday at 9:00 AM
    """
    now = datetime.now()
    days_ahead = 2 - now.weekday()  # Wednesday is 2
    if days_ahead <= 0 or (days_ahead == 0 and now.hour >= 9):
        # Target day already happened this week or happening now, get next week
        days_ahead += 7
    next_wednesday = now + timedelta(days=days_ahead)
    # Set to 9:00 AM
    next_wednesday = next_wednesday.replace(hour=9, minute=0, second=0, microsecond=0)
    return next_wednesday


### Mapbox based traffic tools ###

def benchmark_driving_times(analysis_areas: gpd.GeoDataFrame,
                           mapbox_token: str,
                           save_to: str,
                           chunk_size: int = 5,
                           delay_between_requests: float = 0.1) -> pd.DataFrame:
    """Benchmark actual driving travel times using Mapbox Matrix API.
    
    Uses the mapbox/driving-traffic profile to get real-world travel times
    between selected traffic sample points. Automatically chunks requests
    to handle the 10-coordinate limit for driving-traffic profile.
    
    Args:
        analysis_areas: GeoDataFrame with traffic_sample_selection column
        mapbox_token: Mapbox API access token
        save_to: Path to save the results CSV
        chunk_size: Size of chunks for matrix requests (default 5 for 5x5 matrices)
        delay_between_requests: Seconds to wait between API calls (rate limiting)
    
    Returns:
        DataFrame with columns: origin_id, destination_id, duration_seconds, distance_meters
        
    Raises:
        ValueError: If more than 44 sample points are selected
    """
    # Filter to traffic sample points
    sample_points = analysis_areas[analysis_areas['traffic_sample_selection'] == True].copy()
    
    if len(sample_points) == 0:
        raise ValueError("No points with traffic_sample_selection==True found")
    
    if len(sample_points) > 44:
        raise ValueError(
            f"Too many sample points ({len(sample_points)}). "
            f"Maximum is 44 to stay under ~2,000 coordinate pairs."
        )
    
    print(f"Benchmarking driving times for {len(sample_points)} sample points")
    
    # Get centroids as coordinates
    sample_points['centroid'] = sample_points.geometry.to_crs('EPSG:4326').centroid
    sample_points['lon'] = sample_points['centroid'].x
    sample_points['lat'] = sample_points['centroid'].y
    
    # Get departure time (next Wednesday at 9am)
    departure_time = get_next_wednesday_9am()
    departure_time_str = departure_time.strftime("%Y-%m-%dT%H:%M")
    print(f"Using departure time: {departure_time_str}")
    
    # Prepare to collect results
    all_results = []
    
    # Create chunks of points
    point_ids = list(sample_points.index)
    n_points = len(point_ids)
    
    # Calculate number of chunks needed
    n_chunks = math.ceil(n_points / chunk_size)
    print(f"Making {n_chunks * n_chunks} API requests ({n_chunks}x{n_chunks} chunks)")
    
    # Process in chunks
    request_count = 0
    for i in range(n_chunks):
        origin_start = i * chunk_size
        origin_end = min((i + 1) * chunk_size, n_points)
        origin_chunk = point_ids[origin_start:origin_end]
        
        for j in range(n_chunks):
            dest_start = j * chunk_size
            dest_end = min((j + 1) * chunk_size, n_points)
            dest_chunk = point_ids[dest_start:dest_end]
            
            # Build coordinate string
            # Combine origins and destinations (API requires all coords in one list)
            all_coords = origin_chunk + dest_chunk
            
            # Remove duplicates while preserving order
            seen = set()
            unique_coords = []
            for coord_id in all_coords:
                if coord_id not in seen:
                    seen.add(coord_id)
                    unique_coords.append(coord_id)
            
            # Build coordinate string for API
            coord_strings = []
            for coord_id in unique_coords:
                lon = sample_points.loc[coord_id, 'lon']
                lat = sample_points.loc[coord_id, 'lat']
                coord_strings.append(f"{lon},{lat}")
            
            coordinates = ";".join(coord_strings)
            
            # Map original indices to positions in unique_coords
            origin_indices = [unique_coords.index(oid) for oid in origin_chunk]
            dest_indices = [unique_coords.index(did) for did in dest_chunk]
            
            # Build API request
            sources_param = ";".join(str(idx) for idx in origin_indices)
            destinations_param = ";".join(str(idx) for idx in dest_indices)
            
            url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving-traffic/{coordinates}"
            params = {
                'sources': sources_param,
                'destinations': destinations_param,
                'annotations': 'distance,duration',
                #'depart_at': departure_time_str,
                'access_token': mapbox_token
            }
            
            # Make request
            request_count += 1
            print(f"Request {request_count}/{n_chunks * n_chunks}: "
                  f"Origins {origin_start}-{origin_end-1}, Destinations {dest_start}-{dest_end-1}")


            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                import pdb; pdb.set_trace()
                raise RuntimeError(f"Mapbox API request failed: {response.text}")
            
            data = response.json()
            
            # Parse results
            durations = data.get('durations', [])
            distances = data.get('distances', [])
            
            # Store results
            for origin_idx, origin_id in enumerate(origin_chunk):
                for dest_idx, dest_id in enumerate(dest_chunk):
                    duration = durations[origin_idx][dest_idx]
                    distance = distances[origin_idx][dest_idx]
                    
                    # Mapbox returns null for unreachable destinations
                    if duration is not None and distance is not None:
                        all_results.append({
                            'origin_id': origin_id,
                            'destination_id': dest_id,
                            'duration_seconds': duration,
                            'distance_meters': distance
                        })
            
            # Rate limiting delay
            if delay_between_requests > 0 and request_count < n_chunks * n_chunks:
                time.sleep(delay_between_requests)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"Collected {len(results_df)} origin-destination pairs")
    
    # Save results
    results_df.to_csv(save_to, index=False)
    print(f"Saved results to {save_to}")
    
    return results_df

### High level API function ###


def physical_conditions(scenario_dir,
                        traffic_datasource = None, #None, "mapbox", or "tomtom"
                        volume_datasource = None, #None or "tmas"
                        transcad_source = True,
                        ):
    input_dir = f"{scenario_dir}/input_data/"
    analysis_areas = gpd.read_file(f"{input_dir}/analysis_areas.gpkg")
    analysis_areas.index = analysis_areas['geom_id'].values

    # Get OSM data if it doesn't exist
    if not os.path.exists(f"{input_dir}/osm_study_area.pbf"):
        print("preparing osm data")
        download_osm(
            analysis_areas,
            f"{input_dir}/osm_large_file.pbf",
            f"{input_dir}/osm_large_file_filtered.pbf",
            f"{input_dir}/osm_study_area.pbf",
            buffer_dist=500,
        )
        make_osm_editable(f"{input_dir}/osm_study_area.pbf", f"{input_dir}/osm_study_area_editable.osm")


    #todo consider function calls here to interpolate capacities and volumes?

    if traffic_datasource == "tomtom":
        os.makedirs(f"{input_dir}/traffic/", exist_ok=True)
        if not (os.path.exists(f"{input_dir}/tomtom/tomtom_geom.geojson") and
             os.path.exists(f"{input_dir}/tomtom/tomtom_speeds.json")):
            raise FileNotFoundError(f"Tomtom input data not found in {input_dir}/tomtom/tomtom_geom.geojson and tomtom_speeds.json. Please download.")


        if not os.path.exists(f"{input_dir}/traffic/routing_graph.graphml"):
            G, _, _, _, _ =conflate_tomtom_to_osm(
               scenario_dir,
                f"{input_dir}/tomtom/tomtom_speeds.json",
                f"{input_dir}/tomtom/tomtom_geom.geojson",
                tomtom_night_stats_path = f"{input_dir}/tomtom/tomtom_night_speeds.json",
                tomtom_night_geom_path = f"{input_dir}/tomtom/tomtom_night_geom.geojson",
                debug_gpkg_prefix = f"{input_dir}/tomtom/debug",
            )
        else:
            G = ox.load_graphml(f"{scenario_dir}/input_data/traffic/routing_graph.graphml")

    # if bool(transcad_source):
    #     conflate_transcad_to_osm_graph(
    #         G,
    #         scenario_dir,
    #         f"{input_dir}transcad/AM2.shp",
    #         f"{input_dir}transcad/AM2.xlsx",
    #     )



    # Get GTFS data if it doesn't exist
    if (
            not os.path.exists(f"{input_dir}/GTFS/")# or
            #not any(p.suffix == ".zip" for p in Path(f"{input_dir}/GTFS/").iterdir())
    ):
        print("downloading GTFS data")
        get_GTFS_from_mobility_database(analysis_areas,
                                        f"{input_dir}/GTFS/",
                                        0.2)