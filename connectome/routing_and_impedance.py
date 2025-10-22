import os
import json
import zipfile
import shutil
import tempfile

import pandas as pd
import numpy as np
import r5py
import datetime
import geopandas as gpd
from tqdm import tqdm
import fiona
from shapely.geometry import shape

import communication


MODES = [ #todo - make this universal for the whole codebase
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
]
#
DEPARTURE_TIME = datetime.datetime.now().replace(hour=9, minute=0, second=0) + datetime.timedelta(
    days=(1 - datetime.datetime.now().weekday() + 7) % 7 + (1 if datetime.datetime.now().weekday() >= 3 else 0))

def route_environment(routeenv, mode, scenario_dir, rep_points, departure_time):

    # check if we already have the unprocessed ttm for this routeenv and mode
    raw_ttm_filename = f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv"
    if os.path.exists(raw_ttm_filename):
        ttm_wide = pd.read_csv(raw_ttm_filename, index_col=0)
        ttm_wide.index = ttm_wide.index.astype(str)
        ttm_wide.columns = ttm_wide.columns.astype(str)
        return ttm_wide

    print(f"routing {mode} for {routeenv}")
    gtfs_files = os.listdir(f"{scenario_dir}/routing/{routeenv}/gtfs_files")
    gtfs_fullpaths = [f"{scenario_dir}/routing/{routeenv}/gtfs_files/{filename}" for filename in gtfs_files]

    # load r5 params
    with open(f"{scenario_dir}/routing/{routeenv}/r5_params.json") as f:
        r5_params = json.load(f)

    # create network
    print("creating network")
    try:
        network = r5py.TransportNetwork(
            osm_pbf=f"{scenario_dir}/routing/{routeenv}/osm_file.pbf",
            gtfs=gtfs_fullpaths,
        )
    except ValueError as e:
        print(f"Error loading network for {routeenv}: {e}")
        print("TRY EMPTYING YOUR CACHE /home/user/.cache/r5py/*")
        raise e

    # route on network to create travel time matrix
    print("calculating travel time matrix")
    ttm = r5py.TravelTimeMatrix(network,
                                rep_points,
                                rep_points,
                                transport_modes=[mode],
                                departure=departure_time,
                                **r5_params
                                )
    ttm_df = pd.DataFrame(ttm)  # shouldn't be necessary, but I was getting a weird r5py error

    if ttm_df.isnull().any().any():
        print(f"Warning calculating network for {routeenv}: travel time matrix contains empty values")

    # Ensure ids are strings for consistent indexing/column matching
    ttm_df["from_id"] = ttm_df["from_id"].astype(str)
    ttm_df["to_id"] = ttm_df["to_id"].astype(str)

    # Pivot to wide format: rows=from_id, columns=to_id
    ttm_df_wide = ttm_df.pivot(index="from_id", columns="to_id", values="travel_time")

    if np.isnan(ttm_df_wide.iloc[1,1]):
        print(f"Error calculating network for {routeenv}: all values are nan")
        print("TRY EMPTYING YOUR CACHE /home/user/.cache/r5py/*")
        raise ValueError

    os.makedirs(f"{scenario_dir}/routing/{routeenv}/raw_ttms/", exist_ok=True)
    ttm_df_wide.to_file(f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv")

    return ttm_df_wide


def route_for_all_envs(scenario_dir: str,
                       geoms_with_dests, #maybe this shouldn't require dests? Only used for communication visualization
                       user_classes: pd.DataFrame,
                       departure_time = DEPARTURE_TIME,
                       visualize_all = True,
                       ):
    """Route for all environments and modes.
    
    Args:
        scenario_dir: Directory for scenario
        geoms_with_dests: GeoDataFrame with destinations (any CRS, will be converted to WGS84)
        user_classes: DataFrame of user class definitions
        departure_time: Datetime for routing
        visualize_all: Whether to create visualizations
    """
    # Ensure WGS84 for r5py
    if geoms_with_dests.crs != "EPSG:4326":
        print(f"Converting geoms_with_dests from {geoms_with_dests.crs} to EPSG:4326")
        geoms_with_dests = geoms_with_dests.to_crs("EPSG:4326")
    
    # Create representative points in WGS84
    rep_points = gpd.GeoDataFrame(crs="EPSG:4326", geometry=geoms_with_dests.representative_point())
    rep_points['id'] = geoms_with_dests['geom_id'].astype(str)
    
    rep_points.to_file(f"{scenario_dir}/routing/DEBUG: rep_points.geojson", driver="GeoJSON")
    user_classes.fillna("", inplace=True)
    for mode in MODES:
        print(f"identifying routeenvs for {mode}")
        route_envs_for_mode = set()
        route_envs_for_mode.update(user_classes[f'routeenv_{mode}'].unique())
        route_envs_for_mode.discard("")
        print(f"routeenvs for {mode}: {route_envs_for_mode}")
        for routeenv in route_envs_for_mode:
            user_class_selection = user_classes[user_classes[f'routeenv_{mode}'] == routeenv]
            raw_ttm = route_environment(routeenv, mode, scenario_dir, rep_points, departure_time)
            post_process_matrices(scenario_dir, raw_ttm, mode, user_class_selection, geoms_with_dests) #this also saves the files
            if visualize_all:
                geoms_with_dests.index = geoms_with_dests['geom_id'].values
                communication.visualize_access_to_zone(scenario_dir,
                                                       geoms_with_dests,
                                                       f"routing/{routeenv}/route_{mode}_raw_traveltime_viz.html",
                                                       raw_ttm,
                                                       target_zone_id=None,
                                                       )

###
# helper functions for impedance calculations
###

def parking_cost(ttm, geoms_with_dests):
    tcm = ttm.copy()
    tcm.loc[:,:] = 0
    # PROVIDENCE/RI SPECIFIC
    # Making some assumptions based on Parkopedia
    mid_price_dests = geoms_with_dests[(geoms_with_dests['lodes_jobs_per_sqkm'] > 7000) &
                                       (geoms_with_dests['lodes_jobs_per_sqkm'] < 12000)
    ]
    mid_price_geomids = mid_price_dests['geom_id'].values
    high_price_dests = geoms_with_dests[(geoms_with_dests['lodes_jobs_per_sqkm'] >= 12000)]
    high_price_geomids = high_price_dests['geom_id'].values

    tcm.loc[:,mid_price_geomids] = 5 #5 dollars to park for the day in downtown Pawtucket, at Brown, and on Broadway
    tcm.loc[:,high_price_geomids] = 15 #10 dollars to park for the day in downtown PVD

    return tcm

def transit_cost(ttm):
    #we assume that transit costs $2.00 per ride
    tcm = ttm.copy()
    tcm.loc[:,:] = 2
    return tcm

def car_operating_cost(ttm):
    #we only count fuel because people don't usually think about maintenance when deciding how to travel
    fuel_cost_per_mile = 0.15 # https://data.bts.gov/stories/s/Transportation-Economic-Trends-Transportation-Spen/bzt6-t8cd/
    est_miles_per_minute = 0.75 # total guess. TODO: during routing, save the distance matrix as well as time
    operating_cost_per_minute = fuel_cost_per_mile * est_miles_per_minute

    tcm = ttm.copy()
    tcm *= operating_cost_per_minute
    return tcm

def estimate_ridehailing_impedances(car_ttm, car_tcm):
    ridehail_ttm = car_ttm + 5 #wait 5 minutes
    ridehail_tcm = car_tcm.copy()
    ridehail_tcm.loc[:,:] = 8 # meter starts at 8???
    ridehail_tcm += ridehail_ttm * 0.5 # (50c per minute, not counting per-mile charges. Again, need to save a distance matrix.

    return ridehail_ttm, ridehail_tcm

def generalized_impedance(ttm, tcm, user_class_row):
    gtm = ttm.copy()

    working_hours_per_year = 2080 #40/wk, 52 wks/yr
    income_per_year = user_class_row['max_income']
    income_per_hour = income_per_year / working_hours_per_year
    dollars_per_hour = income_per_hour / 2
    minutes_per_dollar = 60 / dollars_per_hour

    gtm += tcm * minutes_per_dollar

    return gtm


def traffic_speed_adjustment(scenario_dir, analysis_areas, car_ttm):
    """
    Build an adjusted car travel-time matrix (seconds) using observed traffic samples.

    Parameters
    ----------
    scenario_dir : str
        Path to the scenario directory containing 'input_data/traffic_sample_triptimes.csv'.
    analysis_areas : str
        Path to a GeoPackage (e.g., analysis_areas.gpkg) that contains a 'geom_id' field.
        Centroids are computed from geometry for spatial nearness.
    car_ttm : (str or pandas.DataFrame)
        - Path to a wide CSV (minutes) with a 'from_id' column and destination IDs as columns
          (e.g., your ttm_CAR.csv), or
        - A DataFrame in that same wide format.

    Returns
    -------
    pandas.DataFrame
        Wide matrix like input `car_ttm`.
    """
    # --- Load observed traffic (seconds) ---
    obs_path = f"{scenario_dir}/input_data/traffic_sample_triptimes.csv"
    obs = pd.read_csv(obs_path)[["origin_id", "destination_id", "duration_seconds"]].copy()
    obs["origin_id"] = obs["origin_id"].astype(str)
    obs["destination_id"] = obs["destination_id"].astype(str)

    # --- Load car TTM (minutes) ---
    if isinstance(car_ttm, pd.DataFrame):
        ttm = car_ttm.copy()
    else:
        ttm = pd.read_csv(car_ttm)

    ttm = ttm.copy()
    if not 'from_id' in ttm.columns:
        ttm['from_id'] = ttm.index.astype(str)
    ttm["from_id"] = ttm["from_id"].astype(str)
    dest_cols = [c for c in ttm.columns if c != "from_id"]
    ttm.rename(columns={c: str(c) for c in dest_cols}, inplace=True)
    dest_cols = [c for c in ttm.columns if c != "from_id"]

    # --- Centroids for each geom_id (used to find nearest observed O/D) ---
    if isinstance(analysis_areas, (str, os.PathLike)):
        gdf = gpd.read_file(analysis_areas)
    elif isinstance(analysis_areas, gpd.GeoDataFrame):
        gdf = analysis_areas
    else:
        raise TypeError("analysis_areas must be a path or a GeoDataFrame")

    if "geom_id" not in gdf.columns:
        raise KeyError("analysis_areas must contain a 'geom_id' column")

    # Ensure IDs are strings and compute centroids
    gdf = gdf[["geom_id", gdf.geometry.name]].copy()
    gdf["geom_id"] = gdf["geom_id"].astype(str)

    # Use centroids; if polygons are very irregular, consider representative_point()
    centroids = gdf.geometry.centroid

    coords = dict(zip(
        gdf["geom_id"],
        zip(centroids.x.to_numpy(), centroids.y.to_numpy())
    ))

    # --- Precompute observed scaling factors (observed / estimated) ---
    # Quick access to each origin's row of the TTM
    origin_rows = {row["from_id"]: row for _, row in ttm.iterrows()}

    obs_pairs = []
    obs_factors = []
    obs_points = []

    for _, r in obs.iterrows():
        o, d = r["origin_id"], r["destination_id"]
        if o in origin_rows and d in dest_cols and o in coords and d in coords:
            est_min = origin_rows[o][d]
            if pd.isna(est_min):
                continue
            est_sec = float(est_min) * 60.0
            if est_sec <= 0:
                # can't form ratio if estimated is 0 or negative
                continue

            obs_sec = float(r["duration_seconds"])
            factor = obs_sec / est_sec  # observed / estimated
            obs_pairs.append((o, d))
            obs_factors.append(factor)
            ox, oy = coords[o]
            dx, dy = coords[d]
            obs_points.append((ox, oy, dx, dy))

    if not obs_points:
        raise ValueError("No usable observed O/D pairs found that match the car_ttm and analysis_areas IDs.")

    obs_points = np.array(obs_points)     # (N,4)
    obs_factors = np.array(obs_factors)   # (N,)

    # --- Output matrix in seconds (start from estimated * 60) ---
    out = ttm.copy()
    for c in dest_cols:
        out[c] = out[c].astype(float) * 60.0

    # Direct lookup for observed values
    obs_lookup = {
        (str(o), str(d)): float(sec)
        for o, d, sec in obs[["origin_id", "destination_id", "duration_seconds"]].itertuples(index=False, name=None)
    }

    def nearest_factor(o_id, d_id):
        """Return scaling factor from nearest observed O/D in 4D (ox,oy,dx,dy) space."""
        ox, oy = coords[o_id]
        dx, dy = coords[d_id]
        v = np.array([ox, oy, dx, dy])
        d2 = ((obs_points - v) ** 2).sum(axis=1)
        return float(obs_factors[int(d2.argmin())])

    # Fill each cell: observed if available; else scaled by nearest observed factor
    for i, row in tqdm(out.iterrows(), total=len(out), desc="interpolating traffic speeds"):
        o_id = row["from_id"]
        for d_id in dest_cols:
            key = (o_id, d_id)
            if key in obs_lookup:
                out.at[i, d_id] = obs_lookup[key]  # already seconds
            else:
                est_sec = float(out.at[i, d_id])
                if not np.isnan(est_sec) and o_id in coords and d_id in coords:
                    out.at[i, d_id] = est_sec * nearest_factor(o_id, d_id)
                # else: leave as is (NaN or missing coords)


    #clean up, convert to minutes
    out.drop("from_id", axis=1, inplace=True)
    out = out / 60.0

    return out


###
# mode-specific post-processing functions
###

def post_process_WALK(scenario_dir, ttm, user_class_selection, geoms_with_dests):
    print("post-processing walk")
    # we don't yet do anything except save the ttm and empty tcm to each userclass folder
    mode = "WALK"
    tcm = ttm.copy()
    tcm.loc[:, :] = 0
    for userclass in user_class_selection.index:
        os.makedirs(f"{scenario_dir}/impedances/{userclass}/", exist_ok=True)
        ttm.to_csv(f"{scenario_dir}/impedances/{userclass}/ttm_{mode}.csv", )
        tcm.to_csv(f"{scenario_dir}/impedances/{userclass}/tcm_{mode}.csv")
        gtm = generalized_impedance(ttm, tcm, user_class_selection.loc[userclass])
        gtm.to_csv(f"{scenario_dir}/impedances/{userclass}/gtm_{mode}.csv")

def post_process_BICYCLE(scenario_dir, ttm, user_class_selection, geoms_with_dests):
    print("post-processing bicycle")
    # we don't yet do anything except save the ttm and empty tcm to each userclass folder
    mode = "BICYCLE"
    tcm = ttm.copy()
    tcm.loc[:, :] = 0
    for userclass in user_class_selection.index:
        if str(user_class_selection.loc[userclass, "max_bicycle"]) != "0":
            os.makedirs(f"{scenario_dir}/impedances/{userclass}/", exist_ok=True)
            ttm.to_csv(f"{scenario_dir}/impedances/{userclass}/ttm_{mode}.csv", )
            tcm.to_csv(f"{scenario_dir}/impedances/{userclass}/tcm_{mode}.csv")
            gtm = generalized_impedance(ttm, tcm, user_class_selection.loc[userclass])
            gtm.to_csv(f"{scenario_dir}/impedances/{userclass}/gtm_{mode}.csv")


def post_process_TRANSIT(scenario_dir, ttm, user_class_selection, geoms_with_dests):
    print("post-processing transit")
    mode = "TRANSIT"
    tcm = ttm.copy()
    tcm.loc[:, :] = 0

    tcm += transit_cost(ttm)
    for userclass in user_class_selection.index:
        os.makedirs(f"{scenario_dir}/impedances/{userclass}/", exist_ok=True)
        ttm.to_csv(f"{scenario_dir}/impedances/{userclass}/ttm_{mode}.csv", )
        tcm.to_csv(f"{scenario_dir}/impedances/{userclass}/tcm_{mode}.csv")
        gtm = generalized_impedance(ttm, tcm, user_class_selection.loc[userclass])
        gtm.to_csv(f"{scenario_dir}/impedances/{userclass}/gtm_{mode}.csv")

def post_process_CAR(scenario_dir, ttm, user_class_selection, geoms_with_dests):
    print("post-processing car")
    mode = "CAR"
    tcm = ttm.copy()
    tcm.loc[:, :] = 0

    #adjust car travel times for traffic
    if os.path.exists(f"{scenario_dir}/input_data/traffic_sample_triptimes.csv"):
        ttm = traffic_speed_adjustment(scenario_dir, geoms_with_dests, ttm)

    tcm += car_operating_cost(ttm)
    tcm += parking_cost(ttm, geoms_with_dests)

    # calculate university costs and times for ridehail
    ridehail_ttm, ridehail_tcm = estimate_ridehailing_impedances(ttm, tcm)

    #userclass-specific modifications and saving
    for userclass in tqdm(list(user_class_selection.index)):
        os.makedirs(f"{scenario_dir}/impedances/{userclass}/", exist_ok=True)
        car_available = user_class_selection.loc[userclass, "car_owner"] == "car"
        if car_available:
            ttm.to_csv(f"{scenario_dir}/impedances/{userclass}/ttm_{mode}.csv", )
            tcm.to_csv(f"{scenario_dir}/impedances/{userclass}/tcm_{mode}.csv")
            gtm = generalized_impedance(ttm, tcm, user_class_selection.loc[userclass])
            gtm.to_csv(f"{scenario_dir}/impedances/{userclass}/gtm_{mode}.csv")
        # ridehail is available even to non-car-owners
        ridehail_ttm.to_csv(f"{scenario_dir}/impedances/{userclass}/ttm_RIDEHAIL.csv")
        ridehail_tcm.to_csv(f"{scenario_dir}/impedances/{userclass}/tcm_RIDEHAIL.csv")
        ridehail_gtm = generalized_impedance(ridehail_ttm, ridehail_tcm, user_class_selection.loc[userclass])
        ridehail_gtm.to_csv(f"{scenario_dir}/impedances/{userclass}/gtm_RIDEHAIL.csv")

POST_PROC_FUNCTIONS = {
    "WALK": post_process_WALK,
    "BICYCLE": post_process_BICYCLE,
    "CAR": post_process_CAR,
    "TRANSIT": post_process_TRANSIT,
}

def post_process_matrices(scenario_dir, ttm, mode, user_class_selection, geoms_with_dests):
    """
    Post-processes the travel time and cost matrices, adds additional non-routed modes, and saves them.
    """
    POST_PROC_FUNCTIONS[mode](scenario_dir, ttm, user_class_selection, geoms_with_dests)

