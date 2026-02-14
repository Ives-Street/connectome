import os
import json
import zipfile
import csv
import shutil
import tempfile

import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import r5py
import datetime
import geopandas as gpd
from tqdm import tqdm
import fiona
from shapely.geometry import LineString

import car_routing
import communication
import logging

from constants import MODES, WORKING_HOURS_PER_YEAR, VOT_DIVISOR

logger = logging.getLogger(__name__)


DEPARTURE_TIME = datetime.datetime(2026, 1, 21, 8, 0, 0)

#old -- today, 9am
# DEPARTURE_TIME = datetime.datetime.now().replace(hour=9, minute=0, second=0) + datetime.timedelta(
#     days=(1 - datetime.datetime.now().weekday() + 7) % 7 + (1 if datetime.datetime.now().weekday() >= 3 else 0))

# ---------------------------------------------------------------------------
# Helper: filter out GTFS feeds that are known to break r5py
# ---------------------------------------------------------------------------

def _filter_invalid_gtfs_for_r5py(gtfs_fullpaths: list[str]) -> list[str]:
    """
    Inspect GTFS zip files and drop ones that are obviously invalid for r5py.

    Currently:
    - If a feed contains calendar_dates.txt but that file has no data rows
      (only a header), it is skipped because r5py raises EmptyTableError
      on such feeds.

    Parameters
    ----------
    gtfs_fullpaths : list of str
        Full paths to GTFS zip files.

    Returns
    -------
    list of str
        Subset of gtfs_fullpaths that appear valid.
    """
    valid = []
    for path in gtfs_fullpaths:
        # Default: keep the file unless we detect a fatal issue
        keep = True

        if not os.path.isfile(path):
            logger.warning("GTFS file does not exist, skipping: %s", path)
            continue

        try:
            with zipfile.ZipFile(path, "r") as zf:
                # If there's no calendar_dates.txt, we don't reject the feed:
                # some GTFS feeds legitimately use calendar.txt only.
                if "calendar_dates.txt" in zf.namelist():
                    with zf.open("calendar_dates.txt") as f:
                        reader = csv.reader(
                            (line.decode("utf-8", errors="ignore") for line in f)
                        )
                        # Count non-empty, non-header rows
                        row_count = 0
                        header = next(reader, None)
                        for row in reader:
                            # Ignore completely empty rows
                            if not any(cell.strip() for cell in row):
                                continue
                            row_count += 1
                            if row_count > 0:
                                break

                    if row_count == 0:
                        logger.warning(
                            "Skipping GTFS feed with empty calendar_dates.txt: %s",
                            path,
                        )
                        keep = False

        except zipfile.BadZipFile:
            logger.warning("Skipping malformed GTFS zip (BadZipFile): %s", path)
            keep = False
        except Exception as e:
            # Don't be overly aggressive: log and keep the feed unless it's clearly bad
            logger.error(
                "Error while inspecting GTFS file '%s': %s. Keeping it for now.",
                path,
                e,
            )

        if keep:
            valid.append(path)

    if len(valid) < len(gtfs_fullpaths):
        logger.info(
            "Filtered GTFS feeds for r5py: %d valid, %d skipped",
            len(valid),
            len(gtfs_fullpaths) - len(valid),
        )

    return valid

def r5py_route_environment(routeenv, mode, scenario_dir, rep_points, departure_time):

    logger.info("Starting r5py route_environment for routeenv='%s', mode='%s'", routeenv, mode)
    logger.debug("rep_points size: %d, departure_time: %s", len(rep_points), departure_time)

    # check if we already have the unprocessed ttm for this routeenv and mode
    raw_ttm_filename = f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv"
    if os.path.exists(raw_ttm_filename):
        logger.info("Found cached raw TTM for routeenv='%s', mode='%s' at '%s'",
                    routeenv, mode, raw_ttm_filename)
        ttm_wide = pd.read_csv(raw_ttm_filename, index_col=0)
        ttm_wide.index = ttm_wide.index.astype(str)
        ttm_wide.columns = ttm_wide.columns.astype(str)
        return ttm_wide

    logger.info("Routing %s for %s", mode, routeenv)
    if os.path.exists(f"{scenario_dir}/routing/{routeenv}/gtfs_files"):
        gtfs_files = os.listdir(f"{scenario_dir}/routing/{routeenv}/gtfs_files")
        gtfs_fullpaths = [f"{scenario_dir}/routing/{routeenv}/gtfs_files/{filename}" for filename in gtfs_files]
        logger.debug("Using %d GTFS files for routeenv='%s'", len(gtfs_fullpaths), routeenv)
    else:
        gtfs_fullpaths = []

    # Filter out GTFS feeds that are known to break r5py (e.g. empty calendar_dates)
    gtfs_fullpaths = _filter_invalid_gtfs_for_r5py(gtfs_fullpaths)
    if not gtfs_fullpaths:
        logger.warning(
            "No valid GTFS feeds remain for routeenv='%s' after filtering. "
            "Transit routing may be disabled for this environment.",
            routeenv,
        )

    # load r5 params
    with open(f"{scenario_dir}/routing/{routeenv}/r5_params.json") as f:
        r5_params = json.load(f)
    logger.debug("Loaded r5 params for routeenv='%s': %s", routeenv, list(r5_params.keys()))

    # create network
    logger.info("Creating r5py.TransportNetwork for routeenv='%s'", routeenv)
    try:
        network = r5py.TransportNetwork(
            osm_pbf=f"{scenario_dir}/routing/{routeenv}/osm_file.pbf",
            gtfs=gtfs_fullpaths,
        )
    except Exception as e:
        # r5py can raise GtfsFileError or ValueError or others; log clearly
        logger.error(
            "Error loading r5py.TransportNetwork for routeenv='%s' with GTFS=%s: %s",
            routeenv,
            gtfs_fullpaths,
            e,
        )
        logger.error(f"If this persists, you may need to inspect or regenerate GTFS feeds for routeenv '{routeenv}'.")
        raise

    # route on network to create travel time matrix
    logger.info("Calculating travel time matrix for routeenv='%s', mode='%s'", routeenv, mode)
    ttm = r5py.TravelTimeMatrix(network,
                                rep_points,
                                rep_points,
                                transport_modes=[mode],
                                departure=departure_time,
                                **r5_params
                                )
    ttm_df = pd.DataFrame(ttm)  # shouldn't be necessary, but I was getting a weird r5py error
    logger.debug("Raw TTM shape for routeenv='%s', mode='%s': %s", routeenv, mode, ttm_df.shape)

    if ttm_df.isnull().any().any():
        logger.warning("Travel time matrix for routeenv='%s', mode='%s' contains NaN values",
                       routeenv, mode)
        logger.warning(f"travel time matrix for routeenv '{routeenv}' contains empty values")

    # Ensure ids are strings for consistent indexing/column matching
    ttm_df["from_id"] = ttm_df["from_id"].astype(str)
    ttm_df["to_id"] = ttm_df["to_id"].astype(str)

    # Pivot to wide format: rows=from_id, columns=to_id
    ttm_df_wide = ttm_df.pivot(index="from_id", columns="to_id", values="travel_time")
    logger.debug("Pivoted wide TTM shape for routeenv='%s', mode='%s': %s",
                 routeenv, mode, ttm_df_wide.shape)

    if np.isnan(ttm_df_wide.iloc[1,1]):
        logger.error("All values are NaN in TTM for routeenv='%s', mode='%s'", routeenv, mode)
        logger.error(f"All values are NaN in TTM for routeenv '{routeenv}'. TRY EMPTYING YOUR CACHE ~/.cache/r5py/*")
        raise ValueError

    os.makedirs(f"{scenario_dir}/routing/{routeenv}/raw_ttms/", exist_ok=True)
    ttm_df_wide.to_csv(f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv")
    logger.info("Saved raw TTM for routeenv='%s', mode='%s' to '%s'",
                routeenv, mode, raw_ttm_filename)

    return ttm_df_wide


def route_for_all_envs(scenario_dir: str,
                       geoms_with_dests, #maybe this shouldn't require dests? Only used for communication visualization
                       user_classes: pd.DataFrame,
                       departure_time = DEPARTURE_TIME,
                       visualize_all = True,
                       track_volumes = None,
                       ):
    """Route for all environments and modes.

    Args:
        scenario_dir: Directory for scenario
        geoms_with_dests: GeoDataFrame with destinations (any CRS, will be converted to WGS84)
        user_classes: DataFrame of user class definitions
        departure_time: Datetime for routing
        visualize_all: Whether to create visualizations
    """
    logger.info("Starting route_for_all_envs in scenario_dir='%s'", scenario_dir)
    logger.debug("geoms_with_dests count: %d, user_classes count: %d",
                 len(geoms_with_dests), len(user_classes))

    # Ensure WGS84 for r5py
    if geoms_with_dests.crs != "EPSG:4326":
        logger.info("Converting geoms_with_dests from %s to EPSG:4326", geoms_with_dests.crs)
        logger.info(f"Converting geoms_with_dests from {geoms_with_dests.crs} to EPSG:4326")
        geoms_with_dests = geoms_with_dests.to_crs("EPSG:4326")

    # Create representative points in WGS84
    logger.info("Creating representative points from geoms_with_dests")
    rep_points = gpd.GeoDataFrame(crs="EPSG:4326", geometry=geoms_with_dests.representative_point())
    rep_points['id'] = geoms_with_dests['geom_id'].astype(str)
    logger.debug("rep_points created with %d points", len(rep_points))

    rep_points.to_file(f"{scenario_dir}/routing/DEBUG: rep_points.geojson", driver="GeoJSON")
    logger.info("Saved DEBUG rep_points.geojson for scenario_dir='%s'", scenario_dir)

    user_classes.fillna("", inplace=True)
    for mode in MODES:
        logger.info("Identifying routing environments for mode='%s'", mode)
        logger.info(f"identifying routeenvs for {mode}")
        route_envs_for_mode = set()
        route_envs_for_mode.update(user_classes[f'routeenv_{mode}'].unique())
        route_envs_for_mode.discard("")
        logger.debug("Route environments for mode='%s': %s", mode, route_envs_for_mode)
        logger.debug(f"routeenvs for {mode}: {route_envs_for_mode}")
        for routeenv in route_envs_for_mode:
            user_class_selection = user_classes[user_classes[f'routeenv_{mode}'] == routeenv]
            logger.debug("Selected %d user classes for mode='%s', routeenv='%s'",
                         len(user_class_selection), mode, routeenv)
            if not os.path.exists(f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv"):
                logger.info("Routing mode='%s' in routeenv='%s'", mode, routeenv)
                if mode == "CAR":
                    # TODO add some logic here about when to do ordinary OD TTM, checkpoint TTM, or capture paths
                    if bool(track_volumes):
                        raw_ttm, length_df, checkpoint_bool_df = car_routing.od_matrix_times_with_checkpoints(
                            scenario_dir,
                            routeenv,
                            rep_points,
                            mode,
                            departure_time,
                            checkpoint_node_ids=track_volumes['checkpoint_node_ids'],
                            checkpoint_edge_attr=track_volumes['checkpoint_edge_attr'],
                            checkpoint_edge_values=track_volumes['checkpoint_edge_values'],
                            )
                    else:
                        raw_ttm = car_routing.od_matrix_times(
                            scenario_dir,
                            routeenv,
                            rep_points,
                            mode,
                            departure_time,
                        )
                else:
                    raw_ttm = r5py_route_environment(routeenv, mode, scenario_dir, rep_points, departure_time)
            else:
                logger.debug("Found existing raw_ttm for mode='%s', routeenv='%s'", mode, routeenv)
                raw_ttm = pd.read_csv(f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv", index_col=0)
            user_class_example = user_class_selection.user_class_id.iloc[0]
            if not os.path.exists(f"{scenario_dir}/impedances/{user_class_example}/gtm_{mode}.csv"):
                post_process_matrices(scenario_dir, raw_ttm, mode, user_class_selection, geoms_with_dests) #this also saves the files
                logger.info("Finished post-processing for mode='%s', routeenv='%s'", mode, routeenv)

            if visualize_all:
                geoms_with_dests.index = geoms_with_dests['geom_id'].values
                logger.info("Creating visualization for mode='%s', routeenv='%s'", mode, routeenv)
                #TODO fix this communication call
                # communication.visualize_access_to_zone(scenario_dir,
                #                                        geoms_with_dests,
                #                                        f"{scenario_dir}/routing/{routeenv}/route_{mode}_raw_traveltime_viz.html",
                #                                        raw_ttm,
                #                                        target_zone_id=None,
                #                                        )

###
# routing support utils
###

def route_to_gdf_osmnx2(G, route, attrs=None):
    """
    Build a GeoDataFrame representing the edges in a node-based route.
    Compatible with osmnx >= 2.0 (no get_route_edge_attributes).
    """
    records = []
    for u, v in zip(route[:-1], route[1:]):
        # MultiDiGraph: could have multiple parallel edges; choose the first
        for key, data in G[u][v].items():
            geom = data.get("geometry")
            if geom is None:
                # fallback: build straight-line geometry
                xy1 = (G.nodes[u]["x"], G.nodes[u]["y"])
                xy2 = (G.nodes[v]["x"], G.nodes[v]["y"])
                geom = LineString([xy1, xy2])
            rec = {"u": u, "v": v, "key": key, "geometry": geom}
            if attrs:
                for a in attrs:
                    rec[a] = data.get(a)
            records.append(rec)
            break  # if multiple edges, just take first

    return gpd.GeoDataFrame(records, geometry="geometry")

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
    tcm.loc[:,high_price_geomids] = 15 #15 dollars to park for the day in downtown PVD

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
    """Convert a travel-time matrix and travel-cost matrix into a generalized
    impedance matrix (minutes-equivalent) using the user class's income.

    The conversion works by expressing dollar costs as equivalent minutes of
    travel time via a value-of-time (VOT) derived from income:

        hourly_wage     = max_income / WORKING_HOURS_PER_YEAR
        dollars_per_hr  = hourly_wage / VOT_DIVISOR
        minutes_per_$   = 60 / dollars_per_hr
        gtm             = ttm + tcm * minutes_per_$

    Parameters
    ----------
    ttm : DataFrame
        Origin-destination travel-time matrix (minutes).
    tcm : DataFrame
        Origin-destination travel-cost matrix (dollars).
    user_class_row : Series or dict
        Must contain 'max_income' (annual dollars). Must be positive.

    Returns
    -------
    DataFrame
        Generalized impedance matrix (minutes-equivalent).
    """
    income_per_year = user_class_row['max_income']
    if not (income_per_year > 0):
        raise ValueError(
            f"max_income must be positive, got {income_per_year!r} "
            f"for user class {user_class_row.get('user_class_id', '?')}"
        )

    income_per_hour = income_per_year / WORKING_HOURS_PER_YEAR
    dollars_per_hour = income_per_hour / VOT_DIVISOR
    minutes_per_dollar = 60 / dollars_per_hour

    gtm = ttm + tcm * minutes_per_dollar

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
    logger.info("Starting traffic_speed_adjustment for scenario_dir='%s'", scenario_dir)

    # --- Load observed traffic (seconds) ---
    obs_path = f"{scenario_dir}/input_data/traffic_sample_triptimes.csv"
    logger.debug("Loading observed traffic from '%s'", obs_path)
    obs = pd.read_csv(obs_path)[["origin_id", "destination_id", "duration_seconds"]].copy()
    obs["origin_id"] = obs["origin_id"].astype(str)
    obs["destination_id"] = obs["destination_id"].astype(str)
    logger.info("Loaded %d observed traffic samples", len(obs))

    # --- Load car TTM (minutes) ---
    if isinstance(car_ttm, pd.DataFrame):
        logger.debug("car_ttm provided as DataFrame with shape %s", car_ttm.shape)
        ttm = car_ttm.copy()
    else:
        logger.debug("Loading car_ttm from CSV '%s'", car_ttm)
        ttm = pd.read_csv(car_ttm)

    ttm = ttm.copy()
    if not 'from_id' in ttm.columns:
        logger.debug("'from_id' column missing in car_ttm; creating from index")
        ttm['from_id'] = ttm.index.astype(str)
    ttm["from_id"] = ttm["from_id"].astype(str)
    dest_cols = [c for c in ttm.columns if c != "from_id"]
    ttm.rename(columns={c: str(c) for c in dest_cols}, inplace=True)
    dest_cols = [c for c in ttm.columns if c != "from_id"]
    logger.info("Car TTM has %d origins and %d destinations", len(ttm), len(dest_cols))

    # --- Centroids for each geom_id (used to find nearest observed O/D) ---
    if isinstance(analysis_areas, (str, os.PathLike)):
        logger.debug("Loading analysis_areas from file '%s'", analysis_areas)
        gdf = gpd.read_file(analysis_areas)
    elif isinstance(analysis_areas, gpd.GeoDataFrame):
        logger.debug("Using analysis_areas GeoDataFrame with %d rows", len(analysis_areas))
        gdf = analysis_areas
    else:
        raise TypeError("analysis_areas must be a path or a GeoDataFrame")

    if "geom_id" not in gdf.columns:
        logger.error("analysis_areas is missing 'geom_id' column")
        raise KeyError("analysis_areas must contain a 'geom_id' column")

    # Ensure IDs are strings and compute centroids
    gdf = gdf[["geom_id", gdf.geometry.name]].copy()
    gdf["geom_id"] = gdf["geom_id"].astype(str)

    # Use centroids; if polygons are very irregular, consider representative_point()
    centroids = gdf.geometry.centroid
    logger.info("Computed centroids for %d analysis areas", len(gdf))

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
        logger.error("No usable observed O/D pairs found matching car_ttm and analysis_areas")
        raise ValueError("No usable observed O/D pairs found that match the car_ttm and analysis_areas IDs.")

    obs_points = np.array(obs_points)     # (N,4)
    obs_factors = np.array(obs_factors)   # (N,)
    logger.info("Prepared %d observed OD points for interpolation", len(obs_points))

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
    logger.info("Interpolating traffic speeds onto full car TTM")
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
    logger.info("Completed traffic_speed_adjustment; returning adjusted TTM with shape %s", out.shape)

    return out


###
# mode-specific post-processing functions
###

def post_process_WALK(scenario_dir, ttm, user_class_selection, geoms_with_dests):
    logger.info("Post-processing WALK impedances for %d userclasses", len(user_class_selection))
    logger.info("post-processing WALK")
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
    logger.info("Post-processing BICYCLE impedances for %d userclasses", len(user_class_selection))
    logger.info("post-processing BICYCLE")
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
    logger.info("Post-processing TRANSIT impedances for %d userclasses", len(user_class_selection))
    logger.info("post-processing TRANSIT")
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
    logger.info("Post-processing CAR impedances for %d userclasses", len(user_class_selection))
    logger.info("post-processing CAR")
    mode = "CAR"
    tcm = ttm.copy()
    tcm.loc[:, :] = 0

    #adjust car travel times for traffic
    if os.path.exists(f"{scenario_dir}/input_data/traffic_sample_triptimes.csv"):
        logger.info("Found traffic_sample_triptimes.csv; applying traffic_speed_adjustment")
        ttm = traffic_speed_adjustment(scenario_dir, geoms_with_dests, ttm)
    else:
        logger.info("No traffic_sample_triptimes.csv found; using unadjusted car TTM")

    tcm += car_operating_cost(ttm)
    tcm += parking_cost(ttm, geoms_with_dests)

    # calculate university costs and times for ridehail
    ridehail_ttm, ridehail_tcm = estimate_ridehailing_impedances(ttm, tcm)

    #userclass-specific modifications and saving
    for userclass in tqdm(list(user_class_selection.index)):
        os.makedirs(f"{scenario_dir}/impedances/{userclass}/", exist_ok=True)
        car_available = user_class_selection.loc[userclass, "car_owner"] == "car"
        if car_available:
            logger.debug("Saving CAR impedances for userclass='%s'", userclass)
            ttm.to_csv(f"{scenario_dir}/impedances/{userclass}/ttm_{mode}.csv", )
            tcm.to_csv(f"{scenario_dir}/impedances/{userclass}/tcm_{mode}.csv")
            gtm = generalized_impedance(ttm, tcm, user_class_selection.loc[userclass])
            gtm.to_csv(f"{scenario_dir}/impedances/{userclass}/gtm_{mode}.csv")
        # ridehail is available even to non-car-owners
        ridehail_ttm.to_csv(f"{scenario_dir}/impedances/{userclass}/ttm_RIDEHAIL.csv")
        ridehail_tcm.to_csv(f"{scenario_dir}/impedances/{userclass}/tcm_RIDEHAIL.csv")
        ridehail_gtm = generalized_impedance(ridehail_ttm, ridehail_tcm, user_class_selection.loc[userclass])
        ridehail_gtm.to_csv(f"{scenario_dir}/impedances/{userclass}/gtm_RIDEHAIL.csv")
    logger.info("Finished CAR and RIDEHAIL post-processing for all userclasses")

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
