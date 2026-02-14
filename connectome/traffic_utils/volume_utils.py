from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime

from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, Sequence, Union

DateLike = Union[str, date, datetime]
EdgeKey = Tuple[int, int, int]

import statsmodels.api as sm
import numpy as np
import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString

try:
    import osmnx as ox
except ImportError:  # pragma: no cover
    ox = None

logger = logging.getLogger(__name__)

# ==============================================================================
# Traffic analysis parameter loading (JSON)
# ==============================================================================

_TRAFFIC_PARAMS_CACHE: Optional[Dict[str, Any]] = None


def _load_traffic_params(
    json_path: str = "traffic_analysis_parameters.json",
) -> Dict[str, Any]:
    """
    Load and cache traffic analysis parameters from JSON.

    Expects at least:
    {
      "osm_highway_to_tmas_fsystem_rank": {
        "motorway": [1,2,3],
        "primary":  [3,2,4,5],
        ...
      }
    }
    """
    global _TRAFFIC_PARAMS_CACHE
    if _TRAFFIC_PARAMS_CACHE is not None:
        return _TRAFFIC_PARAMS_CACHE

    if not os.path.exists(json_path):
        logger.warning(
            "traffic analysis parameters JSON not found at %s; "
            "proceeding without functional-class rank mapping.",
            json_path,
        )
        _TRAFFIC_PARAMS_CACHE = {}
        return _TRAFFIC_PARAMS_CACHE

    with open(json_path, "r") as f:
        params = json.load(f)

    _TRAFFIC_PARAMS_CACHE = params
    return params


# ==============================================================================
# TMAS helpers
# ==============================================================================


def load_tmas_stations(tmas_sta_filepath: str) -> gpd.GeoDataFrame:
    """
    Load TMAS *.STA file and return a GeoDataFrame of station+direction
    records (aggregated across lanes), in WGS84 degrees.

    Notes
    -----
    - TMAS convention: travel_lane == 0 represents a station-level record
      (all lanes). We keep only these and aggregate volumes across lanes
      later using the VOL file.
    - Latitude/longitude are stored as integer microdegrees.
    """
    df = pd.read_csv(tmas_sta_filepath, sep="|")

    # Keep station-level records; we will aggregate volumes by station later.
    if "travel_lane" in df.columns:
        df = df[df["travel_lane"] == 0].copy()

    lat_col = "latitude"
    lon_col = "longitude"
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(
            "Expected 'latitude' and 'longitude' columns (microdegrees) "
            "in TMAS STA file."
        )

    df["latitude_deg"] = df[lat_col] / 1_000_000.0
    df["longitude_deg"] = -1 * (df[lon_col] / 1_000_000.0)

    geometry = gpd.points_from_xy(df["longitude_deg"], df["latitude_deg"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4326, inplace=True)

    return gdf


def load_tmas_volumes(vol_path: str) -> pd.DataFrame:
    """
    Lightweight helper to load TMAS Volume (*.VOL) file.
    """
    df = pd.read_csv(vol_path, sep="|")
    return df

def parse_date(d: DateLike) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        # Accept "YYYY-MM-DD"
        return datetime.strptime(d.strip(), "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


# ==============================================================================
# Graph helpers
# ==============================================================================


def _ensure_graph(G: nx.MultiDiGraph | str) -> nx.MultiDiGraph:
    """
    Ensure we have a MultiDiGraph. If a string path is provided, load GraphML.
    """
    if isinstance(G, nx.MultiDiGraph):
        return G

    if isinstance(G, str):
        if ox is None:
            raise ImportError(
                "osmnx is required to load a graph from a file path, "
                "but it is not installed."
            )
        logger.info("Loading graph from %s via osmnx.load_graphml", G)
        return ox.load_graphml(G)

    raise TypeError(
        "G must be a networkx.MultiDiGraph or a string path to a GraphML file."
    )


def graph_edges_to_gdf(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Convert an OSMnx-style MultiDiGraph to an edges GeoDataFrame in EPSG:4326.
    """
    if ox is not None:
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        return edges.reset_index()

    records: List[Dict[str, Any]] = []
    for u, v, key, data in G.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            geom = LineString([(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])])

        rec = {"u": u, "v": v, "key": key, "geometry": geom}
        rec.update({k: v for k, v in data.items() if k != "geometry"})
        records.append(rec)

    edges_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    edges_gdf = edges_gdf.reset_index(drop=True)
    return edges_gdf


# ==============================================================================
# Direction + functional-class helpers
# ==============================================================================


def _tmas_travel_dir_bearing(travel_dir: Any) -> Optional[float]:
    """
    Map TMAS travel_dir code to an approximate compass bearing in degrees.
    """
    try:
        d = int(travel_dir)
    except (TypeError, ValueError):
        return None

    mapping = {
        1: 0.0,    # NB
        3: 90.0,   # EB
        5: 180.0,  # SB
        7: 270.0,  # WB
    }
    return mapping.get(d)


def _edge_bearing_from_geometry(geom: LineString) -> Optional[float]:
    """
    Compute approximate bearing (deg from North, clockwise) from a LineString.
    """
    if geom is None:
        return None

    try:
        x1, y1 = geom.coords[0]
        x2, y2 = geom.coords[-1]
    except Exception:  # pragma: no cover - defensive
        return None

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg


def _edge_bearing(edge_row: pd.Series) -> Optional[float]:
    """
    Get or compute an edge bearing in degrees.
    """
    for key in ("bearing", "bearing_mean"):
        if key in edge_row and pd.notna(edge_row[key]):
            try:
                return float(edge_row[key])
            except (TypeError, ValueError):
                pass

    geom = edge_row.get("geometry")
    if isinstance(geom, LineString):
        return _edge_bearing_from_geometry(geom)

    return None


def _bearing_difference_deg(b1: float, b2: float) -> float:
    """
    Smallest absolute difference between two bearings in degrees (0–180).
    """
    diff = abs(b1 - b2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def _normalize_osm_highway_value(highway: Any) -> Optional[str]:
    """
    Normalize OSM 'highway' tag to a single lowercase string for lookup.
    """
    if highway is None or (isinstance(highway, float) and pd.isna(highway)):
        return None

    if isinstance(highway, list):
        hw = str(highway[0]).lower()
    else:
        hw = str(highway).lower()

    hw = hw.split(";")[0].strip()
    return hw or None


def _functional_class_penalty(
    tmas_f_system: Any,
    osm_highway: Any,
    traffic_params: Dict[str, Any],
) -> Tuple[float, Optional[int]]:
    """
    Turn TMAS f_system vs OSM highway mapping into an additive penalty.

    Returns
    -------
    penalty : float
        >= 0, 0 means best possible match.
    rank_index : int or None
        Index of f_system in the rank list (0 = best), or None if not present.
    """
    try:
        f = int(tmas_f_system)
    except (TypeError, ValueError):
        return 1.0, None  # some mild penalty for unknown f_system

    highway_norm = _normalize_osm_highway_value(osm_highway)
    mapping = traffic_params.get("osm_highway_to_tmas_fsystem_rank", {})

    if not highway_norm or highway_norm not in mapping:
        # Missing/nonstandard highway tag: big penalty, unlikely to host TMAS station.
        return 4.0, None

    rank_list = mapping.get(highway_norm, [])
    if not rank_list:
        return 1.0, None

    if f not in rank_list:
        # f_system not in list: fairly large penalty.
        return 3.0, None

    idx = rank_list.index(f)

    # Rank index -> penalty (tunable)
    if idx == 0:
        pen = 0.0
    elif idx == 1:
        pen = 0.5
    elif idx == 2:
        pen = 1.0
    elif idx == 3:
        pen = 2.0
    else:
        pen = 3.0

    return pen, idx


# ==============================================================================
# Route-number matching
# ==============================================================================


def _extract_route_numbers(text: Any) -> List[str]:
    """
    Extract clusters of consecutive digits from a route description string.

    Examples
    --------
    "SR24"              -> ["24"]
    " DE 1D; DE 24"     -> ["1", "24"]
    "I-295 / US 40"     -> ["295", "40"]
    "Route 024 / 5"     -> ["024", "5"]
    """
    if text is None:
        return []

    s = str(text).strip()
    if not s:
        return []

    return re.findall(r"\d+", s)


def _route_number_match(
    tmas_route: Any,
    osm_ref: Any,
) -> Tuple[bool, Set[str], Set[str], Set[str]]:
    """
    Compare TMAS posted_signed_route and OSM ref based on numeric route IDs.
    """
    tmas_nums = set(_extract_route_numbers(tmas_route))
    osm_nums = set(_extract_route_numbers(osm_ref))
    common = tmas_nums & osm_nums
    return bool(common), tmas_nums, osm_nums, common


# ==============================================================================
# Candidate scoring – weighted sum, de-emphasized distance
# ==============================================================================


@dataclass
class TMASCandidateEdge:
    u: Any
    v: Any
    key: Any
    edge_idx: int
    distance_m: float
    score: float
    highway: Optional[str] = None
    ref: Optional[str] = None
    tmas_f_system: Optional[int] = None
    fsystem_rank_index: Optional[int] = None
    edge_bearing_deg: Optional[float] = None
    tmas_bearing_deg: Optional[float] = None
    bearing_diff_deg: Optional[float] = None
    tmas_route_numbers: Optional[List[str]] = None
    osm_route_numbers: Optional[List[str]] = None
    route_numbers_common: Optional[List[str]] = None


def _basic_candidate_score(
    station_row: pd.Series,
    edge_row: pd.Series,
    distance_m: float,
    traffic_params: Dict[str, Any],
    distance_scale_m: float = 200.0,
    distance_cap_m: float = 400.0,
    w_dist: float = 0.2,
    w_func: float = 0.4,
    w_dir: float = 0.3,
    w_route: float = 0.1,
) -> Tuple[
    float,
    Optional[int],
    Optional[float],
    Optional[float],
    Optional[float],
    List[str],
    List[str],
    List[str],
]:
    """
    Compute a weighted-sum score for a candidate edge.

    Final score = w_dist * dist_term + w_func * func_pen + w_dir * dir_pen + w_route * route_pen

    Terms
    -----
    dist_term:
        - distance in meters is first capped at distance_cap_m.
        - then normalized as: 1.0 + (distance_m_capped / distance_scale_m).
          e.g., 50 m → ~1.25, 200 m → 2.0, 400 m → 3.0.
    func_pen:
        - additive penalty from functional-class comparison (0 = perfect).
    dir_pen:
        - based on bearing difference (0–180 degrees):
          diff <=  30 : 0.0
          30–60       : 0.5
          60–120      : 1.5
          >120        : 3.0
    route_pen:
        - 0.0 if there is any overlapping route number.
        - 4.0 if no numeric overlap (or no numbers).
    """
    # Distance term
    d_cap = min(distance_m, distance_cap_m)
    dist_term = 1.0 + d_cap / distance_scale_m

    # Functional-class penalty
    func_pen, rank_idx = _functional_class_penalty(
        tmas_f_system=station_row.get("f_system"),
        osm_highway=edge_row.get("highway"),
        traffic_params=traffic_params,
    )

    # Direction penalty
    tmas_bearing = _tmas_travel_dir_bearing(station_row.get("travel_dir"))
    edge_bearing_val = _edge_bearing(edge_row)
    bearing_diff = None
    if tmas_bearing is not None and edge_bearing_val is not None:
        bearing_diff = _bearing_difference_deg(tmas_bearing, edge_bearing_val)
        if bearing_diff <= 30.0:
            dir_pen = 0.0
        elif bearing_diff <= 60.0:
            dir_pen = 0.5
        elif bearing_diff <= 120.0:
            dir_pen = 1.5
        else:
            dir_pen = 3.0
    else:
        dir_pen = 1.0  # mild penalty if direction cannot be evaluated

    # Route-number penalty
    tmas_route_val = station_row.get("posted_signed_route", "")
    osm_ref_val = edge_row.get("ref", "")
    has_route_match, t_nums, o_nums, common = _route_number_match(
        tmas_route_val, osm_ref_val
    )
    route_pen = 0.0 if has_route_match else 4.0

    # Weighted sum
    score = (
        w_dist * dist_term
        + w_func * func_pen
        + w_dir * dir_pen
        + w_route * route_pen
    )

    return (
        float(score),
        rank_idx,
        tmas_bearing,
        edge_bearing_val,
        bearing_diff,
        sorted(t_nums),
        sorted(o_nums),
        sorted(common),
    )


# ==============================================================================
# Main matching function
# ==============================================================================


def match_tmas_stations_to_graph(
    G: nx.MultiDiGraph | str,
    tmas_sta_filepath: str,
    max_candidates: int = 3,
    candidate_search_k: int = 10,
    save_matches_gpkg_to: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Match TMAS station+direction points to best-match edges in an OSMnx graph.

    Key design notes
    ----------------
    - Matching unit is (station_id, travel_dir), aggregated across lanes.
    - Highways are treated as divided because the graph is directed.
    - Output CRS is EPSG:4326, but distances are computed in meters using a
      local projected CRS (via GeoPandas' estimate_utm_crs).
    - Distance is de-emphasized via:
        * a capped and rescaled distance term, plus
        * a weighted-sum score where distance weight is small.
    - A hard distance cap (currently 400 m) is used to filter candidates.
    """
    G = _ensure_graph(G)
    stations_wgs = load_tmas_stations(tmas_sta_filepath)
    edges_wgs = graph_edges_to_gdf(G)
    traffic_params = _load_traffic_params()

    # Sanity checks
    if stations_wgs.crs is None or stations_wgs.crs.to_epsg() != 4326:
        raise ValueError("stations_gdf must be in EPSG:4326 (WGS84).")
    if edges_wgs.crs is None or edges_wgs.crs.to_epsg() != 4326:
        raise ValueError("edges_gdf must be in EPSG:4326 (WGS84).")

    # Project to local CRS for distance computations.
    proj_crs = stations_wgs.estimate_utm_crs()
    logger.info("Using projected CRS %s for distance calculations.", proj_crs)

    stations = stations_wgs.to_crs(proj_crs)
    edges = edges_wgs.to_crs(proj_crs)

    edges_sindex = edges.sindex

    # Distance search parameters
    buffer_m = 400.0  # search radius in meters for candidate edges
    distance_cap_m = 400.0  # for scoring, must stay in sync with _basic_candidate_score default

    out_records: List[Dict[str, Any]] = []
    unmatched_records: List[Dict[str, Any]] = []

    for idx, s_row_wgs in stations_wgs.iterrows():
        pt_proj: Point = stations.loc[idx].geometry

        # Candidate search in projected CRS (meters)
        bbox = pt_proj.buffer(buffer_m).bounds
        candidate_idx = list(edges_sindex.intersection(bbox))

        if not candidate_idx:
            logger.warning(
                "No edge bbox candidates for TMAS station_id=%s travel_dir=%s",
                s_row_wgs.get("station_id"),
                s_row_wgs.get("travel_dir"),
            )
            rec = s_row_wgs.to_dict()
            rec.update(
                {
                    "match_u": None,
                    "match_v": None,
                    "match_key": None,
                    "match_edge_idx": None,
                    "match_distance_m": None,
                    "match_score": None,
                    "candidate_edges_json": json.dumps([]),
                }
            )
            unmatched_records.append(rec)
            out_records.append(rec)
            continue

        # Compute distances (meters) and keep nearest K,
        # but also enforce hard distance cap.
        candidates_dist: List[Tuple[int, float]] = []
        for e_idx in candidate_idx:
            e_geom = edges.loc[e_idx, "geometry"]
            d_m = float(pt_proj.distance(e_geom))
            if d_m <= distance_cap_m:
                candidates_dist.append((e_idx, d_m))

        if not candidates_dist:
            logger.warning(
                "All candidate edges beyond distance cap for station_id=%s travel_dir=%s",
                s_row_wgs.get("station_id"),
                s_row_wgs.get("travel_dir"),
            )
            rec = s_row_wgs.to_dict()
            rec.update(
                {
                    "match_u": None,
                    "match_v": None,
                    "match_key": None,
                    "match_edge_idx": None,
                    "match_distance_m": None,
                    "match_score": None,
                    "candidate_edges_json": json.dumps([]),
                }
            )
            unmatched_records.append(rec)
            out_records.append(rec)
            continue

        candidates_dist.sort(key=lambda x: x[1])
        candidates_dist = candidates_dist[:candidate_search_k]

        # Score candidates using weighted-sum logic.
        candidates_scored: List[TMASCandidateEdge] = []
        s_row_proj = stations.loc[idx]
        for e_idx, d_m in candidates_dist:
            e_row_proj = edges.loc[e_idx]
            e_row_wgs = edges_wgs.loc[e_idx]

            (
                score,
                rank_idx,
                tmas_bearing,
                edge_bearing_val,
                bearing_diff,
                t_nums,
                o_nums,
                common_nums,
            ) = _basic_candidate_score(
                station_row=s_row_wgs,  # attributes; geometry not needed here
                edge_row=e_row_wgs,
                distance_m=d_m,
                traffic_params=traffic_params,
            )

            score_threshold = 4.4 # if the route ID doesn't match, it has to be extremely close on both direction and distance
            if score > score_threshold:
                continue

            try:
                tmas_f = int(s_row_wgs.get("f_system"))
            except (TypeError, ValueError):
                tmas_f = None

            cand = TMASCandidateEdge(
                u=e_row_wgs.get("u"),
                v=e_row_wgs.get("v"),
                key=e_row_wgs.get("key"),
                edge_idx=int(e_idx),
                distance_m=d_m,
                score=score,
                highway=_normalize_osm_highway_value(e_row_wgs.get("highway")),
                ref=e_row_wgs.get("ref"),
                tmas_f_system=tmas_f,
                fsystem_rank_index=rank_idx,
                edge_bearing_deg=edge_bearing_val,
                tmas_bearing_deg=tmas_bearing,
                bearing_diff_deg=bearing_diff,
                tmas_route_numbers=t_nums,
                osm_route_numbers=o_nums,
                route_numbers_common=common_nums,
            )
            candidates_scored.append(cand)

        candidates_scored.sort(key=lambda c: c.score)
        top_candidates = candidates_scored[:max_candidates]

        if not top_candidates:
            logger.warning(
                "No scored candidates for TMAS station_id=%s travel_dir=%s",
                s_row_wgs.get("station_id"),
                s_row_wgs.get("travel_dir"),
            )
            rec = s_row_wgs.to_dict()
            rec.update(
                {
                    "match_u": None,
                    "match_v": None,
                    "match_key": None,
                    "match_edge_idx": None,
                    "match_distance_m": None,
                    "match_score": None,
                    "candidate_edges_json": json.dumps([]),
                }
            )
            unmatched_records.append(rec)
            out_records.append(rec)
            continue

        best = top_candidates[0]
        candidates_json = json.dumps([asdict(c) for c in top_candidates], default=str)

        rec = s_row_wgs.to_dict()
        rec.update(
            {
                "match_u": best.u,
                "match_v": best.v,
                "match_key": best.key,
                "match_edge_idx": best.edge_idx,
                "match_distance_m": best.distance_m,
                "match_score": best.score,
                "candidate_edges_json": candidates_json,
            }
        )
        out_records.append(rec)

    matches_gdf = gpd.GeoDataFrame(out_records, geometry="geometry", crs="EPSG:4326")

    if save_matches_gpkg_to is not None:
        logger.info("Writing TMAS match debug GeoPackage to %s", save_matches_gpkg_to)
        matches_gdf.to_file(
            save_matches_gpkg_to,
            layer="tmas_station_matches",
            driver="GPKG",
        )

        unmatched_gdf = matches_gdf[matches_gdf["match_u"].isna()].copy()
        if len(unmatched_gdf) > 0:
            unmatched_gdf.to_file(
                save_matches_gpkg_to,
                layer="tmas_unmatched_stations",
                driver="GPKG",
            )

    return G, matches_gdf


# ==============================================================================
# Assign volumes to graph
# ==============================================================================

def apply_tmas_obs_volumes_to_graph(
    scenario_dir,
    *,
    date_start: DateLike,
    date_end: DateLike,
    hours: Sequence[int],
    G: Optional[str | nx.MultiDiGraph] = None, #graph, file location, or None. If None, reads from default path
    attr_name: str = "obs_vol_vph",
    source_attr_name: str = "obs_vol_vph_source",
    matches_layer: Optional[str] = "tmas_station_matches",
    # Columns in the station-matches file identifying the matched edge
    match_u_col: str = "match_u",
    match_v_col: str = "match_v",
    match_key_col: str = "match_key",
    # Columns used to join VOL records to the matches table
    join_cols: Sequence[str] = ("state_code", "station_id", "travel_dir", "travel_lane"),
    # VOL schema columns for date
    vol_year_col: str = "year_record",
    vol_month_col: str = "month_record",
    vol_day_col: str = "day_record",
) -> Tuple[nx.MultiDiGraph, List[EdgeKey]]:
    """
    Load TMAS station matches + a TMAS .VOL file, filter by date range and hour bins,
    and write obs_vol_vph onto matched edges only.

    Constraints enforced (per your spec):
      - all vehicles (uses VOL hourly totals as-is)
      - directed (applies to the directed edge (u,v,key) only)
      - missing data => raise
      - if attr already exists on any target edge => raise
      - does not touch unmatched edges
      - returns list of (u,v,key) edges that were assigned volumes

    Parameters
    ----------
    hours : sequence of ints
        Hour-of-day bins to include, 0..23. Each maps to column "hour_XX".

    date_start, date_end :
        Inclusive range. (i.e., start <= date <= end)

    Returns
    -------
    (G, matched_edges)
        G is mutated in-place and also returned for convenience.
    """
    d0 = parse_date(date_start)
    d1 = parse_date(date_end)
    if d1 < d0:
        raise ValueError(f"date_end ({d1}) is before date_start ({d0}).")

    if not hours:
        raise ValueError("hours must be a non-empty list of integers 0..23.")
    bad_hours = [h for h in hours if not isinstance(h, int) or h < 0 or h > 23]
    if bad_hours:
        raise ValueError(f"Invalid hour values (must be ints 0..23): {bad_hours}")
    hour_cols = [f"hour_{h:02d}" for h in hours]

    if G is None:
        G_path = f"{scenario_dir}/input_data/traffic/post_benchmark/graph_with_relative_demands.graphml"
    if type(G) == str:
        G = ox.load_graphml(G)
    tmas_station_matches_path = f"{scenario_dir}/input_data/traffic/tmas_station_matches.gpkg"
    tmas_vol_path = f"{scenario_dir}/input_data/tmas/tmas.VOL"

    # --- Load station matches (GPKG) ---
    try:
        if matches_layer is None:
            matches_gdf = gpd.read_file(tmas_station_matches_path)
        else:
            matches_gdf = gpd.read_file(tmas_station_matches_path, layer=matches_layer)
    except Exception as e:
        raise RuntimeError(f"Failed to read station matches from {tmas_station_matches_path}: {e}") from e

    required_matches_cols = set([match_u_col, match_v_col, match_key_col, *join_cols])
    missing = required_matches_cols.difference(matches_gdf.columns)

    if missing:
        raise KeyError(
            f"Station matches file is missing required columns: {sorted(missing)}. "
            f"Found: {list(matches_gdf.columns)}"
        )

    # Keep only rows that actually have a match_u/v/key
    m = matches_gdf.dropna(subset=[match_u_col, match_v_col, match_key_col]).copy()
    if m.empty:
        raise ValueError("No matched stations found (all match_u/v/key are null).")

    # Normalize types for edge keys
    for c in (match_u_col, match_v_col, match_key_col):
        # Some GPKGs store ints as floats; enforce clean int conversion
        if (m[c] % 1 != 0).any():
            raise ValueError(f"Column {c} contains non-integer values; cannot form (u,v,key) tuples reliably.")
        m[c] = m[c].astype(int)

    # Reduce to join+edge columns (dedupe join keys if necessary)
    matches_df = m[list(join_cols) + [match_u_col, match_v_col, match_key_col]].drop_duplicates()

    # --- Load VOL (pipe-delimited) ---
    try:
        vol = pd.read_csv(tmas_vol_path, sep="|", dtype=str, engine="python")
    except Exception as e:
        raise RuntimeError(f"Failed to read VOL from {tmas_vol_path}: {e}") from e

    required_vol_cols = set([*join_cols, vol_year_col, vol_month_col, vol_day_col, *hour_cols])
    missing_vol = required_vol_cols.difference(vol.columns)
    if missing_vol:
        raise KeyError(
            f"VOL file is missing required columns: {sorted(missing_vol)}. "
            f"Found: {list(vol.columns)}"
        )

    # Build a proper date column
    for c in (vol_year_col, vol_month_col, vol_day_col):
        vol[c] = pd.to_numeric(vol[c], errors="raise")

    vol["__date__"] = pd.to_datetime(
        dict(year=vol[vol_year_col], month=vol[vol_month_col], day=vol[vol_day_col]),
        errors="raise",
    ).dt.date

    # Filter by date range (inclusive)
    vol = vol[(vol["__date__"] >= d0) & (vol["__date__"] <= d1)].copy()
    if vol.empty:
        raise ValueError(f"No VOL records in date range {d0} .. {d1}.")

    # Convert hourly columns to numeric
    for c in hour_cols:
        # Treat empty strings as NaN
        vol[c] = pd.to_numeric(vol[c].replace({"": None, " ": None}), errors="coerce")

    # Normalize join-key dtypes BEFORE the merge (safe default: string)
    # 1) Normalize join-key dtypes (string) on both sides
    for c in join_cols:
        matches_df[c] = matches_df[c].astype(str).str.strip()
        vol[c] = vol[c].astype(str).str.strip()

    # 2) TMAS station_id normalization: VOL uses 6-char zero-padded strings
    if "station_id" in join_cols:
        matches_df["station_id"] = matches_df["station_id"].str.lstrip("0").replace("", "0").str.zfill(6)
        vol["station_id"] = vol["station_id"].str.lstrip("0").replace("", "0").str.zfill(6)

    # (Optional but recommended) ensure join key uniqueness in matches_df to avoid smearing
    dup = matches_df.duplicated(subset=list(join_cols), keep=False)
    if dup.any():
        examples = matches_df.loc[dup, list(join_cols) + [match_u_col, match_v_col, match_key_col]].head(10).to_dict(
            "records")
        raise ValueError(
            f"Station matches join keys are not unique for {list(join_cols)}; "
            f"this would duplicate VOL rows across edges. Examples (up to 10): {examples}"
        )

    # Join VOL to matches so we only process matched stations
    joined = vol.merge(matches_df, on=list(join_cols), how="inner")
    if joined.empty:
        raise ValueError(
            "After joining VOL to station matches, no records remained. "
            f"Check that join columns align: {list(join_cols)}"
        )

    # Missing-data policy: raise if ANY selected hour is missing in ANY included record
    # (for matched stations within the selected date range)
    missing_mask = joined[hour_cols].isna().any(axis=1)
    if missing_mask.any():
        examples = joined.loc[missing_mask, list(join_cols) + ["__date__"]].head(10).to_dict("records")
        raise ValueError(
            "Missing hourly VOL data detected for at least one matched station/date "
            f"in the selected hours {sorted(set(hours))}. Examples (up to 10): {examples}"
        )

    # Compute mean across selected hours for each row (station-date record)
    joined["__row_mean__"] = joined[hour_cols].mean(axis=1)

    # Aggregate to edge: mean across all contributing station-date records
    edge_means = (
        joined.groupby([match_u_col, match_v_col, match_key_col], as_index=False)["__row_mean__"]
        .mean()
        .rename(columns={"__row_mean__": attr_name})
    )

    # Apply attribute to graph, enforcing "attribute = raise"
    matched_edges: List[EdgeKey] = []
    for _, r in edge_means.iterrows():
        u, v, k = int(r[match_u_col]), int(r[match_v_col]), int(r[match_key_col])
        val = float(r[attr_name])

        if not G.has_edge(u, v, k):
            raise KeyError(f"Matched edge (u,v,key)=({u},{v},{k}) not present in provided MultiDiGraph.")

        data = G.get_edge_data(u, v, k)
        # networkx returns an attribute dict for that specific key
        if data is None:
            raise KeyError(f"Edge data missing for (u,v,key)=({u},{v},{k}).")

        if attr_name in data:
            logger.warning(
                f"Edge (u,v,key)=({u},{v},{k}) already has attribute '{attr_name}'. "
                "Assuming this means the function has already been run, and skipping..."
            )
            break

        data[attr_name] = val
        matched_edges.append((u, v, k))

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    edges.crs=4326
    edges.to_file(f"{scenario_dir}/input_data/traffic/routing_edges.gpkg",driver="GPKG")
    ox.save_graphml(G, f"{scenario_dir}/input_data/traffic/routing_graph.graphml")

    return G, edges

def proxy_to_volume_ratio(scenario_dir,
                           matched_edges,
                           input_variable,
                           input_varaiable_name,
                           ):
    assert 'obs_vol_vph' in matched_edges.columns, 'obs_vol_vph column is missing'

    # Ensure numeric, coercing bad values to NaN
    matched_edges = matched_edges.copy()
    matched_edges['obs_vol_vph'] = pd.to_numeric(
        matched_edges['obs_vol_vph'], errors='coerce'
    )
    matched_edges[input_variable] = pd.to_numeric(
        matched_edges[input_variable], errors='coerce'
    )

    # Keep rows where BOTH obs_vol_vph and proxy variable are finite
    valid_mask = (
        matched_edges['obs_vol_vph'].notna()
        & matched_edges[input_variable].notna()
        & np.isfinite(matched_edges['obs_vol_vph'])
        & np.isfinite(matched_edges[input_variable])
    )
    matches = matched_edges[valid_mask]

    if matches.empty or len(matches) < 2:
        raise ValueError(
            f"Not enough valid data points to fit proxy-to-volume ratio. "
            f"Found {len(matches)} valid rows after filtering for finite "
            f"'obs_vol_vph' and '{input_variable}'."
        )

    X = matches[input_variable]
    y = matches['obs_vol_vph']

    # Fit OLS without intercept (pure proportionality y = coef * X)
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        raise ValueError(
            f"OLS fitting failed for proxy_to_volume_ratio "
            f"({len(matches)} data points, input_variable={input_variable!r}): {e}"
        ) from e

    slope = model.params[input_variable]
    if not np.isfinite(slope):
        raise ValueError(
            f"OLS produced non-finite slope ({slope}) for "
            f"proxy_to_volume_ratio (n={int(model.nobs)}, "
            f"input_variable={input_variable!r}). "
            f"Check for zero-variance or degenerate input data."
        )

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)

    plt.plot(X, X * slope, color='red',
             label=f'Fitted line (slope={slope:.2f})')
    plt.xlabel(input_varaiable_name)
    plt.ylabel('Observed Volume (vph)')
    plt.title(f'{input_varaiable_name} vs Observed Volume')
    plt.legend()
    plt.grid(True)

    # Save plot
    os.makedirs(f"{scenario_dir}/input_data/traffic/debug", exist_ok=True)
    plt.savefig(f"{scenario_dir}/input_data/traffic/debug/proxy_volume_scatter.png")
    plt.close()

    stats = {
        "coef": float(slope),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "n_obs": int(model.nobs),
        "std_err": float(model.bse[input_variable]),
        "t_stat": float(model.tvalues[input_variable]),
        "p_value": float(model.pvalues[input_variable]),
        "rmse": float(np.sqrt(model.mse_resid)),
    }
    stats_out_json_path = f"{scenario_dir}/input_data/traffic/debug/proxy_volume_stats.json"
    with open(stats_out_json_path, "w") as f:
        json.dump(stats, f, indent=2)

    return slope

def infer_volume(scenario_dir,
                 G=None,
                 matched_edges=None,
                 input_variable = "tomtom_sample_size",
                 input_varaiable_name = "Tomtom Sample Size",
                 save_G = False,
                 save_edges = False,
                 ):
    if G is None:
        G = ox.load_graphml(f"{scenario_dir}/input_data/traffic/routing_graph.graphml")
    if matched_edges is None:
        matched_edges = f"{scenario_dir}/input_data/traffic/routing_edges.gpkg"

    coef = proxy_to_volume_ratio(scenario_dir, matched_edges, input_variable, input_varaiable_name)

    for u, v, k, data in G.edges(keys=True, data=True):
        if 'modeled_vol_vph' in data.keys():
            logger.info("modeled_vol already calculated, skipping...")
            break
        rd = data.get(input_variable)
        if rd is not None:
            modeled_volume = float(rd) * coef
            data["modeled_vol_vph"] = modeled_volume
        else: # defaults to 0 modeled vol when TomTom is too low.
            data["modeled_vol_vph"] = 0

    if save_G:
        ox.save_graphml(G, f"{scenario_dir}/input_data/traffic/routing_graph.graphml")
    if save_edges:
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
        edges.crs = 4326
        edges.to_file(f"{scenario_dir}/input_data/traffic/routing_edges.gpkg", driver="GPKG")
        return G, edges
    return G, None

def calibrate_capacities_by_functional_class(
    scenario_dir,
    G=None,
    edges=None,
    params_path="traffic_utils/traffic_analysis_parameters.json",
    winsor_p=(0.05, 0.95),
    round_to = 50,
    save_G=False,
    save_edges=False,
):
    with open(params_path, "r") as f:
        params = json.load(f)
    fc_params = params.get("functional_classes", {})

    if G is None:
        G = ox.load_graphml(f"{scenario_dir}/input_data/traffic/routing_graph.graphml")
    if edges is None:
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Work on a copy to avoid side effects on caller's DataFrame
    edges = edges.copy()

    # ------------------------------------------------------------------
    # Ensure numeric types and filter out invalid rows
    # ------------------------------------------------------------------
    numeric_cols = [
        "ff_speed_kph",
        "obs_speed_kph",
        "vdf_alpha",
        "vdf_beta",
        "modeled_vol_vph",
        "lanes",
    ]
    for col in numeric_cols:
        if col in edges.columns:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")
        else:
            # If a required column is completely missing, fail clearly
            raise KeyError(f"Required column '{col}' is missing from edges GeoDataFrame")

    # Valid rows: all needed cols present and strictly > 0 where required
    valid_mask = (
        edges["ff_speed_kph"].notna() & (edges["ff_speed_kph"] > 0) &
        edges["obs_speed_kph"].notna() & (edges["obs_speed_kph"] > 0) &
        edges["vdf_alpha"].notna() & (edges["vdf_alpha"] > 0) &
        edges["vdf_beta"].notna() & (edges["vdf_beta"] > 0) &
        edges["modeled_vol_vph"].notna() & (edges["modeled_vol_vph"] > 0) &
        edges["lanes"].notna() & (edges["lanes"] > 0)
    )

    # Initialize as NaN so invalid rows remain NaN
    edges["implied_capacity_vph"] = np.nan
    edges["implied_cap_per_lane"] = np.nan

    # ------------------------------------------------------------------
    # Implied capacity from BPR-style speed formulation:
    # u = u_ff / (1 + alpha * (v/c)^beta)
    # ------------------------------------------------------------------
    valid_edges = edges[valid_mask].copy()
    if not valid_edges.empty:
        r = valid_edges["ff_speed_kph"] / valid_edges["obs_speed_kph"] - 1.0
        # Guard against non-positive r / alpha
        vc_term = (r / valid_edges["vdf_alpha"])
        # Any non-positive vc_term cannot be exponentiated meaningfully;
        # mark them invalid
        positive_mask = vc_term > 0
        vc_term = vc_term[positive_mask] ** (1.0 / valid_edges.loc[positive_mask, "vdf_beta"])

        implied_capacity = valid_edges.loc[positive_mask, "modeled_vol_vph"] / vc_term
        implied_cap_per_lane = implied_capacity / valid_edges.loc[positive_mask, "lanes"]

        edges.loc[valid_edges.index[positive_mask], "implied_capacity_vph"] = implied_capacity
        edges.loc[valid_edges.index[positive_mask], "implied_cap_per_lane"] = implied_cap_per_lane

    debug = {
        "params_path": params_path,
        "winsor_percentiles": winsor_p,
        "functional_classes": {},
    }
    cap_per_lane_by_fc = {}

    for fc, g in edges.groupby("functional_class"):
        fc_key = str(fc)

        # Drop NaNs in implied capacity so they don't contaminate stats
        vals = g["implied_cap_per_lane"].to_numpy()
        wts = g["modeled_vol_vph"].to_numpy()
        mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)

        if not np.any(mask):
            # No usable data for this FC; skip calibration
            continue

        vals = vals[mask]
        wts = wts[mask]

        lo_w, hi_w = np.quantile(vals, winsor_p)
        vals_w = np.clip(vals, lo_w, hi_w)
        k_fit = float(np.average(vals_w, weights=wts))

        fc_cfg = fc_params.get(fc_key, {})
        k_min = fc_cfg.get("min_capacity_perlane", None)
        k_max = fc_cfg.get("max_capacity_perlane", None)
        if k_min is not None:
            k_fit = max(k_fit, float(k_min))
        if k_max is not None:
            k_fit = min(k_fit, float(k_max))

        k_fit = round_to * round(k_fit / round_to)

        cap_per_lane_by_fc[fc] = k_fit

        debug["functional_classes"][fc_key] = {
            "n_edges": int(len(vals)),
            "cap_per_lane_vphpl_fit": float(k_fit),
            "min_capacity_perlane_param": None if k_min is None else float(k_min),
            "max_capacity_perlane_param": None if k_max is None else float(k_max),
            "raw_p10": float(np.quantile(vals, 0.1)),
            "raw_p50": float(np.quantile(vals, 0.5)),
            "raw_p90": float(np.quantile(vals, 0.9)),
            "winsor_lo": float(lo_w),
            "winsor_hi": float(hi_w),
        }

    # Assign calibrated capacities per functional class
    if cap_per_lane_by_fc:
        edges["capacity_vph"] = edges["functional_class"].map(cap_per_lane_by_fc) * edges["lanes"]

        for u, v, k, data in G.edges(keys=True, data=True):
            fc = data.get("functional_class")
            lanes = float(data.get("lanes", 1))
            if fc in cap_per_lane_by_fc and lanes is not None and lanes > 0:
                data["capacity_vph"] = cap_per_lane_by_fc[fc] * lanes

    debug_dir = f"{scenario_dir}/input_data/traffic/debug"
    os.makedirs(debug_dir, exist_ok=True)
    with open(f"{scenario_dir}/input_data/traffic/debug/capacity_calibration_by_fc.json", "w") as f:
        json.dump(debug, f, indent=2)

    if save_G:
        ox.save_graphml(G, f"{scenario_dir}/input_data/traffic/routing_graph.graphml")
    if save_edges:
        nodes, edges_out = ox.graph_to_gdfs(G, nodes=True, edges=True)
        edges_out.crs = 4326
        edges_out.to_file(f"{scenario_dir}/input_data/traffic/routing_edges.gpkg", driver="GPKG")

    return G, edges

def calibrate_VDFs(
        scenario_dir,
        G=None,
        edges=None,
        save_G=False,
        save_edges=False,
):
    if G is None:
        G = ox.load_graphml(f"{scenario_dir}/input_data/traffic/routing_graph.graphml")
    if edges is None:
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    for u, v, k, data in G.edges(keys=True, data=True):
        alpha = float(data['vdf_alpha'])
        beta = float(data['vdf_beta'])
        ff_speed = float(data['ff_speed_kph'])
        capacity = float(data['capacity_vph'])
        modeled_volume = float(data.get('modeled_vol_vph'))
        predicted_speed = ff_speed / (1 + (alpha * ((modeled_volume / capacity) ** beta)))
        data['predicted_speed_kph'] = predicted_speed
        data['predicted_speed_source'] = "modeled in volume estimation using default C/a/b"

        peak_speed = float(data['peak_speed_kph'])
        speed_diff = (predicted_speed - peak_speed) / peak_speed * 100
        data['speed_diff_percent'] = speed_diff

        calibration_factor = peak_speed / predicted_speed
        data['calibration_factor'] = calibration_factor

    if save_G:
        ox.save_graphml(G, f"{scenario_dir}/input_data/traffic/routing_graph.graphml")
    if save_edges:
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
        edges.crs = 4326
        edges.to_file(f"{scenario_dir}/input_data/traffic/routing_edges.gpkg", driver="GPKG")
    return G, edges


def add_and_calibrate_volume_attributes(
        scenario_dir,
        G=None,
        edges=None,
        save_G=True,
        save_edges=True
):
    input_dir = f"{scenario_dir}/input_data"

    if os.path.exists(f"{scenario_dir}/input_data/traffic/debug/capacity_calibration_by_fc.json"):
        logger.info("Capacity calibration already performed, skipping all volume utils...")
    else:
        if G is None:
            G = ox.load_graphml(f"{scenario_dir}/input_data/traffic/routing_graph.graphml")
        logger.info("Matching TMAS to graph")
        G, matches_gdf = match_tmas_stations_to_graph(
            G,
            f"{input_dir}/tmas/tmas.STA",
            save_matches_gpkg_to=f"{input_dir}/traffic/tmas_station_matches.gpkg",
        )

        logger.info("applying TMAS volumes to graph")
        G, matched_edges = apply_tmas_obs_volumes_to_graph(scenario_dir,
                                                           G=G,
                                                           date_start="2024-08-05",
                                                           date_end="2024-08-09",
                                                           hours=[8],
                                                           )
        G, all_edges = infer_volume(scenario_dir,
                                    G,
                                    matched_edges)
        G, all_edges = calibrate_capacities_by_functional_class(scenario_dir, G, all_edges)
        G, all_edges = calibrate_VDFs(scenario_dir,
                                      G,
                                      all_edges,
                                      save_G = True,
                                      save_edges = True)
    return G