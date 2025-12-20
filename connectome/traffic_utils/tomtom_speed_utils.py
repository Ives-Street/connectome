
"""
tomtom_osm_conflation.py

Conflate TomTom Traffic Stats segment-level speeds onto an OSM-based
MultiDiGraph (from osmnx).

Inputs:
    - osm_filepath: path to OSM XML (.osm or .osm.pbf)
    - tomtom_stats_json_path: path to TomTom Traffic Stats JSON result
    - tomtom_geojson_path: path to TomTom Traffic Stats GeoJSON result

Output:
    - GeoDataFrame of OSM edges with TomTom-based speed attributes
      attached (per timeSet/dateRange where available).

Dependencies:
    - osmnx
    - networkx
    - geopandas
    - pandas
    - shapely
    - numpy
"""

from __future__ import annotations

import logging

from functools import lru_cache
from pathlib import Path
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, MultiLineString
import shapely
from tqdm import tqdm
from difflib import SequenceMatcher


logging.basicConfig(
    level=logging.INFO,              # or DEBUG
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

TRAFFIC_PARAMS_PATH = Path(__file__).with_name("traffic_analysis_parameters.json")

@lru_cache()
def load_traffic_params(path: str | Path = TRAFFIC_PARAMS_PATH) -> Dict[str, Any]:
    """Load traffic analysis parameters (functional classes + clamps)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def normalize_street_name(name: Optional[str]) -> str:
    """Normalize street names for fuzzy comparison."""
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    # basic punctuation & suffix cleanup
    for ch in [".", ",", ";", ":", "'", '"']:
        s = s.replace(ch, " ")
    # common suffixes
    suffixes = [
        " street", " st", " avenue", " ave", " road", " rd", " boulevard", " blvd",
        " drive", " dr", " lane", " ln", " court", " ct", " highway", " hwy",
        " place", " pl", " way"
    ]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = " ".join(s.split())
    return s


def string_similarity(a: str, b: str) -> float:
    """Return a similarity score between 0 and 1."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def compute_bearing(line: LineString) -> float:
    """
    Compute approximate bearing in degrees (0–360) of a LineString
    from first to last coordinate.
    """
    if line is None or line.is_empty:
        return float("nan")
    x0, y0 = line.coords[0]
    x1, y1 = line.coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0 and dy == 0:
        return float("nan")
    angle = math.degrees(math.atan2(dx, dy))  # note: swap for N=0
    bearing = (angle + 360.0) % 360.0
    return bearing


def angle_diff_deg(a: float, b: float) -> float:
    """Absolute difference between two bearings in degrees (0–180)."""
    if math.isnan(a) or math.isnan(b):
        return 180.0
    diff = abs(a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


from shapely.geometry import LineString

def average_line_to_line_distance(
    line_a: LineString,
    line_b: LineString,
    n_samples: int = 8,
) -> float:
    """
    Approximate average distance from the *shorter* of (line_a, line_b)
    to the other line, by sampling evenly along the shorter line.

    Returns:
        Mean distance [same units as geometry CRS] or NaN if invalid.
    """
    if (
        line_a is None or line_b is None or
        line_a.is_empty or line_b.is_empty
    ):
        return float("nan")

    # Pick the shorter line as the one we sample along
    if line_a.length <= line_b.length:
        src = line_a
        dst = line_b
    else:
        src = line_b
        dst = line_a

    if src.length == 0:
        return float("nan")

    # Precompute for faster nearest-point queries
    dst_prep = shapely.prepared.prep(dst)

    # Sample along src in param space [0, 1]
    distances = []
    for i in range(n_samples):
        # e.g. for n=8 -> t in {0/7, 1/7, ..., 7/7}
        t = i / max(1, n_samples - 1)
        # Interpolate point along src
        pt = src.interpolate(t * src.length)
        # Nearest point on dst: project-then-interpolate
        proj = dst.project(pt)
        nearest = dst.interpolate(proj)
        distances.append(pt.distance(nearest))

    if not distances:
        return float("nan")

    return float(np.mean(distances))

def line_overlap_length(
    line1: LineString,
    line2: LineString,
    max_sep: float = 25.0,
    max_angle_diff: float = 45.0,
) -> float:
    """
    Fast approximate 'overlap length' between two LineStrings:

    - If they are not roughly parallel (bearing diff > max_angle_diff) → 0
    - If they are further apart than max_sep everywhere → 0
    - Otherwise, treat the overlap length as the full length of the shorter line.

    This is intentionally cheap: O(1) Shapely operations instead of
    sampling along the geometry.
    """
    if (
        line1 is None
        or line2 is None
        or line1.is_empty
        or line2.is_empty
    ):
        return 0.0

    # Global bearing check: reject non-parallel lines
    b1 = compute_bearing(line1)
    b2 = compute_bearing(line2)
    if angle_diff_deg(b1, b2) > max_angle_diff:
        return 0.0

    # Quick distance check using Shapely's distance (min distance between lines)
    if line1.distance(line2) > max_sep:
        return 0.0

    # If they are close and roughly parallel, assume they overlap
    # along the full length of the shorter one.
    len1 = line1.length
    len2 = line2.length
    min_len = min(len1, len2)
    if min_len <= 0:
        return 0.0

    return min_len


def map_frc_to_class(frc: Any) -> str:
    """Map TomTom FRC (functional road class) to a coarse category."""
    # FRC: 0 = motorway, 1 = trunk, ... 5+ = locals (exact scale may vary)
    try:
        f = int(frc)
    except Exception:
        return "unknown"
    if f <= 1:
        return "motorway"
    elif f == 2:
        return "primary"
    elif f == 3:
        return "secondary"
    elif f == 4:
        return "tertiary"
    else:
        return "local"


def map_osm_highway_to_class(highway: Any) -> str:
    """Map OSM 'highway' tag to a coarse category."""
    if not isinstance(highway, str):
        return "unknown"
    h = highway.lower()
    if h in {"motorway", "motorway_link"}:
        return "motorway"
    if h in {"trunk", "trunk_link"}:
        return "motorway"
    if h in {"primary", "primary_link"}:
        return "primary"
    if h in {"secondary", "secondary_link"}:
        return "secondary"
    if h in {"tertiary", "tertiary_link"}:
        return "tertiary"
    if h in {"residential", "living_street"}:
        return "local"
    if h in {"unclassified", "service"}:
        return "local"
    return "unknown"


def class_mismatch_penalty(tt_class: str, osm_class: str) -> float:
    """Simple penalty for mismatched road classes."""
    if tt_class == "unknown" or osm_class == "unknown":
        return 0.25
    if tt_class == osm_class:
        return 0.0
    # e.g. motorway vs local is worse than motorway vs primary
    hierarchy = ["motorway", "primary", "secondary", "tertiary", "local"]
    if tt_class not in hierarchy or osm_class not in hierarchy:
        return 0.5
    diff = abs(hierarchy.index(tt_class) - hierarchy.index(osm_class))
    return 0.2 * diff


@dataclass
class MatchConfig:
    target_crs: str = "EPSG:3857"
    max_candidate_distance_m: float = 50.0  # buffer distance
    min_overlap_ratio: float = 0.10

    overlap_weight: float = 2.0
    dist_weight: float = 1.0
    bearing_weight: float = 1.0
    name_weight: float = 1.0

    name_low_threshold: float = 0.3
    bearing_tolerance_deg: float = 30.0
    score_threshold: float = 0.5

    allow_many_to_one: bool = True
    max_matches_per_edge: int = 4


# ---------------------------------------------------------------------------
# Load & flatten TomTom data
# ---------------------------------------------------------------------------

def load_tomtom_segments(
    stats_json_path: str,
    geojson_path: str,
    target_crs: str = "EPSG:3857",
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Load TomTom Traffic Stats JSON and GeoJSON; return:
        - GeoDataFrame with geometry + flattened speed attributes
        - dateRanges dict
        - timeSets dict
    """
    with open(stats_json_path, "r") as f:
        stats = json.load(f)

    date_ranges_raw = stats.get("dateRanges", [])
    time_sets_raw = stats.get("timeSets", [])

    # Convert to dicts keyed by @id
    date_ranges = {dr["@id"]: dr for dr in date_ranges_raw}
    time_sets = {ts["@id"]: ts for ts in time_sets_raw}

    seg_results = stats["network"]["segmentResults"]

    # Flatten per-segment attributes and per-time stats
    rows: List[Dict[str, Any]] = []
    for seg in seg_results:
        base = {
            "segmentId": seg.get("segmentId"),
            "newSegmentId": seg.get("newSegmentId"),
            "speedLimit": seg.get("speedLimit"),
            "frc": seg.get("frc"),
            "streetName": seg.get("streetName"),
            "distance": seg.get("distance"),
        }
        for tr in seg.get("segmentTimeResults", []):
            t_id = tr.get("timeSet")
            d_id = tr.get("dateRange")
            suffix = f"t{t_id}_d{d_id}"
            base[f"avgSpeed_{suffix}"] = tr.get("averageSpeed")
            base[f"harmSpeed_{suffix}"] = tr.get("harmonicAverageSpeed")
            base[f"medianSpeed_{suffix}"] = tr.get("medianSpeed")
            base[f"sampleSize_{suffix}"] = tr.get("sampleSize")
            base[f"avgTT_{suffix}"] = tr.get("averageTravelTime")
            base[f"medTT_{suffix}"] = tr.get("medianTravelTime")
            base[f"ttRatio_{suffix}"] = tr.get("travelTimeRatio")
        rows.append(base)

    attr_df = pd.DataFrame(rows)

    # newSegmentId is a string in JSON; we use this as the stable key
    attr_df["newSegmentId_str"] = attr_df["newSegmentId"].astype(str)

    # Load GeoJSON for geometry
    tt_gdf = gpd.read_file(geojson_path)

    # Drop the header row (job metadata) which has no segment geometry
    tt_gdf = tt_gdf[tt_gdf["newSegmentId"].notna()].copy()

    tt_gdf["newSegmentId_str"] = tt_gdf["newSegmentId"].astype(str)

    # Merge attributes from JSON onto geometry via newSegmentId_str
    tt_gdf = tt_gdf.merge(
        attr_df.drop(columns=["segmentId", "newSegmentId"]),
        on="newSegmentId_str",
        how="left",
        validate="one_to_one",
    )

    def _pick(src, pref_y, pref_x):
        if pref_y in src.columns:
            return src[pref_y]
        if pref_x in src.columns:
            return src[pref_x]
        return np.nan

    # frc
    tt_gdf["frc"] = _pick(tt_gdf, "frc_y", "frc_x")
    # speedLimit
    tt_gdf["speedLimit"] = _pick(tt_gdf, "speedLimit_y", "speedLimit_x")
    # streetName
    tt_gdf["streetName"] = _pick(tt_gdf, "streetName_y", "streetName_x")
    # distance
    tt_gdf["distance"] = _pick(tt_gdf, "distance_y", "distance_x")

    # Reproject
    tt_gdf = tt_gdf.set_crs(epsg=4326, allow_override=True)
    tt_gdf = tt_gdf.to_crs(target_crs)

    # Derived fields (now using normalized columns)
    tt_gdf["tt_class"] = tt_gdf["frc"].apply(map_frc_to_class)
    tt_gdf["name_norm"] = tt_gdf["streetName"].apply(normalize_street_name)
    tt_gdf["bearing"] = tt_gdf["geometry"].apply(compute_bearing)
    tt_gdf["length_geom"] = tt_gdf["geometry"].length

    return tt_gdf, date_ranges, time_sets


# ---------------------------------------------------------------------------
# OSM edges preparation
# ---------------------------------------------------------------------------

def edges_from_graph(
    G: nx.MultiDiGraph, target_crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    Convert OSMnx MultiDiGraph to projected GeoDataFrame of edges,
    with some derived fields.
    """
    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # If graph has no CRS, assume WGS84 (osmnx default)
    if edges.crs is None:
        edges.set_crs(epsg=4326, inplace=True)

    edges = edges.to_crs(target_crs)

    edges = edges.reset_index()  # bring u, v, key into columns
    edges["edge_id"] = range(len(edges))
    edges["osm_class"] = edges["highway"].apply(map_osm_highway_to_class)
    edges["name_norm"] = edges["name"].apply(normalize_street_name)
    edges["bearing"] = edges["geometry"].apply(compute_bearing)
    edges["length_geom"] = edges["geometry"].length

    return edges


# ---------------------------------------------------------------------------
# OSM loading & drive-like filtering
# ---------------------------------------------------------------------------

def get_osm_graph(osm_filepath: str) -> nx.MultiDiGraph:
    """
    Load an OSM graph from XML (.osm or .osm.pbf) and filter it to a
    car-drivable network, using a heuristic similar to osmnx's
    network_type='drive':

      - Keep major/minor roads and most service roads.
      - Include living_street and road, but not busway or raceway.
      - Drop footways, cycleways, tracks, etc. where cars are obviously not allowed.
      - Drop ways where access/motor_vehicle/motorcar == 'no'.
      - Drop some service=* values like parking_aisle, driveway, drive-through, etc.
      - Preserve all other tags and do a conservative pruning of isolated nodes.
      - Retain all connected components (retain_all=True behavior).

    Returns
    -------
    nx.MultiDiGraph
        Subgraph containing only drivable edges (and their nodes).
    """
    logger.info(
        "Loading OSM graph from '%s' (highway + access-related tags)", osm_filepath
    )
    G = ox.graph_from_xml(
        osm_filepath,
        retain_all=True,
    )

    # Define which highway values are considered clearly drivable
    drivable_highways = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "residential",
        "unclassified",
        "service",
        "living_street",
        "road",
    }

    # Highway values that are clearly not car-drivable and should be removed
    non_drivable_highways = {
        "cycleway",
        "footway",
        "path",
        "pedestrian",
        "bridleway",
        "steps",
        "corridor",
        "elevator",
        "escalator",
        "bus_guideway",
        "busway",
        "raceway",
        "platform",
    }

    # Service values that we do NOT want to keep, following osmnx-like behavior
    excluded_service_values = {
        "parking_aisle",
        "driveway",
        "drive-through",
        "emergency_access",
    }

    def _is_drivable_edge(data: dict) -> bool:
        """
        Decide if an edge is part of the drivable network.

        Conservative bias: if in doubt and not explicitly non-drivable,
        we keep the edge.
        """
        # Access restrictions: drop explicit "no" for cars
        for tag in ("access", "motor_vehicle", "motorcar"):
            val = data.get(tag)
            if isinstance(val, list):
                vals = [str(v).lower() for v in val]
                if any(v == "no" for v in vals):
                    return False
            elif val is not None:
                if str(val).lower() == "no":
                    return False

        # Highway classification
        hwy = data.get("highway")
        if hwy is None:
            # No highway tag: keep (conservative)
            return True

        if isinstance(hwy, (list, set, tuple)):
            hwy_vals = [str(v).lower() for v in hwy]
        else:
            hwy_vals = [str(hwy).lower()]

        # Explicitly non-drivable highway types
        if any(v in non_drivable_highways for v in hwy_vals):
            return False

        # Service=* refinements (only meaningful if highway is service or similar)
        service_val = str(data.get("service", "")).lower()
        if "service" in hwy_vals and service_val in excluded_service_values:
            return False

        # If this looks like a typical drivable highway, keep it
        if any(v in drivable_highways for v in hwy_vals):
            return True

        # Otherwise, unknown/rare highway value: keep (conservative),
        # unless clearly excluded by other logic above.
        return True

    edges_to_remove = []

    for u, v, k, data in G.edges(keys=True, data=True):
        if not _is_drivable_edge(data):
            edges_to_remove.append((u, v, k))

    logger.info(
        "Filtering drivable network: removing %d of %d edges",
        len(edges_to_remove),
        G.number_of_edges(),
    )
    G.remove_edges_from(edges_to_remove)

    # Prune isolated nodes (degree 0) conservatively
    isolated_nodes = [n for n, deg in G.degree() if deg == 0]
    if isolated_nodes:
        logger.info(
            "Removing %d isolated nodes after edge filtering", len(isolated_nodes)
        )
        G.remove_nodes_from(isolated_nodes)

    logger.info(
        "Resulting drivable graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    return G



# ---------------------------------------------------------------------------
# Matching: scoring & assignment
# ---------------------------------------------------------------------------

def compute_match_score(
    tt_row: pd.Series,
    osm_row: pd.Series,
    cfg: MatchConfig,
    dist_norm_den: float,
) -> Tuple[float, float, float, float]:
    """
    Compute a match score between a TomTom segment and an OSM edge.

    Returns:
        (score, overlap_ratio, bearing_diff, name_score)
    """
    line_tt = tt_row.geometry
    line_osm = osm_row.geometry

    overlap_len = line_overlap_length(line_tt, line_osm)
    min_len = min(tt_row.length_geom, osm_row.length_geom)
    overlap_ratio = 0.0 if min_len == 0 else overlap_len / min_len

    if overlap_ratio < cfg.min_overlap_ratio:
        return (-1e9, overlap_ratio, 180.0, 0.0)

    # --- NEW: average distance from shorter line to the other line ---
    avg_dist = average_line_to_line_distance(line_tt, line_osm)
    if not np.isfinite(avg_dist):
        # Fall back to a large effective distance if something goes wrong
        avg_dist = dist_norm_den

    # Normalize distance to [0,1] using a rough denominator
    dist_norm = min(avg_dist / dist_norm_den, 1.0) if dist_norm_den > 0 else 0.0

    # bearing difference
    bearing_diff = angle_diff_deg(tt_row.bearing, osm_row.bearing)
    bearing_component = max(0.0, 1.0 - bearing_diff / cfg.bearing_tolerance_deg)
    bearing_component = min(bearing_component, 1.0)

    # name similarity
    name_score = string_similarity(tt_row.name_norm, osm_row.name_norm)

    # class penalty
    penalty = class_mismatch_penalty(tt_row.tt_class, osm_row.osm_class)

    spatial_score = cfg.overlap_weight * overlap_ratio - cfg.dist_weight * dist_norm
    bearing_score = cfg.bearing_weight * bearing_component
    #name_component = cfg.name_weight * name_score
    name_component = 0 # In an anecdotal test, names were more confusing than useful, and spatial + bearing are much more relevant

    score = spatial_score + bearing_score + name_component - penalty

    return score, overlap_ratio, bearing_diff, name_score


def match_tomtom_to_osm(
    tt_gdf: gpd.GeoDataFrame,
    osm_edges: gpd.GeoDataFrame,
    cfg: MatchConfig,
) -> pd.DataFrame:
    """
    For each TomTom segment, find the best-matching OSM edge using
    spatial + attribute scoring. Returns a DataFrame of matches with:

        segmentId, newSegmentId_str, edge_id, score, overlap_ratio,
        bearing_diff, name_score
    """
    # Spatial index on OSM edges
    sindex = osm_edges.sindex

    # A rough denominator for distance normalization: 2× max_candidate_distance
    dist_norm_den = 2.0 * cfg.max_candidate_distance_m

    match_rows: List[Dict[str, Any]] = []

    for idx, tt_row in tqdm(tt_gdf.iterrows(), total=len(tt_gdf), desc="Matching TT→OSM"):
        geom = tt_row.geometry
        if geom is None or geom.is_empty:
            continue

        # bounding box buffered by max_candidate_distance
        minx, miny, maxx, maxy = geom.bounds
        buff = cfg.max_candidate_distance_m
        query_box = shapely.box(minx - buff, miny - buff, maxx + buff, maxy + buff)

        candidate_idx = list(sindex.intersection(query_box.bounds))
        if not candidate_idx:
            continue

        candidates = osm_edges.iloc[candidate_idx].copy()

        best_score = -1e9
        best_row = None
        best_overlap = 0.0
        best_bearing_diff = 180.0
        best_name_score = 0.0

        for _, osm_row in candidates.iterrows():
            score, overlap_ratio, bearing_diff, name_score = compute_match_score(
                tt_row, osm_row, cfg, dist_norm_den=dist_norm_den
            )
            if score > best_score:
                best_score = score
                best_row = osm_row
                best_overlap = overlap_ratio
                best_bearing_diff = bearing_diff
                best_name_score = name_score

        if best_row is None or best_score < cfg.score_threshold:
            # no acceptable match
            continue

        match_rows.append(
            {
                "segmentId": tt_row["segmentId"],
                "newSegmentId_str": tt_row["newSegmentId_str"],
                "edge_id": best_row["edge_id"],
                "score": best_score,
                "overlap_ratio": best_overlap,
                "bearing_diff": best_bearing_diff,
                "name_score": best_name_score,
            }
        )

    matches_df = pd.DataFrame(match_rows)

    return matches_df


# ---------------------------------------------------------------------------
# Aggregation: TomTom speeds → OSM edges
# ---------------------------------------------------------------------------

def aggregate_speeds_to_edges(
    tt_gdf: gpd.GeoDataFrame,
    matches_df: pd.DataFrame,
    osm_edges: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Aggregate TomTom speeds to OSM edges AND attach an aggregate sample size.
    Writes:
      - existing speed columns (avgSpeed_*, harmSpeed_*)
      - tomtom_sample_size  (length-weighted effective sample size)
    """
    tt_with_match = tt_gdf.merge(
        matches_df[["newSegmentId_str", "edge_id", "score", "overlap_ratio"]],
        on="newSegmentId_str",
        how="inner",
    )

    speed_cols = [c for c in tt_with_match.columns if c.startswith(("avgSpeed_", "harmSpeed_"))]
    sample_cols = [c for c in tt_with_match.columns if c.startswith("sampleSize_")]

    agg_records: list[dict] = []
    for edge_id, group in tt_with_match.groupby("edge_id"):
        rec = {"edge_id": edge_id}

        seg_len = group["length_geom"].fillna(0.0).to_numpy(float)
        overlap_ratio = group["overlap_ratio"].fillna(0.0).to_numpy(float)
        L = seg_len * overlap_ratio

        # ---- speeds (unchanged) ----
        for sp_col in speed_cols:
            speeds = group[sp_col].to_numpy(float)
            mask = np.isfinite(speeds) & (speeds > 0) & np.isfinite(L) & (L > 0)
            if not mask.any():
                rec[sp_col] = np.nan
                continue
            L_used = L[mask]
            v_used = speeds[mask]
            total_L = L_used.sum()
            travel_times = L_used / v_used
            total_time = travel_times.sum()
            rec[sp_col] = float(total_L / total_time) if total_time > 0 else np.nan

        # ---- NEW: aggregate sample size ----
        # length-weighted effective sample size across all matched TT segments
        eff_samples = []
        eff_weights = []
        for sc in sample_cols:
            s = group[sc].to_numpy(float)
            mask = np.isfinite(s) & (s > 0) & np.isfinite(L) & (L > 0)
            if mask.any():
                eff_samples.append(s[mask])
                eff_weights.append(L[mask])

        if eff_samples:
            S = np.concatenate(eff_samples)
            W = np.concatenate(eff_weights)
            rec["tomtom_sample_size"] = float(np.sum(S * (W / W.sum())))
        else:
            rec["tomtom_sample_size"] = np.nan

        agg_records.append(rec)

    agg_df = pd.DataFrame(agg_records)
    return osm_edges.merge(agg_df, on="edge_id", how="left")


def default_speed_from_highway(highway: Any, fallback: float = 30.0) -> float:
    """
    Default free-flow speed in km/h based on OSM 'highway' tag.
    Tune this table as needed for your context.
    """
    if not isinstance(highway, str):
        return fallback

    h = highway.lower()
    table = {
        # highways / fast roads
        "motorway": 110.0,
        "motorway_link": 80.0,
        "trunk": 100.0,
        "trunk_link": 70.0,
        "primary": 80.0,
        "primary_link": 60.0,
        "secondary": 65.0,
        "secondary_link": 55.0,
        "tertiary": 55.0,
        "tertiary_link": 45.0,
        # local / access
        "residential": 35.0,
        "living_street": 15.0,
        "unclassified": 15.0,
        "service": 12.0,
    }

    return table.get(h, fallback)



def _normalize_highway(value: Union[str, list, None]) -> Union[str, None]:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _parse_lanes(val: Union[str, int, float, None]) -> Union[int, None]:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            return int(round(val))
        parts = [p.strip() for p in str(val).split(";") if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if not nums:
            return None
        return int(round(max(nums)))
    except Exception as e:
        logger.debug(f"Could not parse lanes='{val}': {e}")
        return None


def _is_oneway(data: dict, hwy: Union[str, None]) -> bool:
    val = data.get("oneway")
    if isinstance(val, bool):
        return val
    if val is not None:
        s = str(val).lower()
        if s in ("yes", "true", "1"):
            return True
        if s in ("no", "false", "0"):
            return False
    return hwy in {"motorway", "motorway_link"}


def _parse_maxspeed_to_kph(val: Union[str, int, float, list, None]) -> Union[float, None]:
    if val is None:
        return None
    try:
        if isinstance(val, list) and val:
            val = val[0]
        if isinstance(val, (int, float)):
            return float(val)

        s = str(val).strip().lower()
        if not s:
            return None
        if ";" in s:
            s = s.split(";", 1)[0].strip()
        if s in {"none", "signals", "walk", "variable"}:
            return None

        num_str, unit_str = "", ""
        for ch in s:
            if (ch.isdigit() or ch == "." or (ch == "-" and not num_str)):
                num_str += ch
            else:
                unit_str += ch
        if not num_str:
            return None

        speed_val = float(num_str)
        unit_str = unit_str.strip()

        if not unit_str or unit_str in {"km/h", "kph", "kmh"}:
            return speed_val
        if "mph" in unit_str:
            return speed_val * 1.60934
        return None
    except Exception as e:
        logger.debug(f"Could not parse maxspeed='{val}': {e}")
        return None


def _classify_functional_class(hwy: Union[str, None], params: Dict[str, Any]) -> str:
    h = (hwy or "").lower()
    fcs = params["functional_classes"]
    for name, cfg in fcs.items():
        tags = {t.lower() for t in cfg.get("osm_highway_tags", [])}
        if h in tags:
            return name
    return "local" if "local" in fcs else next(iter(fcs.keys()))


def add_osm_default_freeflowspeed_and_capacity(
    G: nx.MultiDiGraph,
    capacity_attr: str = "capacity_vph",
    capacity_source_attr: str = "capacity_source",
    max_lanes: int = 6,
    speed_attr: str = "ff_speed_kph",
    ff_speed_source_attr: str = "ff_speed_source",
    alpha_attr: str = "vdf_alpha",
    beta_attr: str = "vdf_beta",
    func_class_attr: str = "functional_class",
    params_path: str | Path = TRAFFIC_PARAMS_PATH,
) -> nx.MultiDiGraph:
    """
    Add free-flow speed, directional capacity, and VDF parameters (alpha/beta)
    to an OSMnx-style graph using a unified parameters JSON.

    Signalization capacity reduction:
      - Precompute signal nodes once from node tags (highway=traffic_signals).
      - Reduce capacity if the edge ENDS at a signal node (v in signal_nodes).
    """
    params = load_traffic_params(params_path)

    lanes_defaults = params.get("default_dir_lanes_by_highway", {}) or {}
    cap_defaults = params.get("capacity_per_lane_by_highway", {}) or {}
    default_cap_per_lane = int(cap_defaults.get("default", 1000))

    # Precompute signal nodes (O(N))
    def _node_is_signal(nd: dict) -> bool:
        val = nd.get("highway")
        if val is None:
            return False
        if isinstance(val, (list, tuple, set)):
            vals = [str(x).lower() for x in val]
        else:
            vals = [str(val).lower()]
        return any(x in ("traffic_signals", "traffic_signal") for x in vals)

    signal_nodes = {n for n, nd in G.nodes(data=True) if _node_is_signal(nd)}


    # Global fallback if a FC omits it
    signal_factor_default = float(params.get("signal_capacity_factor_default", 0.6))

    for u, v, k, data in G.edges(keys=True, data=True):
        hwy = _normalize_highway(data.get("highway"))
        is_oneway = _is_oneway(data, hwy)
        lanes_total = _parse_lanes(data.get("lanes"))

        cap_source_bits: list[str] = []

        # Directional lanes
        if lanes_total is not None:
            if is_oneway:
                dir_lanes = int(lanes_total)
                cap_source_bits.append("osm_lanes_oneway")
            else:
                dir_lanes = max(1, int(round(float(lanes_total) / 2.0)))
                cap_source_bits.append("osm_lanes_split_2way")
        else:
            dir_lanes = int(lanes_defaults.get(hwy, lanes_defaults.get("local", 1)))
            cap_source_bits.append("default_dir_lanes")

        if dir_lanes > max_lanes:
            dir_lanes = max_lanes
            cap_source_bits.append(f"capped_at_{max_lanes}")

        # Per-lane capacity
        cap_per_lane = int(cap_defaults.get(hwy, default_cap_per_lane))
        if hwy not in cap_defaults:
            cap_source_bits.append("default_cap_per_lane")

        capacity = dir_lanes * cap_per_lane

        # Functional class + VDF params
        fc_name = _classify_functional_class(hwy, params)
        fc_cfg = params["functional_classes"][fc_name]
        alpha = float(fc_cfg["alpha"])
        beta = float(fc_cfg["beta"])

        # Signal-end reduction (only check v, per your requirement)
        signal_factor = float(fc_cfg.get("signal_capacity_factor", signal_factor_default))
        if v in signal_nodes and signal_factor != 1.0:
            capacity = int(round(capacity * signal_factor))
            cap_source_bits.append(f"signal_factor_{signal_factor:g}")

        # Free-flow speed
        speed_source_bits: list[str] = []
        maxspeed_kph = _parse_maxspeed_to_kph(data.get("maxspeed"))
        if maxspeed_kph is not None:
            freeflow_speed = float(maxspeed_kph)
            speed_source_bits.append("osm_maxspeed")
        else:
            freeflow_speed = float(default_speed_from_highway(hwy))
            speed_source_bits.append("default_speed_from_highway")

        # Write attributes
        data[capacity_attr] = int(capacity)
        data[capacity_source_attr] = "+".join(cap_source_bits) if cap_source_bits else "unspecified"
        data[speed_attr] = freeflow_speed
        data[ff_speed_source_attr] = "+".join(speed_source_bits) if speed_source_bits else "unspecified"
        data[func_class_attr] = fc_name
        data[alpha_attr] = alpha
        data[beta_attr] = beta

    return G

###
# BPR volume/capacity functions
###

def bpr_time_from_volume(
    ff_time_s: float,
    volume_vph: float,
    capacity_vph: float,
    alpha: float = 0.15,
    beta: float = 4.0,
    max_factor: float = 10.0,
) -> float:
    """
    BPR-like forward function: (ff_time, volume, capacity) -> congested_time.
    """
    if ff_time_s is None or ff_time_s <= 0:
        return ff_time_s
    if capacity_vph is None or capacity_vph <= 0 or volume_vph is None:
        return ff_time_s

    x = volume_vph / capacity_vph
    tt = ff_time_s * (1.0 + alpha * (x ** beta))
    # cap to avoid numeric blow-ups
    return min(tt, ff_time_s * max_factor)

def bpr_volume_from_time(
    cong_time_s: float,
    ff_time_s: float,
    capacity_vph: float,
    alpha: float = 0.15,
    beta: float = 4.0,
    max_x: float = 5.0,
) -> float:
    """
    Inverse BPR: (congested_time, free_flow_time, capacity) -> implied volume.

    Returns volume_vph. If cong_time <= ff_time, returns 0.
    """
    if (cong_time_s is None or cong_time_s <= 0 or
        ff_time_s is None or ff_time_s <= 0 or
        capacity_vph is None or capacity_vph <= 0):
        return 0.0

    ratio = cong_time_s / ff_time_s

    if ratio <= 1.0:
        # no congestion (or measurement noise): treat as v ~ 0
        return 0.0

    base = (ratio - 1.0) / alpha
    if base <= 0:
        return 0.0

    x = base ** (1.0 / beta)  # v/c
    x = min(x, max_x)         # avoid insane v/c

    return x * capacity_vph

def time_from_speed(length_m: float, speed_kph: float) -> float:
    """
    Convert speed [km/h] to time [s] for a given length [m].
    """
    if length_m is None or length_m <= 0 or speed_kph is None or speed_kph <= 0:
        return None
    v_ms = speed_kph * 1000.0 / 3600.0
    return length_m / v_ms


def speed_from_time(length_m: float, time_s: float) -> float:
    """
    Convert time [s] to speed [km/h] for a given length [m].
    """
    if length_m is None or length_m <= 0 or time_s is None or time_s <= 0:
        return None
    v_ms = length_m / time_s
    return v_ms * 3.6


def volume_from_speeds_bpr(
    length_m: float,
    ff_speed_kph: float,
    obs_speed_kph: float,
    capacity_vph: float,
    alpha: float = 0.15,
    beta: float = 4.0,
    uncongested_tol: float = 0.05,  # 5% tolerance
    max_vc_ratio: float = 5.0,      # clamp v/c
) -> float:
    """
    Infer volume from observed speed using inverse BPR with:
      - uncongested clamp: t_obs <= (1+tol)*t0 -> v = 0
      - max v/c clamp: v/c <= max_vc_ratio
    """
    if (
        length_m is None or length_m <= 0 or
        ff_speed_kph is None or ff_speed_kph <= 0 or
        obs_speed_kph is None or obs_speed_kph <= 0 or
        capacity_vph is None or capacity_vph <= 0
    ):
        return 0.0

    ff_time_s = time_from_speed(length_m, ff_speed_kph)
    cong_time_s = time_from_speed(length_m, obs_speed_kph)
    if ff_time_s is None or cong_time_s is None:
        return 0.0

    # Clamp 1: effectively uncongested (or measurement noise)
    if cong_time_s <= ff_time_s * (1.0 + uncongested_tol):
        return 0.0

    # Inverse BPR
    ratio = cong_time_s / ff_time_s
    base = (ratio - 1.0) / alpha
    if base <= 0:
        return 0.0

    x = base ** (1.0 / beta)  # v/c

    # Clamp 2: cap v/c
    x = min(x, max_vc_ratio)

    return x * capacity_vph

def speed_from_volume_bpr(
    length_m: float,
    ff_speed_kph: float,
    volume_vph: float,
    capacity_vph: float,
    alpha: float = 0.15,
    beta: float = 4.0,
    max_factor: float = 10.0,
) -> float:
    ff_time_s = time_from_speed(length_m, ff_speed_kph)
    cong_time_s = bpr_time_from_volume(
        ff_time_s, volume_vph, capacity_vph,
        alpha=alpha, beta=beta, max_factor=max_factor
    )
    return speed_from_time(length_m, cong_time_s)


def estimate_volumes_from_obs_speeds(
    G: nx.MultiDiGraph,
    length_attr: str = "length",
    ff_speed_attr: str = "ff_speed_kph",
    obs_speed_attr: str = "obs_speed_kph",
    capacity_attr: str = "capacity_vph",
    out_volume_attr: str = "volume_vph",
    alpha: float = 0.15,
    beta: float = 4.0,
    uncongested_tol: float = 0.05,
    max_vc_ratio: float = 3.0,
) -> None:
    """
    For each edge, infer volume from observed congested speed using
    inverse BPR, with:
      - uncongested clamp (t_obs <= (1+tol)*t0 -> v=0)
      - max v/c clamp (v/c <= max_vc_ratio).

    Modifies G in-place, writing volumes to out_volume_attr.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get(length_attr)
        ff_speed_kph = data.get(ff_speed_attr)
        obs_speed_kph = data.get(obs_speed_attr)
        cap = data.get(capacity_attr)

        vol = volume_from_speeds_bpr(
            length_m=length_m,
            ff_speed_kph=ff_speed_kph,
            obs_speed_kph=obs_speed_kph,
            capacity_vph=cap,
            alpha=alpha,
            beta=beta,
            uncongested_tol=uncongested_tol,
            max_vc_ratio=max_vc_ratio,
        )

        data[out_volume_attr] = vol


def calculate_traversal_time_from_available_speeds(
    G: nx.MultiDiGraph,
    length_attr: str = "length",
    freeflow_speed_attr: str = "ff_speed_kph",
    obs_speed_attr: str = "obs_speed_kph",
    scenario_speed_attr: str = "scenario_speed_kph",
    travel_time_attr: str = "traversal_time_sec",
    traversal_time_source_attr: str = "traversal_time_source",
    params_path: str | Path = TRAFFIC_PARAMS_PATH,
) -> None:
    """
    Calculate traversal time for each edge based on available speeds.
    Priority order: scenario > observed > freeflow speeds.
    Applies optional penalties to short *_link edges to discourage "off-and-on" cheating.
    Updates graph in-place.
    """
    logger.info("Calculating traversal times from available speeds")

    params = load_traffic_params(params_path)

    # Link penalties (robust defaults)
    link_penalties = params.get("link_penalties") or {}
    link_penalty_sec = float(link_penalties.get("link_traversal_time_penalty_sec", 0.0) or 0.0)
    link_penalty_max_len_m = float(link_penalties.get("link_traversal_time_penalty_max_length_m", 0.0) or 0.0)

    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get(length_attr)
        if not isinstance(length_m, (int, float)) or length_m <= 0:
            logger.warning(f"Invalid length for edge ({u}, {v}, {k})")
            continue

        # Speeds in priority order
        speed_kmh = None
        source = None
        if scenario_speed_attr in data:
            speed_kmh = data.get(scenario_speed_attr)
            source = "scenario"
        elif obs_speed_attr in data:
            speed_kmh = data.get(obs_speed_attr)
            source = "observed"
        elif freeflow_speed_attr in data:
            speed_kmh = data.get(freeflow_speed_attr)
            source = "freeflow"

        if not isinstance(speed_kmh, (int, float)) or speed_kmh <= 0:
            logger.warning(f"No valid speed found for edge ({u}, {v}, {k})")
            continue

        speed_ms = speed_kmh * (1000.0 / 3600.0)
        time_sec = length_m / speed_ms

        # Penalize short *_link edges
        if link_penalty_sec > 0 and link_penalty_max_len_m > 0:
            hwy = _normalize_highway(data.get("highway"))
            is_link = isinstance(hwy, str) and hwy.endswith("_link")
            if is_link and length_m <= link_penalty_max_len_m:
                time_sec += link_penalty_sec
                source = f"{source}+link_penalty"

        data[travel_time_attr] = float(time_sec)
        data[traversal_time_source_attr] = source

    logger.info("Finished calculating traversal times")


###
# Initialization - build routable graph function, including init estimation of volume and capacity
###


def build_routable_graph_from_edges(
    G: nx.MultiDiGraph,
    edges_with_speeds: gpd.GeoDataFrame,
    speed_col: str = "harmSpeed_t2_d1",
    length_col: str = "length_geom",
    ff_speed_attr: str = "ff_speed_kph",
    obs_speed_attr: str = "obs_speed_kph",
    travel_time_attr: str = "traversal_time_sec",
    obs_speed_source_attr: str = "speed_source",
    tomtom_sample_attr: str = "tomtom_sample_size",   # NEW
) -> nx.MultiDiGraph:
    """
    Create a routable graph and attach TomTom speeds + sample size to edges.
    """
    G_modified = G.copy()
    G_modified = add_osm_default_freeflowspeed_and_capacity(G_modified)

    for _, row in edges_with_speeds.iterrows():
        u, v, k = row["u"], row["v"], row["key"]
        if not G_modified.has_edge(u, v, k):
            continue

        edge_data = G_modified[u][v][k]

        speed_kmh = row.get(speed_col)
        if isinstance(speed_kmh, (int, float)) and speed_kmh > 0:
            edge_data[obs_speed_attr] = float(speed_kmh)
            edge_data[obs_speed_source_attr] = "tomtom"

        # attach sample size (even if speed missing)
        ss = row.get("tomtom_sample_size")
        if isinstance(ss, (int, float)) and np.isfinite(ss):
            edge_data[tomtom_sample_attr] = float(ss)

    calculate_traversal_time_from_available_speeds(G_modified)
    return G_modified


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
def conflate_tomtom_to_osm(
        osm_filepath: str,
        tomtom_stats_json_path: str,
        tomtom_geojson_path: str,
        speed_col: str = "obs_speed_kph",
        cfg: Optional[MatchConfig] = None,
        debug_gpkg_prefix: Optional[str] = None,
        save_graphml_to: Optional[str] = None,
        save_nodes_to: Optional[str] = None,
        save_edges_to: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    High-level function:

    1. Load & flatten TomTom stats and geometry.
    2. Load OSM graph from XML and filter to a drivable (car) network.
    3. Convert OSM MultiDiGraph to projected edges GeoDataFrame.
    4. Match each TomTom segment to best OSM edge.
    5. Aggregate TomTom speeds per OSM edge.
    6. Build a routable graph with TomTom speeds and optionally save it.

    Parameters
    ----------
    osm_filepath : str
        Path to OSM XML (.osm or .osm.pbf).
    tomtom_stats_json_path : str
        Path to TomTom Traffic Stats JSON result.
    tomtom_geojson_path : str
        Path to TomTom Traffic Stats GeoJSON result.
    speed_col : str
        Name of the TomTom speed column (e.g. 'harmSpeed_t2_d1') to use
        when building the routable graph.
    cfg : MatchConfig, optional
        Matching configuration (projection, thresholds, weights).
    debug_gpkg_prefix : str, optional
        If provided, write unmatched TomTom segments and OSM edges as GPKGs
        with this prefix.
    save_graphml_to : str, optional
        If provided, save the routable graph as GraphML to this path.
    save_nodes_to : str, optional
        If provided, save the routable graph's nodes GeoDataFrame (GPKG).
    save_edges_to : str, optional
        If provided, save the routable graph's edges GeoDataFrame (GPKG).

    Returns
    -------
    edges_with_speeds : GeoDataFrame
        OSM edges with TomTom attributes.
    date_ranges : dict
        dateRange metadata (by @id).
    time_sets : dict
        timeSet metadata (by @id).
    matches_df : DataFrame
        Segment-to-edge matches.
    """
    logger.info(
        "Starting TomTom–OSM conflation: osm='%s', stats_json='%s', geojson='%s'",
        osm_filepath,
        tomtom_stats_json_path,
        tomtom_geojson_path,
    )

    if cfg is None:
        cfg = MatchConfig()
        logger.debug("No MatchConfig provided; using default MatchConfig()")
    else:
        logger.debug(
            "Using provided MatchConfig with target_crs=%s and thresholds=%s",
            getattr(cfg, "target_crs", None),
            {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
        )

    # 0. Load drivable OSM graph from file
    G_drive = get_osm_graph(osm_filepath)

    # 1. TomTom
    logger.info("Loading TomTom segments and metadata")
    tt_gdf, date_ranges, time_sets = load_tomtom_segments(
        tomtom_stats_json_path,
        tomtom_geojson_path,
        target_crs=cfg.target_crs,
    )
    logger.info(
        "Loaded %d TomTom segments; date_ranges=%d, time_sets=%d",
        len(tt_gdf),
        len(date_ranges),
        len(time_sets),
    )

    # 2. OSM edges
    logger.info("Converting drivable OSM graph to edges GeoDataFrame (target_crs=%s)", cfg.target_crs)
    osm_edges = edges_from_graph(G_drive, target_crs=cfg.target_crs)
    logger.info("Extracted %d drivable OSM edges from graph", len(osm_edges))

    # 3. Matching
    logger.info("Matching TomTom segments to OSM edges")
    matches_df = match_tomtom_to_osm(tt_gdf, osm_edges, cfg)
    logger.info(
        "Completed matching: %d matches, %d unique OSM edges with at least one match",
        len(matches_df),
        matches_df["edge_id"].nunique() if "edge_id" in matches_df.columns else -1,
    )

    # 3b. Debug: report and optionally save unmatched TomTom segments / OSM edges
    total_tt = len(tt_gdf)
    total_osm = len(osm_edges)
    total_matches = len(matches_df)

    if total_matches == 0:
        logger.info(f"[DEBUG] No matches found at all. TomTom segments: {total_tt}, OSM edges: {total_osm}")
        unmatched_tt = tt_gdf.copy()
        unmatched_osm = osm_edges.copy()
    else:
        matched_tt_ids = set(matches_df["newSegmentId_str"].unique())
        matched_edge_ids = set(matches_df["edge_id"].unique())
        unmatched_tt = tt_gdf[~tt_gdf["newSegmentId_str"].isin(matched_tt_ids)].copy()
        unmatched_osm = osm_edges[~osm_edges["edge_id"].isin(matched_edge_ids)].copy()
        logger.info(
            f"[DEBUG] Matches: {total_matches} | "
            f"TomTom matched: {total_tt - len(unmatched_tt)}/{total_tt} | "
            f"OSM matched: {total_osm - len(unmatched_osm)}/{total_osm}"
        )

    if debug_gpkg_prefix is not None:
        # Save unmatched TomTom segments
        try:
            unmatched_tt.to_file(f"{debug_gpkg_prefix}_tomtom_unmatched.gpkg", driver="GPKG")
            logger.info(f"[DEBUG] Wrote unmatched TomTom segments to {debug_gpkg_prefix}_tomtom_unmatched.gpkg")
        except Exception as e:
            logger.info(f"[DEBUG] Failed to write unmatched TomTom GPKG: {e}")
        # Save unmatched OSM edges
        try:
            unmatched_osm.to_file(f"{debug_gpkg_prefix}_osm_unmatched.gpkg", driver="GPKG")
            logger.info(f"[DEBUG] Wrote unmatched OSM edges to {debug_gpkg_prefix}_osm_unmatched.gpkg")
        except Exception as e:
            logger.info(f"[DEBUG] Failed to write unmatched OSM GPKG: {e}")

    # 4. Aggregation
    logger.info("Aggregating TomTom speeds to OSM edges")
    edges_with_speeds = aggregate_speeds_to_edges(tt_gdf, matches_df, osm_edges)
    logger.info(
        "Aggregation complete: %d edges with TomTom speed attributes",
        len(edges_with_speeds),
    )

    # 5. Build routable graph and optionally save
    logger.info("Building routable graph from drivable OSM network and TomTom speeds")
    G_routable = build_routable_graph_from_edges(
        G_drive,
        edges_with_speeds,
    )

    nodes, edges = ox.graph_to_gdfs(G_routable)

    if save_graphml_to is not None:
        logger.info(f"Saving routable graph as GraphML to {save_graphml_to}")
        ox.save_graphml(G_routable, save_graphml_to)

    if save_nodes_to is not None:
        logger.info(f"Saving routable graph nodes to {save_nodes_to}")
        nodes.to_file(save_nodes_to, driver="GPKG")

    if save_edges_to is not None:
        logger.info(f"Saving routable graph edges to {save_edges_to}")
        edges.to_file(save_edges_to, driver="GPKG")

    logger.debug(
        "TomTom–OSM conflation finished. Returning edges_with_speeds, date_ranges, time_sets, matches_df"
    )
    return edges_with_speeds, date_ranges, time_sets, matches_df
