
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

logging.basicConfig(
    level=logging.INFO,              # or DEBUG
    format="%(asctime)s %(levelname)s %(message)s"
)

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
import shapely
from tqdm import tqdm
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


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

def load_drivable_osm_graph(osm_filepath: str) -> nx.MultiDiGraph:
    """
    Load an OSM graph from XML (.osm or .osm.pbf) and filter it to a
    car-drivable network, using a heuristic similar to osmnx's
    network_type='drive':

      - Parse only highway + access-related tags via osmnx's `tags` argument.
      - Keep major/minor roads and most service roads.
      - Drop footways, cycleways, tracks, etc.
      - Drop ways where access/motor_vehicle/motorcar == 'no'.
      - Drop some service=* values like parking_aisle, driveway, private.

    Returns
    -------
    nx.MultiDiGraph
        Subgraph containing only drivable edges (and their nodes).
    """
    logger.info("Loading OSM graph from '%s' (highway + access-related tags)", osm_filepath)
    G_raw = ox.graph_from_xml(
        osm_filepath,
        retain_all=True,
    )

    return G_raw

    # Failed attempt to vibe-code a filter to only car roads. This doesn't really matter.

    # logger.info("Filtering OSM graph to edges approximating 'drive' network_type")
    # nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_raw, nodes=True, edges=True)
    #
    # # Highways that can plausibly be driven by cars
    # drive_highways = {
    #     "motorway", "motorway_link",
    #     "trunk", "trunk_link",
    #     "primary", "primary_link",
    #     "secondary", "secondary_link",
    #     "tertiary", "tertiary_link",
    #     "residential", "living_street",
    #     "unclassified", "service",
    #     "road",
    # }
    #
    # # Highways to definitely exclude from a "drive" network
    # non_drive_highways = {
    #     "footway", "path", "pedestrian", "steps", "track",
    #     "bridleway", "cycleway", "corridor", "construction",
    # }
    #
    # # service=* subtypes we want to exclude, similar to osmnx's drive filter
    # excluded_service_values = {
    #     "parking_aisle", "driveway", "private",
    # }
    #
    # def _norm_list_or_str(val: Any) -> List[str]:
    #     if isinstance(val, (list, tuple, set)):
    #         return [str(v).lower() for v in val]
    #     if isinstance(val, str):
    #         return [val.lower()]
    #     return []
    #
    # def _is_drive_edge(row: pd.Series) -> bool:
    #     hw_vals = _norm_list_or_str(row.get("highway"))
    #
    #     # Must have at least one highway tag
    #     if not hw_vals:
    #         return False
    #
    #     # If any highway value is in the non-drive list → exclude
    #     if any(h in non_drive_highways for h in hw_vals):
    #         return False
    #
    #     # Require at least one value in drive_highways
    #     if not any(h in drive_highways for h in hw_vals):
    #         return False
    #
    #     # Respect explicit "no" access for motor vehicles
    #     access = str(row.get("access", "")).lower()
    #     motor_vehicle = str(row.get("motor_vehicle", "")).lower()
    #     motorcar = str(row.get("motorcar", "")).lower()
    #
    #     if access == "no" or motor_vehicle == "no" or motorcar == "no":
    #         return False
    #
    #     # Refine service roads: exclude certain service=* values
    #     if any(h == "service" for h in hw_vals):
    #         serv_vals = _norm_list_or_str(row.get("service"))
    #         if any(s in excluded_service_values for s in serv_vals):
    #             return False
    #
    #     return True
    #
    # drive_mask = edges_gdf.apply(_is_drive_edge, axis=1)
    # edges_drive = edges_gdf[drive_mask].copy()
    #
    # if edges_drive.empty:
    #     logger.warning(
    #         "No drivable edges found in OSM file after applying drive-like filter"
    #     )
    #
    # edge_ids = list(zip(edges_drive["u"], edges_drive["v"], edges_drive["key"]))
    # G_drive = G_raw.edge_subgraph(edge_ids).copy()
    #
    # logger.info(
    #     "Drive-like filter kept %d of %d edges",
    #     len(edges_drive),
    #     len(edges_gdf),
    # )
    return G_drive


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

    # centroids distance
    c_tt = line_tt.centroid
    c_osm = line_osm.centroid
    dist = c_tt.distance(c_osm)
    # normalize distance to [0,1] using a rough denominator
    dist_norm = min(dist / dist_norm_den, 1.0) if dist_norm_den > 0 else 0.0

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
    name_component = cfg.name_weight * name_score

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
    For each OSM edge, aggregate TomTom speeds (per timeSet/dateRange)
    from all matched segments using length-weighted average (and sampleSize
    as an additional weight if present).

    Returns a new edges GeoDataFrame with TomTom-based speed columns.
    """

    # Merge match info into TomTom rows
    tt_with_match = tt_gdf.merge(
        matches_df[["newSegmentId_str", "edge_id", "score", "overlap_ratio"]],
        on="newSegmentId_str",
        how="inner",
    )

    # Identify all speed columns (avgSpeed_tX_dY etc.)
    speed_cols = [c for c in tt_with_match.columns if c.startswith(("avgSpeed_", "harmSpeed"))]

    # Build per-edge aggregates
    agg_records: List[Dict[str, Any]] = []
    for edge_id, group in tt_with_match.groupby("edge_id"):
        rec: Dict[str, Any] = {"edge_id": edge_id}

        # Base segment length and overlap ratio (must exist due to merge above)
        seg_len = group["length_geom"].fillna(0.0).to_numpy(dtype=float)
        overlap_ratio = group["overlap_ratio"].fillna(0.0).to_numpy(dtype=float)

        # Effective length actually overlapping this edge
        L = seg_len * overlap_ratio

        for sp_col in speed_cols:
            # Pull speeds for this timeSet/dateRange
            speeds = group[sp_col].to_numpy(dtype=float)

            # Valid where we have finite, positive speed and positive overlap length
            mask = (
                np.isfinite(speeds)
                & (speeds > 0)
                & np.isfinite(L)
                & (L > 0)
            )
            if not mask.any():
                rec[sp_col] = np.nan
                continue

            L_used = L[mask]
            v_used = speeds[mask]

            total_L = L_used.sum()
            if total_L <= 0:
                rec[sp_col] = np.nan
                continue

            # Total travel time along the covered portions
            # time_i = L_i / v_i   (units: distance / (distance/time) = time)
            travel_times = L_used / v_used
            total_time = travel_times.sum()
            if total_time <= 0:
                rec[sp_col] = np.nan
                continue

            # Effective edge speed = total distance / total time
            rec[sp_col] = float(total_L / total_time)

        agg_records.append(rec)

    agg_df = pd.DataFrame(agg_records)

    # Merge onto edges
    edges_with_speeds = osm_edges.merge(agg_df, on="edge_id", how="left")

    return edges_with_speeds


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
        "residential": 40.0,
        "living_street": 20.0,
        "unclassified": 40.0,
        "service": 25.0,
    }

    return table.get(h, fallback)


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


def build_routable_graph_from_edges(
    G: nx.MultiDiGraph,
    edges_with_speeds: gpd.GeoDataFrame,
    speed_col: str,
    length_col: str = "length_geom",
    speed_attr: str = "tt_speed_kmh",
    travel_time_attr: str = "traversal_time_sec",
    speed_source_attr: str = "speed_source",
) -> nx.MultiDiGraph:
    """
    Create a routable osmnx MultiDiGraph from the conflated edges table.

    For each edge (u, v, key):
      - If a TomTom speed is available in `speed_col`, use it.
      - Otherwise, estimate speed from the OSM 'highway' tag via a lookup table.
      - Compute traversal time in seconds from length / speed.
      - Attach:
          edge[speed_attr]        = chosen speed (km/h)
          edge[travel_time_attr]  = traversal time (sec)
          edge[speed_source_attr] = "tomtom" or "fallback"

    Returns:
        New MultiDiGraph with added attributes, ready for routing with
        weight=travel_time_attr.
    """
    H = G.copy()

    # Ensure we have u, v, key in the edges GDF
    if not {"u", "v", "key"}.issubset(edges_with_speeds.columns):
        raise ValueError("edges_with_speeds must have 'u', 'v', 'key' columns from graph_to_gdfs")

    for _, row in edges_with_speeds.iterrows():
        u = row["u"]
        v = row["v"]
        key = row["key"]

        if not H.has_edge(u, v, key):
            continue

        edge_data = H[u][v][key]

        # 1) Get length in meters: prefer conflated length_col, else edge 'length'
        length_m = row.get(length_col, np.nan)
        if not (isinstance(length_m, (int, float)) and np.isfinite(length_m) and length_m > 0):
            length_m = edge_data.get("length", None)
        if length_m is None or length_m <= 0:
            # No usable length: can't compute traversal time
            continue

        # 2) Try TomTom speed first
        speed_kmh = row.get(speed_col, np.nan)
        src = "tomtom"

        if not (isinstance(speed_kmh, (int, float)) and np.isfinite(speed_kmh) and speed_kmh > 0):
            # 3) Fallback to highway-based default
            hw = row.get("highway", edge_data.get("highway", None))
            speed_kmh = default_speed_from_highway(hw)
            src = "fallback"

        if speed_kmh is None or not np.isfinite(speed_kmh) or speed_kmh <= 0:
            continue

        # 4) Compute traversal time in seconds
        speed_mps = speed_kmh * (1000.0 / 3600.0)
        travel_time_sec = length_m / speed_mps

        edge_data[speed_attr] = float(speed_kmh)
        edge_data[travel_time_attr] = float(travel_time_sec)
        edge_data[speed_source_attr] = src

    return H


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
def conflate_tomtom_to_osm(
        osm_filepath: str,
        tomtom_stats_json_path: str,
        tomtom_geojson_path: str,
        speed_col: str = "harmSpeed_t2_d1",
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
    G_drive = load_drivable_osm_graph(osm_filepath)

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
        speed_col=speed_col,
        length_col="length_geom",
        speed_attr=f"tt_{speed_col}",
        travel_time_attr="traversal_time_sec",
        speed_source_attr="speed_source",
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
