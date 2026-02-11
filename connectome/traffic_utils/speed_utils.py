
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
from collections import deque

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

#minizing list for memory efficiency
custom_tags = ['access', 'area', 'highway', 'lanes', 'maxspeed', 'name', 'oneway', 'ref', 'service', 'toll', 'bicycle']
ox.settings.useful_tags_way = custom_tags

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

def remove_isolated_and_orphan_components(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Clean a routing graph by:
      1) Removing degree-0 nodes (true isolates).
      2) Keeping only the largest weakly connected component.

    This prevents rep_points from snapping onto tiny fragments or
    completely isolated nodes that cannot reach the main network.
    """

    # If anything remains, keep only the largest weakly connected component
    if not nx.is_empty(G):
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    return G

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
def get_osm_graph(
    osm_filepath: str,
    filtered_osm_filepath: str,
    *,
    include_residential: bool = False,
    include_unclassified: bool = False,
    include_living_street: bool = False,
    include_road: bool = False,
    include_service: bool = False,
) -> nx.MultiDiGraph:
    """
    Stream-filter an OSM XML/PBF file to a new .osm XML extract using osmium,
    then load it into a NetworkX MultiDiGraph via OSMnx.

    This avoids constructing the full graph in memory before filtering, but it
    does keep an in-memory set of node IDs referenced by kept ways.

    Parameters
    ----------
    osm_filepath : str
        Input .osm or .osm.pbf file.
    filtered_osm_filepath : str
        Output .osm file path to write the filtered extract to.
    include_residential, include_unclassified, include_living_street, include_road, include_service : bool
        Whether to include those highway classes and (for service) any way with service=*.

    Returns
    -------
    nx.MultiDiGraph
        Drivable graph built from the filtered extract.
    """
    import osmium as osm

    logger.info("Streaming-filtering OSM input '%s' -> '%s'", osm_filepath, filtered_osm_filepath)

    # Base drivable set (always-on)
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
    }
    # Optional classes
    if include_residential:
        drivable_highways.add("residential")
    if include_unclassified:
        drivable_highways.add("unclassified")
    if include_living_street:
        drivable_highways.add("living_street")
    if include_road:
        drivable_highways.add("road")
    if include_service:
        drivable_highways.add("service")

    # Explicitly non-drivable highway types
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
    # Only treat service as non-drivable when we are not including it
    if not include_service:
        non_drivable_highways.add("service")
    if not include_residential:
        non_drivable_highways.add("residential")

    def _split_hwy_values(hwy_val: str) -> list[str]:
        # OSM commonly stores multiple highway values as "foo;bar"
        parts = [p.strip().lower() for p in str(hwy_val).split(";")]
        return [p for p in parts if p]

    def _tag_value(tags: osm.osm.Tags, key: str) -> str | None:
        # pyosmium tags behave like a mapping
        try:
            v = tags.get(key)
        except Exception:
            v = None
        return v

    def _is_drivable_way(w: osm.osm.Way) -> bool:
        """
        Decide if a WAY is part of the drivable network.
        Conservative bias: if in doubt and not explicitly excluded, keep it.
        """
        tags = w.tags

        # Access restrictions: drop explicit "no" for cars
        for key in ("access", "motor_vehicle", "motorcar"):
            val = _tag_value(tags, key)
            if val is not None and str(val).strip().lower() == "no":
                return False

        # If service=* exists and we are excluding service, drop
        if not include_service:
            svc = _tag_value(tags, "service")
            if svc is not None and str(svc).strip() != "":
                return False

        hwy = _tag_value(tags, "highway")
        if hwy is None:
            # No highway tag: keep (conservative)
            return True

        hwy_vals = _split_hwy_values(hwy)

        # Explicitly non-drivable highway types
        if any(v in non_drivable_highways for v in hwy_vals):
            return False

        # Typical drivable highway types
        if any(v in drivable_highways for v in hwy_vals):
            return True

        # Unknown/rare highway value: keep (conservative)
        return True

    # -------------------------
    # Pass 1: collect kept ways + required node IDs
    # -------------------------
    kept_way_ids: set[int] = set()
    needed_node_ids: set[int] = set()

    class Pass1(osm.SimpleHandler):
        def way(self, w):
            if _is_drivable_way(w):
                kept_way_ids.add(w.id)
                for n in w.nodes:
                    needed_node_ids.add(n.ref)

    logger.info("Pass 1/2: scanning ways + collecting referenced node IDs")
    Pass1().apply_file(osm_filepath, locations=False)

    logger.info(
        "Pass 1 complete: kept %d ways; referenced %d nodes",
        len(kept_way_ids),
        len(needed_node_ids),
    )

    # -------------------------
    # Pass 2: write nodes + ways to filtered output
    # -------------------------
    writer = osm.SimpleWriter(filtered_osm_filepath)

    class Pass2(osm.SimpleHandler):
        def node(self, n):
            if n.id in needed_node_ids:
                writer.add_node(n)

        def way(self, w):
            if w.id in kept_way_ids:
                writer.add_way(w)

        # Relations are typically not required for drivable graphs.
        # If you later discover you need them (rare for routing),
        # you can implement relation() similarly.

    logger.info("Pass 2/2: writing filtered extract to '%s'", filtered_osm_filepath)
    Pass2().apply_file(osm_filepath, locations=False)
    writer.close()

    # -------------------------
    # Load filtered extract into a graph
    # -------------------------
    logger.info("Loading filtered OSM graph from '%s'", filtered_osm_filepath)
    G = ox.graph_from_xml(filtered_osm_filepath, retain_all=True)

    # Prune isolates (degree 0) aggressively
    # TODO add this as a configurable parameter - some cities may actually have islands.
    G = remove_isolated_and_orphan_components(G)

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

#new LLM attempt
def untested_match_tomtom_to_osm(
    tt_gdf: gpd.GeoDataFrame,
    osm_edges: gpd.GeoDataFrame,
    cfg: MatchConfig,
) -> pd.DataFrame:
    """
    For each TomTom segment, find the best-matching OSM edge using
    spatial + attribute scoring. Returns a DataFrame of matches with:

        segmentId, newSegmentId_str, edge_id, score, overlap_ratio,
        bearing_diff, name_score

    Efficiency notes (accuracy-preserving):
    - Reuses the OSM spatial index (and a few invariants) across calls for the
      *same* `osm_edges` object, so running twice with different TomTom datasets
      does not rebuild the index.
    - Avoids constructing shapely boxes; queries the spatial index by bounds.
    - Avoids copying candidate slices and avoids pandas iterrows on candidates.
    """
    # ---- lightweight per-process cache keyed by the *object identity* of osm_edges ----
    # This is intentionally identity-based: it avoids expensive hashing and works well
    # when you reuse the same GeoDataFrame across multiple runs.
    global _MATCH_TT_OSM_CACHE  # type: ignore
    try:
        _MATCH_TT_OSM_CACHE
    except NameError:
        _MATCH_TT_OSM_CACHE = {}

    cache_key = (id(osm_edges), float(cfg.max_candidate_distance_m))
    cached = _MATCH_TT_OSM_CACHE.get(cache_key)

    # If `osm_edges` is reused but replaced/rebuilt, id() will differ and we rebuild cache.
    # If `osm_edges` is mutated in-place between calls, you should manually clear the cache
    # (restart process) to guarantee consistency.
    if cached is None:
        cached = {
            "sindex": osm_edges.sindex,
            # distance normalization is a function of cfg only
            "dist_norm_den": 2.0 * float(cfg.max_candidate_distance_m),
        }
        _MATCH_TT_OSM_CACHE[cache_key] = cached

    sindex = cached["sindex"]
    dist_norm_den = cached["dist_norm_den"]

    buff = float(cfg.max_candidate_distance_m)

    match_rows: List[Dict[str, Any]] = []

    # Keep tt_row as a Series (for compute_match_score compatibility), but avoid iterrows().
    # itertuples is faster; we then pull the Series by index only for rows we will process.
    # This typically reduces overhead substantially at large N.
    tt_index = tt_gdf.index

    for i, row_key in enumerate(tqdm(tt_index, total=len(tt_gdf), desc="Matching TT→OSM")):
        tt_row = tt_gdf.loc[row_key]
        geom = tt_row.geometry
        if geom is None or geom.is_empty:
            continue

        # Query by expanded bounds (no shapely.box construction).
        minx, miny, maxx, maxy = geom.bounds
        query_bounds = (minx - buff, miny - buff, maxx + buff, maxy + buff)

        # sindex.intersection returns an iterator of positional indices for GeoPandas R-tree
        candidate_pos_idx = list(sindex.intersection(query_bounds))
        if not candidate_pos_idx:
            continue

        best_score = -1e9
        best_edge_id = None
        best_overlap = 0.0
        best_bearing_diff = 180.0
        best_name_score = 0.0

        # Iterate positional indices directly; avoid candidates = osm_edges.iloc[..].copy()
        for pos in candidate_pos_idx:
            osm_row = osm_edges.iloc[pos]
            score, overlap_ratio, bearing_diff, name_score = compute_match_score(
                tt_row, osm_row, cfg, dist_norm_den=dist_norm_den
            )
            if score > best_score:
                best_score = score
                best_edge_id = osm_row.get("edge_id", None)
                best_overlap = overlap_ratio
                best_bearing_diff = bearing_diff
                best_name_score = name_score

        if best_edge_id is None or best_score < cfg.score_threshold:
            continue

        match_rows.append(
            {
                "segmentId": tt_row["segmentId"],
                "newSegmentId_str": tt_row["newSegmentId_str"],
                "edge_id": best_edge_id,
                "score": best_score,
                "overlap_ratio": best_overlap,
                "bearing_diff": best_bearing_diff,
                "name_score": best_name_score,
            }
        )

    return pd.DataFrame(match_rows)


#old, tested, works
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
    sample_size_out_attr: str = "tomtom_sample_size",
) -> gpd.GeoDataFrame:
    """
    Aggregate TomTom speeds to OSM edges AND attach an aggregate sample size.
    Writes:
      - existing speed columns (avgSpeed_*, harmSpeed_*)
      - sample_size_out_attr (length-weighted effective sample size)
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
            rec[sample_size_out_attr] = float(np.sum(S * (W / W.sum())))
        else:
            rec[sample_size_out_attr] = np.nan

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


def _parse_lanes(val: Union[str, int, float, None],
                 how="min" #min or max
                 )-> Union[int, None]:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            return int(round(val))
        if type(val) is list:
            val = [float(x) for x in val]
            if how == "min":
                return int(round(min(val)))
            if how == "max":
                return int(round(max(val)))
        parts = [p.strip() for p in str(val).split(";") if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                import pdb; pdb.set_trace()
                continue
        if not nums:
            return None
        if how == "min":
            if len(nums) > 1:
                import pdb; pdb.set_trace()
            return int(round(min(nums)))
        if how == "max":
            return int(round(max(nums)))
        else:
            raise ValueError(f"Invalid how={how!r}")
    except Exception as e:
        logger.debug(f"Could not parse lanes='{val}': {e}")
        import pdb; pdb.set_trace()
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


def _classify_functional_class(hwy: Union[str, None],
                               dir_lanes: int,
                               params: Dict[str, Any],
                               ) -> str:
    h = (hwy or "").lower()
    fcs = params["functional_classes"]
    for name, cfg in fcs.items():
        tags = {t.lower() for t in cfg.get("osm_highway_tags", [])}
        min_lanes = cfg.get("min_lanes_per_direction", 0) #deprecated for now because it was causing ramps to be classified as local. Could re-include later.
        if h in tags:
            if dir_lanes >= min_lanes:
                return name
    return "local" if "local" in fcs else next(iter(fcs.keys()))


def _propagate_signal_tag(G: nx.MultiDiGraph, length: float = 300.0) -> nx.MultiDiGraph:
    """
    Mark edges that are upstream of traffic signals within a given network distance.

    For every node with highway=traffic_signal or highway=traffic_signals, walk
    *against* edge direction along predecessors, accumulating edge 'length'
    (meters). Any edge whose downstream endpoint can reach a signal node within
    `length` meters is tagged with edge['near_signal'] = True.

    The function:
      - respects directed edge orientation (u -> v is traffic flow)
      - avoids infinite loops by tracking the best (shortest) distance seen per node
      - does not clear existing 'near_signal' flags; it only sets them to True

    Args:
        G: Directed MultiDiGraph with edge attribute 'length' in meters.
        length: Maximum upstream network distance (meters) from a signal node.

    Returns:
        The same graph object, mutated in place.
    """

    def _node_is_signal(node_data: dict) -> bool:
        h = node_data.get("highway")
        if h is None:
            return False
        if isinstance(h, (list, tuple, set)):
            return any(x in {"traffic_signal", "traffic_signals"} for x in h)
        return h in {"traffic_signal", "traffic_signals"}

    # Collect all signal nodes
    signal_nodes = [n for n, data in G.nodes(data=True) if _node_is_signal(data)]
    n_signals = len(signal_nodes)
    logger.info(
        "_propagate_signal_tag: found %d signal nodes (length_cutoff=%.1f m)",
        n_signals,
        length,
    )

    if not signal_nodes:
        return G

    n_edges_marked_before = sum(
        1 for _, _, _, d in G.edges(keys=True, data=True) if d.get("near_signal", False)
    )
    total_new_marked = 0

    # Iterate over signals with progress bar
    for sig_node in tqdm(signal_nodes, desc="Propagating near_signal from traffic lights"):
        # best_dist[n] = shortest upstream distance (m) from sig_node to n
        best_dist = {sig_node: 0.0}
        q = deque([(sig_node, 0.0)])

        while q:
            cur_node, cur_dist = q.popleft()

            # Walk *upstream*: incoming edges to cur_node
            for u, v, k, edata in G.in_edges(cur_node, keys=True, data=True):
                edge_len = edata.get("length", 0.0)
                try:
                    edge_len = float(edge_len)
                except (TypeError, ValueError):
                    edge_len = 0.0

                new_dist = cur_dist + edge_len
                if new_dist > length:
                    continue

                # Mark this edge as being near a signal
                if not edata.get("near_signal", False):
                    edata["near_signal"] = True
                    total_new_marked += 1

                # If we've already reached u with a shorter distance, skip
                prev = best_dist.get(u)
                if prev is not None and prev <= new_dist:
                    continue

                best_dist[u] = new_dist
                q.append((u, new_dist))

    n_edges_marked_after = sum(
        1 for _, _, _, d in G.edges(keys=True, data=True) if d.get("near_signal", False)
    )
    logger.info(
        "_propagate_signal_tag: marked %d additional edges as near_signal "
        "(total now %d, previously %d)",
        total_new_marked,
        n_edges_marked_after,
        n_edges_marked_before,
    )

    return G


def add_osm_default_freeflowspeed_and_capacity(
    G: nx.MultiDiGraph,
    capacity_attr: str = "capacity_vph",
    capacity_source_attr: str = "capacity_source",
    speed_attr: str = "ff_speed_kph",
    ff_speed_source_attr: str = "ff_speed_source",
    osm_maxspeed_attr: str = "osm_maxspeed_kph",
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
    custom_capacities = params['custom_capacities']

    # Global fallback if a FC omits it
    signal_factor_default = float(params.get("signal_capacity_factor_default", 0.6))

    G = _propagate_signal_tag(G)


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

        # Per-lane capacity
        cap_per_lane = int(cap_defaults.get(hwy, default_cap_per_lane))
        if hwy not in cap_defaults:
            cap_source_bits.append("default_cap_per_lane")

        capacity = dir_lanes * cap_per_lane


        # Functional class + VDF params
        fc_name = _classify_functional_class(hwy, dir_lanes, params)

        fc_cfg = params["functional_classes"][fc_name]
        alpha = float(fc_cfg["alpha"])
        beta = float(fc_cfg["beta"])

        # Signal-end reduction
        signal_factor = float(fc_cfg.get("signal_capacity_factor", signal_factor_default))
        if (
                ("near_signal" in data.keys() and data['near_signal'] == True and signal_factor != 1.0)
                or
                (fc_cfg["always_treat_as_signalized"] == 1)
            ):
            capacity = int(round(capacity * signal_factor))
            cap_source_bits.append(f"signal_factor_{signal_factor:g}")

        # Free-flow speed
        speed_source_bits: list[str] = []
        maxspeed_kph = _parse_maxspeed_to_kph(data.get("maxspeed"))
        # Persist parsed maxspeed (even if not selected as free-flow speed)
        data[osm_maxspeed_attr] = float(maxspeed_kph) if maxspeed_kph is not None else None
        if maxspeed_kph is not None:
            freeflow_speed = float(maxspeed_kph)
            speed_source_bits.append("osm_maxspeed")
        else:
            freeflow_speed = float(default_speed_from_highway(hwy))
            speed_source_bits.append("default_speed_from_highway")

        #overwrite for custom capacities
        for custom_ref in custom_capacities.keys():
            if 'ref' in data.keys():
                if custom_ref in data['ref']:
                    capacity = custom_capacities[custom_ref]
                    speed_source_bits.append(f"custom_capacity_{custom_ref}")

        # Write attributes
        data['osm_lanes'] = data.get("lanes")
        data['lanes'] = dir_lanes
        data[capacity_attr] = capacity
        data[capacity_source_attr] = "|".join(cap_source_bits)
        data[speed_attr] = freeflow_speed
        data[ff_speed_source_attr] = "|".join(speed_source_bits)
        data[alpha_attr] = alpha
        data[beta_attr] = beta
        data[capacity_attr] = int(capacity)
        data[capacity_source_attr] = "+".join(cap_source_bits) if cap_source_bits else "unspecified"
        data[speed_attr] = freeflow_speed
        data['osm_freeflow_speed'] = freeflow_speed
        data[ff_speed_source_attr] = "+".join(speed_source_bits) if speed_source_bits else "unspecified"
        data[func_class_attr] = fc_name
        data[alpha_attr] = alpha
        data[beta_attr] = beta

    return G

def update_edge_speed_traveltime(
        u,
        v,
        k,
        data,
        length_attr: str = "length",
        freeflow_speed_attr: str = "ff_speed_kph", ##ff, obs, and forecast are INPUT
        ff_speed_source_attr: str = "ff_speed_source",
        obs_speed_attr: str = "obs_speed_kph",
        obs_speed_source_attr: str = "obs_speed_source",
        forecast_speed_attr: str = "forecast_speed_kph",
        forecast_speed_source_attr: str = "forecast_speed_source",
        peak_speed_attr: str = "peak_speed_kph", #peak speed and traversal time are OUTPUT
        peak_speed_source_attr: str = "peak_speed_source",
        travel_time_attr: str = "peak_traversal_time_sec",
        traversal_time_source_attr: str = "peak_traversal_time_source",
        params_path: str | Path = TRAFFIC_PARAMS_PATH,
):
    params = load_traffic_params(params_path)

    # Link penalties (robust defaults)
    link_penalties = params.get("link_penalties") or {}
    link_penalty_sec = float(link_penalties.get("link_traversal_time_penalty_sec", 0.0) or 0.0)
    link_penalty_max_len_m = float(link_penalties.get("link_traversal_time_penalty_max_length_m", 0.0) or 0.0)

    length_m = data.get(length_attr)

    # Speeds in priority order
    speed_kmh = None
    source = None
    if forecast_speed_attr in data:
        speed_kmh = data[forecast_speed_attr]
        source = f"forecast ({data[forecast_speed_source_attr]})"
    elif obs_speed_attr in data:
        speed_kmh = data.get(obs_speed_attr)
        source = f"observed ({data.get(obs_speed_source_attr)})"
    elif freeflow_speed_attr in data:
        speed_kmh = data.get(freeflow_speed_attr)
        source = f"freeflow ({data.get(ff_speed_source_attr)}"

    data[peak_speed_attr] = speed_kmh
    data[peak_speed_source_attr] = source

    if not isinstance(speed_kmh, (int, float)) or speed_kmh <= 0:
        logger.error(f"No valid speed found for edge ({u}, {v}, {k})")
        raise ValueError

    speed_ms = speed_kmh * (1000.0 / 3600.0)
    time_sec = length_m / speed_ms

    # Penalize short *_link edges
    if link_penalty_sec > 0 and link_penalty_max_len_m > 0:
        hwy = _normalize_highway(data.get("highway"))
        is_link = isinstance(hwy, str) and hwy.endswith("_link")
        if is_link and length_m <= link_penalty_max_len_m:
            time_sec += link_penalty_sec
            data[
                'link_penalty'] = link_penalty_sec  # TODO this should all be moved into a standard function so I don't forget it in the future
            source = f"{source}+link_penalty"

    data[travel_time_attr] = float(time_sec)
    data[traversal_time_source_attr] = source

def update_graph_speed_traveltime(
    G: nx.MultiDiGraph,
    length_attr: str = "length",
    freeflow_speed_attr: str = "ff_speed_kph",
    ff_speed_source_attr: str = "ff_speed_source",
    obs_speed_attr: str = "obs_speed_kph",
    obs_speed_source_attr: str = "obs_speed_source",
    peak_speed_attr: str = "peak_speed_kph",
    peak_speed_source_attr: str = "peak_speed_source",
    travel_time_attr: str = "peak_traversal_time_sec",
    traversal_time_source_attr: str = "peak_traversal_time_source",
    params_path: str | Path = TRAFFIC_PARAMS_PATH,
) -> None:
    """
    Calculate traversal time for each edge based on available speeds.
    Priority order: scenario > observed > freeflow speeds.
    Applies optional penalties to short *_link edges to discourage "off-and-on" cheating.
    Updates graph in-place.
    """
    logger.info("Calculating traversal times from available speeds")

    for u, v, k, data in G.edges(keys=True, data=True):
        update_edge_speed_traveltime(u,v,k,data)

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
    ff_speed_source_attr: str = "ff_speed_source",
    obs_speed_attr: str = "obs_speed_kph",
    obs_speed_source_attr: str = "obs_speed_source",
    tomtom_sample_attr: str = "tomtom_sample_size",
    tomtom_night_speed_attr: str = "tomtom_night_speed_kph",
    tomtom_sample_night_attr: str = "tomtom_sample_size_night",
    night_sample_min: int = 3,
) -> nx.MultiDiGraph:
    """
    Create a routable graph and attach TomTom speeds + sample size to edges.
    """
    G = add_osm_default_freeflowspeed_and_capacity(G)

    for _, row in edges_with_speeds.iterrows():
        u, v, k = row["u"], row["v"], row["key"]
        if not G.has_edge(u, v, k):
            continue

        edge_data = G[u][v][k]

        speed_kmh = row.get(speed_col)
        if isinstance(speed_kmh, (int, float)) and speed_kmh > 0:
            edge_data[obs_speed_attr] = float(speed_kmh)
            edge_data[obs_speed_source_attr] = "tomtom"

        # attach sample size (even if speed missing)
        ss = row.get(tomtom_sample_attr)
        if isinstance(ss, (int, float)) and np.isfinite(ss):
            edge_data[tomtom_sample_attr] = float(ss)

        # attach night speeds + sample size (if present)
        night_speed = row.get(tomtom_night_speed_attr)
        night_ss = row.get(tomtom_sample_night_attr)

        if isinstance(night_speed, (int, float)) and np.isfinite(night_speed) and night_speed > 0:
            edge_data[tomtom_night_speed_attr] = float(night_speed)

        if isinstance(night_ss, (int, float)) and np.isfinite(night_ss):
            edge_data[tomtom_sample_night_attr] = float(night_ss)

        # If night sample size clears the threshold, use as free-flow speed
        if (
            isinstance(night_speed, (int, float))
            and np.isfinite(night_speed)
            and night_speed > 0
            and isinstance(night_ss, (int, float))
            and np.isfinite(night_ss)
            and night_ss >= night_sample_min
        ):
            edge_data[ff_speed_attr] = float(night_speed)
            edge_data[ff_speed_source_attr] = "tomtom_night"


    update_graph_speed_traveltime(G)
    return G


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
def conflate_tomtom_to_osm(
        scenario_dir: str,
        tomtom_stats_json_path: str,
        tomtom_geojson_path: str,
        tomtom_night_stats_path: Optional[str] = None,
        tomtom_night_geom_path: Optional[str] = None,
        speed_col: str = "harmSpeed_t2_d1",
        cfg: Optional[MatchConfig] = None,
        debug_gpkg_prefix: Optional[str] = None,
        night_sample_min: int = 10,
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
    osm_filepath = Path(scenario_dir) / "input_data/osm_study_area_editable.osm"
    new_osm_filepath = Path(scenario_dir) / "input_data/traffic/filtered_osm.osm"
    save_graphml_to = Path(scenario_dir) / "input_data/traffic/routing_graph.graphml"
    save_nodes_to = Path(scenario_dir) / "input_data/traffic/routing_nodes.gpkg"
    save_edges_to = Path(scenario_dir) / "input_data/traffic/routing_edges.gpkg"

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
    G_drive = get_osm_graph(osm_filepath, new_osm_filepath)

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
    # 4b. Optional: Nighttime speeds (for free-flow speed inference)
    if tomtom_night_stats_path and tomtom_night_geom_path:
        logger.info(
            "Loading nighttime TomTom exports: stats_json='%s', geojson='%s'",
            tomtom_night_stats_path,
            tomtom_night_geom_path,
        )

        tt_night_gdf, night_date_ranges, night_time_sets = load_tomtom_segments(
            tomtom_night_stats_path,
            tomtom_night_geom_path,
            target_crs=cfg.target_crs,
        )

        logger.info(
            "Loaded %d nighttime TomTom segments; night_date_ranges=%d, night_time_sets=%d",
            len(tt_night_gdf),
            len(night_date_ranges),
            len(night_time_sets),
        )

        logger.info("Matching nighttime TomTom segments to OSM edges")
        matches_night_df = match_tomtom_to_osm(tt_night_gdf, osm_edges, cfg)

        logger.info(
            "Completed nighttime matching: %d matches, %d unique OSM edges with at least one night match",
            len(matches_night_df),
            matches_night_df["edge_id"].nunique() if "edge_id" in matches_night_df.columns else -1,
        )

        logger.info("Aggregating nighttime TomTom speeds to OSM edges")
        edges_with_night = aggregate_speeds_to_edges(
            tt_night_gdf,
            matches_night_df,
            osm_edges,
            sample_size_out_attr="tomtom_sample_size_night",
        )

        harm_cols = [c for c in edges_with_night.columns if c.startswith("harmSpeed_")]
        if len(harm_cols) != 1:
            raise ValueError(
                f"Expected exactly 1 nighttime harmonic speed column (harmSpeed_*), found {len(harm_cols)}: {harm_cols}"
            )
        night_harm_col = harm_cols[0]

        night_keep = edges_with_night[["edge_id", night_harm_col, "tomtom_sample_size_night"]].copy()
        night_keep = night_keep.rename(columns={night_harm_col: "tomtom_night_speed_kph"})

        # Merge night attributes onto day-aggregated edges
        edges_with_speeds = edges_with_speeds.merge(night_keep, on="edge_id", how="left")

        logger.info(
            "Nighttime merge complete: %d edges with night attributes (tomtom_night_speed_kph, tomtom_sample_size_night)",
            night_keep["tomtom_night_speed_kph"].notna().sum(),
        )


    # 5. Build routable graph and optionally save
    logger.info("Building routable graph from drivable OSM network and TomTom speeds")
    G_routable = build_routable_graph_from_edges(
        G_drive,
        edges_with_speeds,
        speed_col=speed_col,
        night_sample_min=night_sample_min,
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
    return G_routable, edges_with_speeds, date_ranges, time_sets, matches_df
