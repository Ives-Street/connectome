"""car_routing.py

CAR routing utilities for Connectome.

Provides the `od_matrix_times(...)` surface API, plus a checkpoint-aware routing API
with optional route (edge) dumping.

Key behaviors
-------------
- Graph source (CAR):
    G = ox.load_graphml(f"{scenario_dir}/routing/{routeenv}/traffic/routing_graph.graphml")
  The graph is kept as a NetworkX MultiDiGraph.

- Units:
    * travel times are stored as minutes in returned matrices (input edge weights are seconds)
    * route lengths are stored as meters (input edge lengths are read from the `length` attribute)

- Output caching:
    Matrices are saved to:
      - f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv"
      - f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_distances_matrix_CAR.csv"
      - f"{scenario_dir}/routing/{routeenv}/raw_ttms/checkpoint_crosses_matrix_CAR.csv"

- Optional route dumping (save_routes=True):
    Writes a normalized "long" Parquet table where each row represents one traversed edge of one OD pair.
    Ordering is not guaranteed (and not required). Columns:
      - origin_id (string): rep point ID for matrix row
      - dest_id (string): rep point ID for matrix column
      - u (int), v (int), key (int): MultiDiGraph edge triple identifiers

    File:
      - f"{scenario_dir}/routing/{routeenv}/raw_ttms/routes_edges_long_CAR.parquet"

    Notes:
      - This can become very large (O(#OD_pairs * avg_path_edges)). Logging includes row counts written.
      - Requires `pyarrow` or `fastparquet` for Parquet writing.

"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import osmnx as ox
import networkx as nx

from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    csr_matrix = None
    csgraph_dijkstra = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _load_cached_matrix(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _sanitize_edge_weights(G: nx.MultiDiGraph, weight_attr: str) -> None:
    """Ensure weight_attr exists and is numeric on all edges; missing/bad -> inf."""
    bad_weight_count = 0
    missing_weight_count = 0
    for _, _, _, data in G.edges(keys=True, data=True):
        if weight_attr not in data:
            missing_weight_count += 1
            data[weight_attr] = float("inf")
            continue
        val = data[weight_attr]
        try:
            if not isinstance(val, (int, float, np.number)):
                data[weight_attr] = float(val)
        except (TypeError, ValueError):
            bad_weight_count += 1
            data[weight_attr] = float("inf")

    if bad_weight_count or missing_weight_count:
        logger.warning(
            "Sanitized '%s' on %d edges (bad) and %d edges (missing)",
            weight_attr,
            bad_weight_count,
            missing_weight_count,
        )


def _snap_rep_points_to_nodes(G: nx.MultiDiGraph, rep_points) -> Tuple[List[str], List[int]]:
    """Return (rep_ids, snapped_node_ids) preserving rep_points order."""
    if "id" not in rep_points.columns:
        raise ValueError("rep_points must have an 'id' column.")

    rep_ids = rep_points["id"].astype(str).tolist()
    xs = rep_points.geometry.x.values
    ys = rep_points.geometry.y.values
    snapped = ox.distance.nearest_nodes(G, xs, ys)
    snapped_node_ids = list(map(int, snapped))

    if len(rep_ids) != len(snapped_node_ids):
        raise RuntimeError("Mismatch between rep_points IDs and snapped nodes.")
    return rep_ids, snapped_node_ids


def _choose_min_weight_edge_key(
    G: nx.MultiDiGraph, u: int, v: int, weight_attr: str
) -> Optional[int]:
    """Choose the edge key for (u,v) with minimum weight_attr."""
    try:
        edge_dict = G[u][v]
    except KeyError:
        return None
    best_key = None
    best_w = float("inf")
    for k, data in edge_dict.items():
        w = data.get(weight_attr, float("inf"))
        if w < best_w:
            best_w = w
            best_key = k
    return best_key


def _edge_attr_matches(
    edge_data: Dict[str, Any], attr: str, values: Set[Any]
) -> bool:
    """Match edge attribute to a set of values (exact match), including list-like tags."""
    if attr not in edge_data:
        return False
    val = edge_data.get(attr)
    if isinstance(val, (list, tuple, set)):
        return any(v in values for v in val)
    return val in values


@dataclass
class _CSRGraph:
    mat: "csr_matrix"
    node_to_idx: Dict[int, int]
    idx_to_node: np.ndarray
    best_key_by_uv: Dict[Tuple[int, int], int]
    min_len_m_by_uv: Dict[Tuple[int, int], float]
    checkpoint_uv: Optional[Set[Tuple[int, int]]]  # directed (u,v)


def _build_csr_from_multidigraph(
    G: nx.MultiDiGraph,
    weight_attr: str,
    checkpoint_edge_attr: Optional[str] = None,
    checkpoint_edge_values: Optional[Set[Any]] = None,
) -> _CSRGraph:
    """Collapse MultiDiGraph to a CSR adjacency with min weight per (u,v).

    Also captures:
      - best_key_by_uv[(u,v)] = key of the minimum-weight parallel edge
      - checkpoint_uv: set of (u,v) pairs where ANY parallel edge matches the checkpoint predicate
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy is not available; cannot build CSR graph.")

    nodes = np.array(list(G.nodes()), dtype=np.int64)
    node_to_idx = {int(n): i for i, n in enumerate(nodes)}

    # Collect minimum-weight edges per (u,v)
    min_w: Dict[Tuple[int, int], float] = {}
    best_key: Dict[Tuple[int, int], int] = {}
    min_len_m: Dict[Tuple[int, int], float] = {}
    checkpoint_uv: Optional[Set[Tuple[int, int]]] = set() if checkpoint_edge_attr else None

    # Iterate edges once; do not build dense structures
    for u, v, k, data in G.edges(keys=True, data=True):
        u = int(u)
        v = int(v)
        w = data.get(weight_attr, float("inf"))
        uv = (u, v)
        if w < min_w.get(uv, float("inf")):
            min_w[uv] = float(w)
            best_key[uv] = int(k)
            try:
                min_len_m[uv] = float(data.get("length", 0.0) or 0.0)
            except Exception:
                min_len_m[uv] = 0.0
        if checkpoint_uv is not None and checkpoint_edge_attr and checkpoint_edge_values is not None:
            if _edge_attr_matches(data, checkpoint_edge_attr, checkpoint_edge_values):
                checkpoint_uv.add(uv)

    # Build CSR
    rows: List[int] = []
    cols: List[int] = []
    data_w: List[float] = []
    for (u, v), w in min_w.items():
        # Ignore completely impassable links
        if not np.isfinite(w):
            continue
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])
        data_w.append(w)

    n = len(nodes)
    mat = csr_matrix((data_w, (rows, cols)), shape=(n, n))

    return _CSRGraph(
        mat=mat,
        node_to_idx=node_to_idx,
        idx_to_node=nodes,
        best_key_by_uv=best_key,
        min_len_m_by_uv=min_len_m,
        checkpoint_uv=checkpoint_uv,
    )


def _propagate_checkpoint_flags_from_predecessors(
    dist: np.ndarray,
    pred: np.ndarray,
    checkpoint_node_idx: Set[int],
    checkpoint_uv_idx: Optional[Set[Tuple[int, int]]],
    origin_idx: int,
    include_endpoints: bool = True,
) -> np.ndarray:
    """Given distances and predecessor indices for one origin, compute a boolean flag per node.

    The flag indicates whether the chosen shortest path from origin to that node passes through
    any checkpoint node and/or checkpoint edge.
    """
    n = dist.shape[0]
    passed = np.zeros(n, dtype=bool)

    # If endpoints count, origin can trigger.
    if include_endpoints and origin_idx in checkpoint_node_idx:
        passed[origin_idx] = True

    # Process reachable nodes in increasing distance order.
    order = np.argsort(dist)
    for v in order:
        if not np.isfinite(dist[v]):
            break  # unreachable tail
        if v == origin_idx:
            # already initialized
            continue
        p = int(pred[v])
        if p < 0:
            # unreachable or origin
            continue

        flag = passed[p]
        if include_endpoints and v in checkpoint_node_idx:
            flag = True
        if checkpoint_uv_idx is not None and (p, v) in checkpoint_uv_idx:
            flag = True
        passed[v] = flag

    return passed


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def od_matrix_times(
    scenario_dir: str,
    routeenv: str,
    rep_points: pd.DataFrame,
    mode: str = "CAR",
    departure_time=None,
    weight_attr: str = "peak_traversal_time_sec",
) -> pd.DataFrame:
    """All-to-all OD travel-time matrix (minutes). No path info.

    File I/O (matches existing conventions):
      - reads graph from:  {scenario_dir}/routing/{routeenv}/traffic/routing_graph.graphml
      - caches matrix to:  {scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv
    """
    logger.info('Starting OD matrix (times only)')
    raw_ttm_filename = f"{scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv"
    cached = _load_cached_matrix(raw_ttm_filename)
    if cached is not None:
        logger.info("Loaded cached raw TTM at '%s'", raw_ttm_filename)
        return cached

    G = ox.load_graphml(f"{scenario_dir}/routing/{routeenv}/traffic/routing_graph.graphml")
    _sanitize_edge_weights(G, weight_attr)
    rep_ids, snapped_nodes = _snap_rep_points_to_nodes(G, rep_points)

    n = len(rep_ids)
    mat = np.full((n, n), np.nan, dtype=float)

    logger.info("Computing OD matrix (%s) for %d points", mode, n)

    csr = _build_csr_from_multidigraph(G, weight_attr=weight_attr)
    idxs = [csr.node_to_idx[int(u)] for u in snapped_nodes]
    for i, orig_idx in tqdm(list(enumerate(idxs))):
        dist = csgraph_dijkstra(csr.mat, directed=True, indices=orig_idx, return_predecessors=False)
        # Fill only requested destinations
        mat[i, :] = dist[idxs]

    tt_df = pd.DataFrame(mat / 60.0, index=rep_ids, columns=rep_ids)  # seconds -> minutes
    _ensure_dirs(f"{scenario_dir}/routing/{routeenv}/raw_ttms/")
    tt_df.to_csv(raw_ttm_filename)
    logger.info("Saved raw TTM to '%s'", raw_ttm_filename)
    return tt_df




def od_matrix_times_with_checkpoints(
    scenario_dir: str,
    routeenv: str,
    rep_points: pd.DataFrame,
    mode: str = "CAR",
    departure_time=None,
    weight_attr: str = "peak_traversal_time_sec",
    checkpoint_node_ids: Optional[Iterable[int]] = None,
    checkpoint_edge_attr: Optional[str] = None,
    checkpoint_edge_values: Optional[Iterable[Any]] = None,
    save_routes: bool = True,
    include_endpoints: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute all-to-all matrices for CAR routing using SciPy CSR shortest paths, plus checkpoint detection.

    Surface API (required)
    ----------------------
    car_routing.od_matrix_times_with_checkpoints(
        scenario_dir,
        routeenv,
        mode="CAR",
        rep_points,  # GeoDataFrame/DataFrame with 'id' and 'geometry' (POINT), no snapped nodes required
        departure_time,
        checkpoint_node_ids,
        checkpoint_edge_attr,
        checkpoint_edge_values,
        save_routes=True,
    )

    Returns
    -------
    travel_time_df : pd.DataFrame
        Wide travel-time matrix in minutes. Index/columns are rep_points IDs (as strings).
    length_df : pd.DataFrame
        Wide distance matrix in meters (total route length along the chosen shortest path).
    checkpoint_df : pd.DataFrame
        Wide boolean matrix, True if the chosen shortest path crosses any checkpoint node/edge.

    Files written
    ------------
    Always writes (CSV):
      - {scenario_dir}/routing/{routeenv}/raw_ttms/raw_ttm_{mode}.csv
      - {scenario_dir}/routing/{routeenv}/raw_ttms/raw_distances_matrix_CAR.csv
      - {scenario_dir}/routing/{routeenv}/raw_ttms/checkpoint_crosses_matrix_CAR.csv

    If save_routes=True, also writes (Parquet, long/normalized):
      - {scenario_dir}/routing/{routeenv}/raw_ttms/routes_edges_long_CAR.parquet

      Each row represents one traversed edge of one OD route (order not guaranteed):
        origin_id (str), dest_id (str), u (int), v (int), key (int)

    Notes on checkpoint semantics
    -----------------------------
    - Node checkpoints: if any node on the route is in checkpoint_node_ids.
    - Edge checkpoints: if any traversed directed (u,v) has ANY parallel edge where
      edge_data[checkpoint_edge_attr] is in checkpoint_edge_values.
    - Endpoints count if include_endpoints=True (default).

    Backend requirements
    --------------------
    Requires SciPy. This function will raise if SciPy is not available.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy is required for od_matrix_times_with_checkpoints (no NetworkX fallback).")

    raw_dir = f"{scenario_dir}/routing/{routeenv}/raw_ttms"
    _ensure_dirs(raw_dir)

    ttm_path = f"{raw_dir}/raw_ttm_{mode}.csv"
    dist_path = f"{raw_dir}/raw_distances_matrix_CAR.csv"
    chk_path = f"{raw_dir}/checkpoint_crosses_matrix_CAR.csv"
    routes_path = f"{raw_dir}/routes_edges_long_CAR.parquet"

    logger.info(
        "Starting CAR OD matrix with checkpoints: scenario_dir=%s routeenv=%s mode=%s weight=%s save_routes=%s",
        scenario_dir, routeenv, mode, weight_attr, save_routes,
    )

    # Load graph (keep as MultiDiGraph)
    graph_path = f"{scenario_dir}/routing/{routeenv}/traffic/routing_graph.graphml"
    logger.info("Loading routing graph: %s", graph_path)
    G = ox.load_graphml(graph_path)

    _sanitize_edge_weights(G, weight_attr)

    # Snap rep points (rep_points has no snapped nodes)
    rep_ids, snapped_nodes = _snap_rep_points_to_nodes(G, rep_points)
    n = len(rep_ids)
    logger.info("Snapped %d rep points to graph nodes", n)

    # Checkpoint predicates
    node_checkpoints: Set[int] = set(int(x) for x in (checkpoint_node_ids or []))
    edge_values: Optional[Set[Any]] = None
    if checkpoint_edge_attr is not None:
        if checkpoint_edge_values is None:
            raise ValueError("checkpoint_edge_values must be provided when checkpoint_edge_attr is set")
        edge_values = set(checkpoint_edge_values)

    # Build CSR graph and checkpoint edge set
    csr = _build_csr_from_multidigraph(
        G,
        weight_attr=weight_attr,
        checkpoint_edge_attr=checkpoint_edge_attr,
        checkpoint_edge_values=edge_values,
    )

    # Translate snapped nodes into CSR indices (preserving order)
    try:
        idxs = [csr.node_to_idx[int(u)] for u in snapped_nodes]
    except KeyError as e:
        raise KeyError("One or more snapped nodes were not found in the CSR node index. Graph/node typing mismatch.") from e

    # Precompute checkpoint node indices
    checkpoint_node_idx = {csr.node_to_idx[u] for u in node_checkpoints if u in csr.node_to_idx}

    # Precompute checkpoint edge indices (directed pairs of node indices)
    checkpoint_uv_idx: Optional[Set[Tuple[int, int]]] = None
    if csr.checkpoint_uv is not None:
        checkpoint_uv_idx = {
            (csr.node_to_idx[u], csr.node_to_idx[v])
            for (u, v) in csr.checkpoint_uv
            if u in csr.node_to_idx and v in csr.node_to_idx
        }

    # Allocate outputs
    tt_sec = np.full((n, n), np.nan, dtype=float)
    dist_m = np.full((n, n), np.nan, dtype=float)
    chk = np.zeros((n, n), dtype=bool)

    # Optional routes output buffer
    # We stream in chunks to avoid huge in-memory tables.
    route_rows: List[Dict[str, Any]] = []
    chunk_flush = 500_000  # rows; tune if needed

    def _flush_routes(rows: List[Dict[str, Any]], first_write: bool) -> bool:
        """Append/write routes parquet in chunks. Returns False after first write."""
        if not rows:
            return first_write
        df = pd.DataFrame.from_records(rows)
        rows.clear()
        try:
            if first_write:
                df.to_parquet(routes_path, index=False)
                logger.info("Wrote routes parquet (initial): %s rows=%d", routes_path, len(df))
                return False
            else:
                # Append requires pyarrow dataset or fastparquet engine supporting append.
                # Use pyarrow if available.
                import pyarrow as pa  # type: ignore
                import pyarrow.parquet as pq  # type: ignore

                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_to_dataset(table, root_path=routes_path + ".dataset", partition_cols=[])
                logger.info("Appended routes parquet dataset: %s.dataset rows=%d", routes_path, len(df))
                return first_write
        except Exception as e:
            raise RuntimeError(
                "Failed to write/append Parquet routes table. Install 'pyarrow' (recommended). "
                "If you only have pandas without a parquet engine, this will fail."
            ) from e

    first_write = True

    # Per-origin routing
    for i, orig_idx in enumerate(tqdm(idxs, desc="Routing origins (SciPy)")):
        dist, pred = csgraph_dijkstra(
            csr.mat,
            directed=True,
            indices=orig_idx,
            return_predecessors=True,
        )
        # Fill travel times for requested destinations
        tt_sec[i, :] = dist[idxs]

        # Propagate checkpoint flags from predecessor tree
        passed = _propagate_checkpoint_flags_from_predecessors(
            dist=dist,
            pred=pred,
            checkpoint_node_idx=checkpoint_node_idx,
            checkpoint_uv_idx=checkpoint_uv_idx,
            origin_idx=orig_idx,
            include_endpoints=include_endpoints,
        )
        chk[i, :] = passed[idxs]

        # Compute total lengths to all nodes in the predecessor tree in one pass (dynamic programming)
        # length_to[v] = length_to[p] + length(p->v), where length(p->v) uses the min-weight chosen edge
        length_to = np.full(dist.shape[0], np.nan, dtype=float)
        length_to[orig_idx] = 0.0

        order = np.argsort(dist)
        for v in order:
            if not np.isfinite(dist[v]):
                break
            if v == orig_idx:
                continue
            p = int(pred[v])
            if p < 0 or not np.isfinite(length_to[p]):
                continue
            u_node = int(csr.idx_to_node[p])
            v_node = int(csr.idx_to_node[v])
            uv = (u_node, v_node)
            edge_len = csr.min_len_m_by_uv.get(uv, 0.0)
            length_to[v] = float(length_to[p]) + float(edge_len)

        dist_m[i, :] = length_to[idxs]

        # Optional: dump traversed edges for each OD pair (unordered edge triples)
        if save_routes:
            origin_id = rep_ids[i]
            # For each destination, walk predecessor chain and emit edges
            for j, dest_idx in enumerate(idxs):
                dest_id = rep_ids[j]
                if dest_idx == orig_idx:
                    continue
                if int(pred[dest_idx]) < 0:
                    continue  # unreachable
                cur = int(dest_idx)
                safety = 0
                while cur != int(orig_idx):
                    p = int(pred[cur])
                    if p < 0:
                        break
                    u_node = int(csr.idx_to_node[p])
                    v_node = int(csr.idx_to_node[cur])
                    key = int(csr.best_key_by_uv.get((u_node, v_node), -1))
                    if key >= 0:
                        route_rows.append(
                            {"origin_id": origin_id, "dest_id": dest_id, "u": u_node, "v": v_node, "key": key}
                        )
                    cur = p
                    safety += 1
                    if safety > dist.shape[0]:
                        break

                if len(route_rows) >= chunk_flush:
                    first_write = _flush_routes(route_rows, first_write)

    # Flush remaining routes
    if save_routes:
        first_write = _flush_routes(route_rows, first_write)
        if first_write is False and os.path.exists(routes_path):
            logger.info("Final routes parquet written at: %s", routes_path)
        elif os.path.exists(routes_path + ".dataset"):
            logger.info("Final routes parquet dataset written at: %s.dataset", routes_path)

    # Build DataFrames (minutes and meters)
    travel_time_df = pd.DataFrame(tt_sec / 60.0, index=rep_ids, columns=rep_ids)
    length_df = pd.DataFrame(dist_m, index=rep_ids, columns=rep_ids)
    checkpoint_df = pd.DataFrame(chk, index=rep_ids, columns=rep_ids)

    # Save matrices
    travel_time_df.to_csv(ttm_path)
    length_df.to_csv(dist_path)
    checkpoint_df.to_csv(chk_path)

    logger.info("Saved travel time matrix: %s", ttm_path)
    logger.info("Saved distance matrix: %s", dist_path)
    logger.info("Saved checkpoint matrix: %s", chk_path)

    return travel_time_df, length_df, checkpoint_df
