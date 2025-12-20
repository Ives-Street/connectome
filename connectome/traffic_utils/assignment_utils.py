
import osmnx as ox
import geopandas as gpd
import pandas as pd
import networkx as nx
import logging
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Any, Dict, Iterable, Optional, Tuple


logger = logging.getLogger(__name__)

#TODO - consider reversing this whole approach for performance:
# when I do the routing, tag each edge (maybe in a geoparquet) with all the sets of (uc_id, origin, dest) that use it
# then after evaluation, iterate through all edges and increment relative_volumes


def assign_relative_demand(scenario_dir,):
    # load graph
    # load userclasses_with_REs, population #s
    # for each userclass:
        # load value_sum_total_by_OD_by_userclass data
        # load RE-specific path data
        # for each OD
            # get population of userclass at O
            # identify links
            # increment link relative_volumes by value_sum for OD

    G_path = f"{scenario_dir}/input_data/traffic/pre_benchmark/routing_graph.graphml"
    logger.info("assign_relative_demand: loading graph from '%s'", G_path)
    G = ox.load_graphml(G_path)
    logger.info(
        "assign_relative_demand: loaded graph with %d nodes and %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    userclasses_path = f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"
    logger.info("assign_relative_demand: loading user classes from '%s'", userclasses_path)
    userclasses = pd.read_csv(userclasses_path)
    userclasses.index = userclasses.user_class_id.values
    userclasses.fillna("", inplace=True)
    logger.info(
        "assign_relative_demand: loaded %d user classes",
        len(userclasses.index),
    )

    userclass_stats_path = f"{scenario_dir}/input_data/userclass_statistics.csv"
    logger.info("assign_relative_demand: loading userclass statistics from '%s'", userclass_stats_path)
    userclass_stats = pd.read_csv(userclass_stats_path)
    userclass_stats.index = userclass_stats.geom_id.values
    userclass_stats.fillna("", inplace=True)

    for userclass_id in tqdm(list(userclasses.index)):
        tqdm.write(f"assign_relative_demand: analyzing userclass {userclass_id}")
        userclass_routeenv_CAR = userclasses.loc[userclass_id, "routeenv_CAR"]


        val_by_OD_path = f"{scenario_dir}/results/detailed_data/value_sum_total_by_OD_by_userclass/{userclass_id}.csv"
        val_by_OD = pd.read_csv(val_by_OD_path, index_col=0)
        val_by_OD.index = pd.to_numeric(val_by_OD.index)
        val_by_OD.columns = pd.to_numeric(val_by_OD.columns)

        mode_choices_path = f"{scenario_dir}/results/detailed_data/mode_selections_by_userclass/{userclass_id}.csv"
        mode_choices = pd.read_csv(mode_choices_path, index_col=0)
        mode_choices.index = pd.to_numeric(mode_choices.index)
        mode_choices.columns = pd.to_numeric(mode_choices.columns)

        val_by_OD_car = val_by_OD.copy()
        val_by_OD_car[mode_choices != "CAR"] = 0

        userclass_paths = pd.read_parquet(f"{scenario_dir}/routing/{userclass_routeenv_CAR}/paths/paths_CAR.parquet")

        for o_id in val_by_OD_car.index:
            for d_id in val_by_OD_car.columns:
                if not o_id == d_id :
                    hansens_for_ucOD = val_by_OD_car.loc[o_id, d_id]
                    if not np.isnan(hansens_for_ucOD) and hansens_for_ucOD > 0:
                        # Get path links for this O-D pair
                        path_rows = userclass_paths[
                            (userclass_paths.origin_id == str(o_id)) & #maybe if the types are integers the parquet will perform better?
                            (userclass_paths.dest_id == str(d_id))
                            ]

                        for _, row in path_rows.iterrows():
                            u = row["u"]
                            v = row["v"]
                            key = row["key"]

                            try:
                                edge_data = G.edges[u, v, key]
                            except KeyError:
                                logger.warning(
                                    "Edge (%s, %s, %s) not found in graph when assigning relative volumes "
                                    "for userclass %s, origin %s, dest %s",
                                    u, v, key, userclass_id, o_id, d_id,
                                )
                                continue

                            if "relative_demand" not in edge_data:
                                edge_data["relative_demand"] = 0.0
                            edge_data["relative_demand"] += hansens_for_ucOD

    # summarize how many edges received any demand
    edges_with_demand = sum(
        1 for _, _, _, data in G.edges(keys=True, data=True)
        if "relative_demand" in data
    )
    logger.info(
        "assign_relative_demand: finished assigning relative demand; "
        "%d edges have non-zero demand attribute",
        edges_with_demand,
    )

    out_dir = f"{scenario_dir}/input_data/traffic/post_benchmark/"
    logger.info("assign_relative_demand: saving updated graph to '%s'", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ox.save_graphml(G, f"{out_dir}/graph_with_relative_demands.graphml")
    logger.info("assign_relative_demand: graph saved successfully")





def back_calculate_freeflow_speeds_from_obs(
    scenario_dir,
    G: nx.MultiDiGraph | str = None,
    traffic_params_json_path: str = 'traffic_utils/traffic_analysis_parameters.json',
    observed_speed_attr: str = "obs_speed_kph",          # e.g., TomTom hour-window speed on the edge
    volume_attr: str = "modeled_vol_vph",                # your scaled volumes (vph, directional)
    capacity_attr: str = "capacity_vph",                 # directional capacity (vph)
    highway_attr: str = "highway",                       # OSM highway tag (string or list)
    ff_speed_attr: str = "backcalc_ff_speed_kph",                 # output attribute
    ff_speed_source_attr: str = "ff_speed_source",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Back-calculate per-edge free-flow speed u_ff from observed speed u_obs and a BPR-style
    congestion relationship using parameters in traffic_analysis_parameters.json.

    Uses the JSON's speed formula: u = u_ff / (1 + alpha * (vc_ratio ** beta)),
    so u_ff = u * (1 + alpha * (vc_ratio ** beta)). :contentReference[oaicite:0]{index=0}

    Applies vc_ratio clamps from JSON. :contentReference[oaicite:1]{index=1}
    Uses alpha/beta by functional class based on OSM highway tags. :contentReference[oaicite:2]{index=2}

    Returns a small stats dict; mutates G in-place.
    """
    logger.info(
        "Starting back_calculate_freeflow_speeds_from_obs for scenario_dir='%s', "
        "observed_speed_attr='%s', volume_attr='%s', capacity_attr='%s', overwrite=%s",
        scenario_dir,
        observed_speed_attr,
        volume_attr,
        capacity_attr,
        overwrite,
    )

    if not G:
        G_path = f"{scenario_dir}/input_data/traffic/post_benchmark/graph_with_demands_and_modeled_vols.graphml"
        logger.info(
            "No graph provided; loading graph from '%s' for free-flow back-calculation",
            G_path,
        )
        G = ox.load_graphml(G_path)
    else:
        logger.debug("Using provided graph object for free-flow back-calculation")

    G_path_out = f"{scenario_dir}/input_data/traffic/post_benchmark/routing_graph.graphml"
    logger.debug(
        "Free-flow back-calculation outputs will be saved to '%s' (graphml) "
        "and '%s' (edges gpkg)",
        G_path_out,
        f"{scenario_dir}/input_data/traffic/post_benchmark/routing_graph.gpkg",
    )

    logger.info("Loading traffic analysis parameters from '%s'", traffic_params_json_path)
    with open(traffic_params_json_path, "r") as f:
        params = json.load(f)

    fclasses = params["functional_classes"]
    clamps = params["clamps"]

    min_vc = float(clamps["vc_ratio"]["min_vc"])
    max_vc = float(clamps["vc_ratio"]["max_vc"])
    min_speed_kph = float(clamps["speed"]["min_speed_kph"])
    logger.debug(
        "Loaded clamps: vc_ratio in [%s, %s], min_speed_kph=%s",
        min_vc,
        max_vc,
        min_speed_kph,
    )

    # Build tag -> (alpha, beta, class_name)
    tag_to_params: Dict[str, Tuple[float, float, str]] = {}
    for cls_name, cfg in fclasses.items():
        a = float(cfg["alpha"])
        b = float(cfg["beta"])
        for tag in cfg.get("osm_highway_tags", []):
            tag_to_params[str(tag)] = (a, b, cls_name)

    logger.info(
        "Constructed functional class lookup for %d OSM highway tags across %d classes",
        len(tag_to_params),
        len(fclasses),
    )

    def _pick_highway_tag(val: Any) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return str(val[0]) if len(val) else None
        return str(val)

    stats = {
        "edges_total": 0,
        "edges_set": 0,
        "edges_skipped_missing_inputs": 0,
        "edges_skipped_no_class_params": 0,
        "edges_skipped_existing_ff": 0,
    }

    logger.info("Beginning per-edge free-flow back-calculation")
    for u, v, k, data in G.edges(keys=True, data=True):
        stats["edges_total"] += 1

        if (not overwrite) and (ff_speed_attr in data):
            stats["edges_skipped_existing_ff"] += 1
            continue

        u_obs = data.get(observed_speed_attr)
        vol = data.get(volume_attr)
        cap = data.get(capacity_attr)

        if u_obs is None or vol is None or cap is None:
            stats["edges_skipped_missing_inputs"] += 1
            continue

        try:
            u_obs = float(u_obs)
            vol = float(vol)
            cap = float(cap)
        except (TypeError, ValueError):
            stats["edges_skipped_missing_inputs"] += 1
            continue

        if cap <= 0 or vol < 0:
            stats["edges_skipped_missing_inputs"] += 1
            continue

        hwy = _pick_highway_tag(data.get(highway_attr))
        if hwy not in tag_to_params:
            stats["edges_skipped_no_class_params"] += 1
            continue

        alpha, beta, cls_name = tag_to_params[hwy]

        # Clamp observed speed (avoid pathological / zero speeds)
        if u_obs < min_speed_kph:
            u_obs_eff = min_speed_kph
        else:
            u_obs_eff = u_obs

        vc = vol / cap
        if vc < min_vc:
            vc_eff = min_vc
        elif vc > max_vc:
            vc_eff = max_vc
        else:
            vc_eff = vc

        multiplier = 1.0 + alpha * (vc_eff ** beta)
        u_ff = u_obs_eff * multiplier

        # Store
        data[ff_speed_attr] = u_ff
        data[ff_speed_source_attr] = "backcalc_bpr_from_obs"
        data["ff_backcalc_vc_ratio_raw"] = vc
        data["ff_backcalc_vc_ratio_used"] = vc_eff
        data["ff_backcalc_multiplier"] = multiplier
        data["ff_backcalc_class"] = cls_name

        stats["edges_set"] += 1

    logger.info(
        "Finished free-flow back-calculation: total_edges=%d, ff_set=%d, "
        "skipped_existing_ff=%d, skipped_missing_inputs=%d, skipped_no_class_params=%d",
        stats["edges_total"],
        stats["edges_set"],
        stats["edges_skipped_existing_ff"],
        stats["edges_skipped_missing_inputs"],
        stats["edges_skipped_no_class_params"],
    )

    ox.save_graphml(G, G_path_out)
    logger.info("Saved updated graph with back-calculated free-flow speeds to '%s'", G_path_out)

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    edges_out_path = f"{scenario_dir}/input_data/traffic/post_benchmark/routing_graph.gpkg"
    edges.to_file(edges_out_path, driver="GPKG")
    logger.info("Saved edges with benchmark attributes to '%s'", edges_out_path)

    return stats

