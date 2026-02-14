
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
from scipy import stats
import os.path

from traffic_utils.speed_utils import update_edge_speed_traveltime

logger = logging.getLogger(__name__)

#TODO - consider reversing this whole approach for performance:
# when I do the routing, tag each edge (maybe in a geoparquet) with all the sets of (uc_id, origin, dest) that use it
# then after evaluation, iterate through all edges and increment relative_volumes

def _align_matrices(matrices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Force all matrices to share identical row/column indices *and order*.
    Uses the intersection of indices/columns across all matrices.
    Raises if alignment would drop anything or if order cannot be made identical.
    """
    logger.info("Aligning OD matrices to a common index/column order")

    # Compute common index/columns
    common_index = None
    common_columns = None

    for name, df in matrices.items():
        if common_index is None:
            common_index = df.index
            common_columns = df.columns
        else:
            common_index = common_index.intersection(df.index)
            common_columns = common_columns.intersection(df.columns)

    if common_index.empty or common_columns.empty:
        raise ValueError("No common OD indices/columns across matrices")

    aligned = {}
    for name, df in matrices.items():
        idx_changed = not df.index.equals(common_index)
        col_changed = not df.columns.equals(common_columns)

        # Reindex *and* reorder
        new_df = df.reindex(index=common_index, columns=common_columns)

        if idx_changed or col_changed:
            logger.info(
                f"Reordered matrix '{name}' to match common OD index/column order"
            )

        aligned[name] = new_df

    # Final sanity check: identical order everywhere
    ref = next(iter(aligned.values()))
    for name, df in aligned.items():
        if not (df.index.equals(ref.index) and df.columns.equals(ref.columns)):
            raise RuntimeError(f"Post-alignment mismatch in matrix '{name}'")

    return aligned


#new chatgpt version
def assign_relative_demand(
    scenario_dir,
):
    """
    Faster induced-demand assignment:
    - Builds an OD long table once per userclass (filtered to relevant ODs)
    - Merges OD weights onto the (OD, edge) parquet once
    - Aggregates to (u,v,key) with a single groupby
    - Applies updates to G in one pass over touched edges
    """
    # --- Load graph ---
    G_path = f"{scenario_dir}/input_data/traffic/routing_graph.graphml"
    logger.info("relative_to_absolute_induced_demand: loading graph from '%s'", G_path)
    G = ox.load_graphml(G_path)

    # --- Load userclasses + stats (kept for parity with your current pipeline) ---
    userclasses_path = f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"
    logger.info("relative_to_absolute_induced_demand: loading user classes from '%s'", userclasses_path)
    userclasses = pd.read_csv(userclasses_path)
    userclasses.index = userclasses.user_class_id.values
    userclasses.fillna("", inplace=True)
    logger.info("relative_to_absolute_induced_demand: loaded %d user classes", len(userclasses.index))

    userclass_stats_path = f"{scenario_dir}/input_data/userclass_statistics.csv"
    logger.info("relative_to_absolute_induced_demand: loading userclass statistics from '%s'", userclass_stats_path)
    userclass_stats = pd.read_csv(userclass_stats_path)
    userclass_stats.index = userclass_stats.geom_id.values
    userclass_stats.fillna("", inplace=True)

    modified_edges: set[tuple[Any, Any, Any]] = set()
    total_relative_demand = 0.0
    meters_per_mile = 1609.344

    # --- Core loop: per userclass ---
    for userclass_id in tqdm(list(userclasses.index), desc="relative_to_absolute_induced_demand (userclasses)"):
        userclass_routeenv_CAR = userclasses.loc[userclass_id, "routeenv_CAR"]
        if not userclass_routeenv_CAR:
            continue

        # OD values (hansens / "value units" of the connection)
        val_by_OD_path = (
            f"{scenario_dir}/results/detailed_data/value_sum_total_by_OD_by_userclass/{userclass_id}.csv"
        )
        val_by_OD = pd.read_csv(val_by_OD_path, index_col=0)
        val_by_OD.index = pd.to_numeric(val_by_OD.index, errors="coerce")
        val_by_OD.columns = pd.to_numeric(val_by_OD.columns, errors="coerce")

        # Mode choices
        mode_choices_path = (
            f"{scenario_dir}/results/detailed_data/mode_selections_by_userclass/{userclass_id}.csv"
        )
        mode_choices = pd.read_csv(mode_choices_path, index_col=0)
        mode_choices.index = pd.to_numeric(mode_choices.index, errors="coerce")
        mode_choices.columns = pd.to_numeric(mode_choices.columns, errors="coerce")

        # Path lengths
        lengths_mtx_path = f"{scenario_dir}/routing/{userclass_routeenv_CAR}/raw_ttms/raw_distances_matrix_CAR.csv"
        lengths_mtx = pd.read_csv(lengths_mtx_path, index_col=0)
        lengths_mtx.index = pd.to_numeric(lengths_mtx.index, errors="coerce")
        lengths_mtx.columns = pd.to_numeric(lengths_mtx.columns, errors="coerce")

        # Optional checkpoint filter
        checkpoint_mtx_path = f"{scenario_dir}/routing/{userclass_routeenv_CAR}/raw_ttms/checkpoint_crosses_matrix_CAR.csv"
        if os.path.exists(checkpoint_mtx_path):
            checkpoint_mtx = pd.read_csv(checkpoint_mtx_path, index_col=0)
            checkpoint_mtx.index = pd.to_numeric(checkpoint_mtx.index, errors="coerce")
            checkpoint_mtx.columns = pd.to_numeric(checkpoint_mtx.columns, errors="coerce")
        else:
            checkpoint_mtx = None

        matrices = {
            "val_by_OD": val_by_OD,
            "mode_choices": mode_choices,
            "lengths_mtx": lengths_mtx,
        }
        if checkpoint_mtx is not None:
            matrices["checkpoint_mtx"] = checkpoint_mtx

        aligned = _align_matrices(matrices)

        val_by_OD = aligned["val_by_OD"]
        lengths_mtx = aligned["lengths_mtx"]
        mode_choices = aligned["mode_choices"]
        checkpoint_mtx = aligned.get("checkpoint_mtx")

        # Keep CAR only
        val_by_OD_car = val_by_OD.copy()
        val_by_OD_car[mode_choices != "CAR"] = 0.0

        # Routes (one row per OD-edge)
        userclass_paths_path = (
            f"{scenario_dir}/routing/{userclass_routeenv_CAR}/raw_ttms/routes_edges_long_CAR.parquet"
        )
        userclass_paths = pd.read_parquet(userclass_paths_path)

        # Normalize OD id types for merge (parquet often stores as strings)
        # Keep u/v/key untouched (they must match the graph’s node ids/edge keys exactly).
        if "origin_id" not in userclass_paths.columns or "dest_id" not in userclass_paths.columns:
            raise ValueError(
                "routes_edges_long_CAR.parquet must include 'origin_id' and 'dest_id' columns."
            )
        userclass_paths = userclass_paths.copy()
        userclass_paths["origin_id"] = pd.to_numeric(userclass_paths["origin_id"], errors="coerce")
        userclass_paths["dest_id"] = pd.to_numeric(userclass_paths["dest_id"], errors="coerce")

        # --- Build OD long table (filtered) ---
        od_val = val_by_OD_car.stack(future_stack=True)
        od_len = lengths_mtx.stack(future_stack=True)


        od = (
            pd.DataFrame({"hansens": od_val})
            .join(od_len.rename("path_len_m"), how="inner")
            .reset_index()
            .rename(columns={"level_0": "origin_id", "level_1": "dest_id"})
        )

        od = od[
            (od["origin_id"].notna())
            & (od["dest_id"].notna())
            & (od["origin_id"] != od["dest_id"])
            & (od["hansens"].notna())
            & (od["hansens"] > 0)
            & (od["path_len_m"].notna())
            & (od["path_len_m"] > 0)
            ]

        if checkpoint_mtx is not None:
            od_cp = checkpoint_mtx.stack(future_stack=True).rename("crosses")

            # CRITICAL: make index names match before joining
            od_cp.index = od_cp.index.set_names(["origin_id", "dest_id"])

            od = (
                od.set_index(["origin_id", "dest_id"])
                .join(od_cp, how="left")
                .reset_index()
            )
            od = od[od["crosses"].fillna(False).astype(bool)]

        od["hansens_per_m"] = od["hansens"] / od["path_len_m"]

        # --- Merge OD weights onto (OD, edge) rows, aggregate to edges ---
        merged = userclass_paths.merge(
            od[["origin_id", "dest_id", "hansens_per_m"]],
            on=["origin_id", "dest_id"],
            how="inner",
            copy=False,
        )
        if merged.empty:
            continue

        # Sum hansens_per_m across all ODs that traverse each edge
        if not {"u", "v", "key"}.issubset(merged.columns):
            raise ValueError("routes_edges_long_CAR.parquet must include 'u', 'v', and 'key' columns.")

        edge_hpm = (
            merged.groupby(["u", "v", "key"], sort=False)["hansens_per_m"]
            .sum()
        )

        # --- Apply aggregated updates to graph (one pass over touched edges) ---
        for (u, v, k), hansens_per_m_sum in edge_hpm.items():
            try:
                edge_data = G.edges[u, v, k]
            except KeyError:
                logger.warning(
                    "Edge (%s, %s, %s) not found in graph when assigning relative demand "
                    "for userclass %s",
                    u, v, k, userclass_id,
                )
                continue

            if "relative_demand_per_meter" not in edge_data:
                edge_data["relative_demand_per_meter"] = 0.0
            edge_data["relative_demand_per_meter"] += float(hansens_per_m_sum)

            length_m = float(edge_data.get("length", 0.0))
            hansens_to_add = float(hansens_per_m_sum) * length_m

            if "relative_demand" not in edge_data:
                edge_data["relative_demand"] = 0.0
            edge_data["relative_demand"] += hansens_to_add

            modified_edges.add((u, v, k))
            total_relative_demand += hansens_to_add

    # --- Summary ---
    logger.info(
        "assign_relative_demand: finished assigning relative demand; %d edges have non-zero demand attribute",
        len(modified_edges),
    )
    out_dir = f"{scenario_dir}/results/traffic/"
    logger.info("assign_relative_demand: saving updated graph to '%s'", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ox.save_graphml(G, f"{out_dir}/routing_graph_with_relative_demand.graphml")
    logger.info("assign_relative_demand: graph saved successfully in '%s'", out_dir)

def relative_to_absolute_induced_demand(
        scenario_dir,
        G,
        induced_demand_vmt,
        save=False,
):
    # --- Induced demand scaling + speed forecast update ---
    total_relative_demand = 0
    for (u, v, k, edge_data) in tqdm(list(G.edges(keys=True, data=True)), desc="get total_relative_demand"):
        total_relative_demand += float(edge_data.get("relative_demand", 0.0))

    if bool(induced_demand_vmt) and total_relative_demand > 0:
        vmt_per_hansen = float(induced_demand_vmt) / float(total_relative_demand)
        meters_per_mile = 1609.344

        for (u, v, k, edge_data) in tqdm(list(G.edges(keys=True, data=True)), desc="apply induced demand"):
            length_m = float(edge_data.get("length", 0.0))
            if length_m <= 0:
                continue

            volume = float(edge_data.get("modeled_vol_vph", 0.0))
            relative_demand = float(edge_data.get("relative_demand", 0.0))

            additional_vmt = relative_demand * vmt_per_hansen
            additional_v_meters_t = additional_vmt * meters_per_mile
            additional_volume = additional_v_meters_t / length_m
            if np.isnan(additional_volume):
                additional_volume = 0.0

            edge_data["induced_volume"] = additional_volume
            post_induction_vol_vph = volume + additional_volume
            edge_data["post_induction_vol_vph"] = post_induction_vol_vph

    # --- Save outputs ---
    if save:
        out_dir = f"{scenario_dir}/input_data/traffic/"
        logger.info("relative_to_absolute_induced_demand: saving updated graph to '%s'", out_dir)
        os.makedirs(out_dir, exist_ok=True)
        ox.save_graphml(G, f"{out_dir}/routing_graph.graphml")
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        edges.to_file(f"{out_dir}/routing_edges.gpkg", driver="GPKG")
        logger.info("assign_induced_demand: graph saved successfully in '%s'", out_dir)

        # Generate traffic statistics
        save_traffic_stats(G, scenario_dir)

    return G


def recalculate_speeds(G):
    logger.info("recalculate_speeds: recalculating speeds on graph")
    for u, v, k, edge_data in tqdm(list(G.edges(keys=True, data=True))):

        length_m = float(edge_data.get("length", 0.0))
        if length_m <= 0:
            continue
        
        ff_speed = float(edge_data.get("ff_speed_kph", np.nan))
        previous_peak_speed = float(edge_data.get("peak_speed_kph", np.nan))
        previous_traversal_time = float(edge_data.get("peak_traversal_time_sec", 0.0))
        capacity = float(edge_data.get("capacity_vph", np.nan))
        alpha = float(edge_data.get("vdf_alpha", np.nan))
        beta = float(edge_data.get("vdf_beta", np.nan))
        post_induction_vol_vph = float(edge_data.get("post_induction_vol_vph", np.nan))

        if np.isnan(post_induction_vol_vph):
            if bool(edge_data.get('recalculate_speed')):
                post_induction_vol_vph = float(edge_data.get("modeled_vol_vph"))
            else:
                continue

        if (
                np.isnan(ff_speed)
                or np.isnan(previous_peak_speed)
                or np.isnan(capacity)
                or capacity <= 0
                or np.isnan(alpha)
                or np.isnan(beta)
        ):
            raise ValueError
    
        predicted_speed = ff_speed / (1.0 + (alpha * ((post_induction_vol_vph / capacity) ** beta)))
    
        calibration_factor = float(edge_data.get("calibration_factor", 1.0))
        calibrated_predicted_speed = predicted_speed * calibration_factor

        edge_data['previous_peak_speed_kph'] = previous_peak_speed
        edge_data['previous_traversal_time'] = previous_traversal_time

        edge_data["forecast_speed_kph"] = calibrated_predicted_speed
        edge_data["forecast_speed_source"] = "modeled with induced demand"
    
        speed_change_percent = (calibrated_predicted_speed - previous_peak_speed) / previous_peak_speed * 100.0
        edge_data["forecast_speed_change_percent"] = speed_change_percent
    
        update_edge_speed_traveltime(u, v, k, edge_data)

        new_traversal_time = float(edge_data.get("peak_traversal_time_sec"))
        scenario_delay_sec = new_traversal_time - previous_traversal_time
        scenario_delay_veh_sec = scenario_delay_sec * post_induction_vol_vph
        scenario_delay_veh_min = scenario_delay_veh_sec / 60
        edge_data["scenario_delay_veh_sec"] = scenario_delay_veh_sec
        edge_data["scenario_delay_veh_min"] = scenario_delay_veh_min

        edge_data["v_c_ratio"] = post_induction_vol_vph / capacity

    return G

def save_traffic_stats(G, scenario_dir, out_subdir="input_data/traffic"):
    """
    Save routing_graph.gpkg and routing_graph.graphml,
    Save key road network statistics in both human and machine readable formats.
    """
    logger.info("saving traffic graphml, gpkg, and statistics")
    out_dir = f"{scenario_dir}/{out_subdir}/"

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    edges['scenario_delay_veh_min_per_mile'] = edges['scenario_delay_veh_min'] / (edges['length'] / 1609)
    edges['scenario_delay_veh_hr_per_mile'] = edges['scenario_delay_veh_min_per_mile'] / 60

    edges['induced_vmt'] = edges['induced_volume'] * (edges['length'] / 1609)

    # Calculate statistics for key metrics
    stats = {}
    metrics = [
        'length', 'modeled_vol_vph', 'capacity_vph', 'ff_speed_kph',
        'peak_speed_kph', 'forecast_speed_kph', 'v_c_ratio',
        'induced_volume', 'forecast_speed_change_percent',
        'scenario_delay_veh_sec', 'scenario_delay_veh_min',
        'induced_vmt'
    ]


    edges['sc_delay_veh_min_per_mile'] = edges['scenario_delay_veh_min'] / (edges['length'] / 1609)

    for metric in metrics:
        if metric in edges.columns:
            data = edges[metric].dropna().astype(float)
            if not data.empty:
                stats[metric] = {
                    'count': len(data),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'sum': float(data.sum()),
                }

    os.makedirs(f"{scenario_dir}/{out_subdir}/", exist_ok=True)

    # Save machine-readable JSON
    with open(f"{out_dir}/traffic_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Save human-readable text summary
    with open(f"{out_dir}/traffic_statistics.txt", 'w') as f:
        f.write("Traffic Network Statistics Summary\n")
        f.write("================================\n\n")
        for metric, values in stats.items():
            f.write(f"{metric}:\n")
            f.write(f"  Count: {values['count']:,.0f}\n")
            f.write(f"  Mean: {values['mean']:,.2f}\n")
            f.write(f"  Median: {values['median']:,.2f}\n")
            f.write(f"  Std Dev: {values['std']:,.2f}\n")
            f.write(f"  Min: {values['min']:,.2f}\n")
            f.write(f"  Max: {values['max']:,.2f}\n")
            f.write(f"  Sum: {values['sum']:,.2f}\n\n")

    logger.info("saving modified road graph")
    ox.save_graphml(G, f"{scenario_dir}/{out_subdir}/routing_graph.graphml")
    edges.to_file(f"{scenario_dir}/{out_subdir}/routing_edges.gpkg", driver="GPKG")


#this was the old function
def assign_induced_demand_handwritten(scenario_dir,
                          induced_demand_vmt: float|bool = None,
                          ):
    # load graph
    # load userclasses_with_REs, population #s
    # for each userclass:
        # load value_sum_total_by_OD_by_userclass data
        # load RE-specific path data
        # for each OD
            # get population of userclass at O
            # identify links
            # increment link relative_volumes by value_sum for OD

    G_path = f"{scenario_dir}/input_data/traffic/routing_graph.graphml"
    logger.info("assign_induced_demand: loading graph from '%s'", G_path)
    G = ox.load_graphml(G_path)
    logger.info(
        "assign_induced_demand: loaded graph with %d nodes and %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    userclasses_path = f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"
    logger.info("assign_induced_demand: loading user classes from '%s'", userclasses_path)
    userclasses = pd.read_csv(userclasses_path)
    userclasses.index = userclasses.user_class_id.values
    userclasses.fillna("", inplace=True)
    logger.info(
        "assign_induced_demand: loaded %d user classes",
        len(userclasses.index),
    )

    userclass_stats_path = f"{scenario_dir}/input_data/userclass_statistics.csv"
    logger.info("assign_induced_demand: loading userclass statistics from '%s'", userclass_stats_path)
    userclass_stats = pd.read_csv(userclass_stats_path)
    userclass_stats.index = userclass_stats.geom_id.values
    userclass_stats.fillna("", inplace=True)

    modified_edges = set()
    total_relative_demand = 0

    for userclass_id in tqdm(list(userclasses.index)):
        #tqdm.write(f"assign_induced_demand: analyzing userclass {userclass_id}")
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

        checkpoint_mtx_path = f"{scenario_dir}/routing/{userclass_routeenv_CAR}/raw_ttms/checkpoint_crosses_matrix_CAR.csv"
        checkpoint_mtx = pd.read_csv(checkpoint_mtx_path, index_col=0)
        checkpoint_mtx.index = pd.to_numeric(checkpoint_mtx.index)
        checkpoint_mtx.columns = pd.to_numeric(checkpoint_mtx.columns)

        lengths_mtx_path = f"{scenario_dir}/routing/{userclass_routeenv_CAR}/raw_ttms/raw_distances_matrix_CAR.csv"
        lengths_mtx = pd.read_csv(lengths_mtx_path, index_col=0)
        lengths_mtx.index = pd.to_numeric(lengths_mtx.index)
        lengths_mtx.columns = pd.to_numeric(lengths_mtx.columns)

        userclass_paths_path = f"{scenario_dir}/routing/{userclass_routeenv_CAR}/raw_ttms/routes_edges_long_CAR.parquet"
        userclass_paths = pd.read_parquet(userclass_paths_path)


        for o_id in val_by_OD_car.index:
            for d_id in val_by_OD_car.columns:
                if not o_id == d_id :
                    hansens_for_ucOD = val_by_OD_car.loc[o_id, d_id] #the 'value units' of the connection
                    length_m = lengths_mtx.loc[o_id, d_id]
                    hansens_per_m = hansens_for_ucOD / length_m

                    if bool(induced_demand_vmt): #if we're doing induced demand, we only care about routes that involve a given highway
                        crosses_checkpoints = bool(checkpoint_mtx.loc[o_id, d_id])
                    else: #otherwise, we care about everything
                        crosses_checkpoints = True
                    if (not np.isnan(hansens_for_ucOD)) and (hansens_for_ucOD > 0) and (crosses_checkpoints):
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

                            if "relative_demand_per_meter" not in edge_data:
                                edge_data["relative_demand_per_meter"] = 0.0
                            edge_data["relative_demand_per_meter"] += hansens_per_m

                            length = float(edge_data["length"])
                            hansens_to_add_to_edge = hansens_per_m * length
                            if "relative_demand" not in edge_data:
                                edge_data["relative_demand"] = 0.0
                            edge_data["relative_demand"] += hansens_to_add_to_edge

                            modified_edges.add((u, v, key))
                            total_relative_demand += hansens_to_add_to_edge

    # summarize how many edges received any demand
    edges_with_demand = len(modified_edges)
    logger.info(
        "assign_induced_demand: finished assigning relative demand; "
        "%d edges have non-zero demand attribute",
        edges_with_demand,
    )

    meters_per_mile = 1609.344

    if bool(induced_demand_vmt):
        vmt_per_hansen = induced_demand_vmt/total_relative_demand
        for edge_uvk in modified_edges:
            u, v, k = edge_uvk
            edge_data = G.edges[u, v, k]
            length = float(edge_data["length"])
            volume = float(edge_data.get("modeled_vol_vph", 0.0))
            relative_demand = float(edge_data["relative_demand"])
            additional_vmt = relative_demand * vmt_per_hansen
            additional_v_meters_t = additional_vmt * meters_per_mile
            additional_volume = additional_v_meters_t / length
            edge_data["induced_volume"] = additional_volume
            post_induction_vol_vph = volume + additional_volume
            edge_data["post_induction_vol_vph"] = post_induction_vol_vph

            alpha = float(edge_data['vdf_alpha'])
            beta = float(edge_data['vdf_beta'])
            ff_speed = float(edge_data['ff_speed_kph'])
            previous_peak_speed = float(edge_data['peak_speed_kph'])
            capacity = float(edge_data['capacity_vph'])
            predicted_speed = ff_speed / (1 + (alpha * ((post_induction_vol_vph / capacity) ** beta)))

            calibration_factor = float(edge_data.get('calibration_factor', 1))
            calibrated_predicted_speed = predicted_speed * calibration_factor
            edge_data['forecast_speed_kph'] = calibrated_predicted_speed
            edge_data['forecast_speed_source'] = "modeled with induced demand"

            speed_change_percent = (calibrated_predicted_speed - previous_peak_speed) / previous_peak_speed * 100
            edge_data['forecast_speed_change_percent'] = speed_change_percent

            #TODO add hours-lost metrics

            update_edge_speed_traveltime(u, v, k, edge_data)

    out_dir = f"{scenario_dir}/results/traffic/"
    logger.info("assign_induced_demand: saving updated graph to '%s'", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ox.save_graphml(G, f"{out_dir}/modified_routing_graph.graphml")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges.to_file(f"{out_dir}/modified_routing_edges.gpkg", driver="GPKG")
    logger.info("assign_induced_demand: graph saved successfully in '%s'", out_dir)


