import pandas as pd
import geopandas as gpd
import os
import numpy as np

from tqdm import tqdm

import communication


MODES = [ #todo - make this universal for the whole codebase
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
    "RIDEHAIL",
]

#def evaluate_scenario(scenario_dir):
# scenario_dir = "burlington_test/existing_conditions"
#
# user_classes = pd.read_csv(f"{scenario_dir}/routing/user_classes_with_routeenvs.csv")
# user_classes.index = user_classes.user_class_id.values
# user_classes.fillna("",inplace=True)
# geometry_and_dests = gpd.read_file(f"{scenario_dir}/input_data/destination_statistics.gpkg")
# geometry_and_dests.index = geometry_and_dests.geom_id.values

# per user class:
# 1) turn time, $, and comfort matrices into generalized cost matrices for each mode
# 2) identify the mode choice for each O/D pair
    # Record which mode is chosen for each cell
    # Create a unified matrix representing the lowest cost for each O/D pair
# 3) apply decay function to each cell of the generalized gtm, to get per-destination value for each O/D pair
# 4) multiply the decayed gtm by the destination density to get a total utility for each O/D pair
# 5) sum the total utility of the user class
    # broken down by mode

def generalize_cost(user_class, gtm, cost_matrix = None, comfort_matrix = None):
    if cost_matrix is not None or comfort_matrix is not None:
        raise ValueError("Cost and comfort matrices are not yet supported.")
    else:
        return gtm



def load_gtm(filename):
    """" Loads a travel time matrix from a CSV file

    Currently provides index and column names as strings, for maximum flexibility.
    May be more efficient to load as integers?
    """
    df = pd.read_csv(filename, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df

def load_gtms_for_userclasses(scenario_dir, user_classes):
    gtms = {}
    for userclass in user_classes.index:
        gtms[userclass] = {}
        for mode in MODES:
            filename = f"{scenario_dir}/impedances/{userclass}/gtm_{mode}.csv"
            if os.path.exists(filename):
                print("loading", filename)
                gtm = load_gtm(filename)
                gtms[userclass][mode] = gtm
    return gtms

def get_population_by_geom_and_userclass(geometry_and_dests, userclass_statistics):
    population_by_geom_and_userclass = geometry_and_dests.drop(
        columns=[col for col in geometry_and_dests.columns if col not in
                 ['geom_id', 'geometry']])

    # Pivot userclass_statistics to get population by geom_id and user_class_id
    population_pivot = userclass_statistics.pivot_table(
        values='population',
        index='geom_id',
        columns='user_class_id',
        aggfunc='sum',
        fill_value=0
    )
    # Convert geom_id column and index to string type for consistent merging
    population_by_geom_and_userclass['geom_id'] = population_by_geom_and_userclass['geom_id'].astype(str)
    population_pivot.index = population_pivot.index.astype(str)

    population_by_geom_and_userclass = pd.merge(population_by_geom_and_userclass, population_pivot, left_on='geom_id',
                                                right_index=True,
                                                how='left')

    return population_by_geom_and_userclass

def choose_modes_for_userclasses(userclass_gtms):
    """
    input: a dictionary of impedance matrices by mode, indexed by mode name
    returns two dataframes: lowest_values and mode_selections
    Both dataframes have the same index and columns, which are the same as the impedance matrices in the input
    lowest_values is a dataframe with the lowest impedance for each O/D pair
    mode_selections is a dataframe with the mode that provides the lowest impedence for each O/D pair
    """
    lowest_traveltimes_by_userclass = {}
    mode_selections_by_userclass = {}
    for userclass in userclass_gtms.keys():
        impedance_matrices_by_mode = userclass_gtms[userclass]
        if not impedance_matrices_by_mode:
            import pdb; pdb.set_trace()
            raise ValueError("Input dictionary cannot be empty")

        # Initialize with first mode's values
        first_mode = list(impedance_matrices_by_mode.keys())[0]
        lowest_traveltimes = impedance_matrices_by_mode[first_mode].copy()
        mode_selections = pd.DataFrame(first_mode, index=lowest_traveltimes.index, columns=lowest_traveltimes.columns)

        # Compare with other modes
        for mode, matrix in impedance_matrices_by_mode.items():
            mask = matrix < lowest_traveltimes
            lowest_traveltimes = lowest_traveltimes.where(~mask, matrix)
            mode_selections = mode_selections.where(~mask, mode)
        lowest_traveltimes_by_userclass[userclass] = lowest_traveltimes
        mode_selections_by_userclass[userclass] = mode_selections
    return lowest_traveltimes_by_userclass, mode_selections_by_userclass

def value_per_destination_unit(minute_equivalents) -> float:
    val_per_dest = np.exp(-0.05 * minute_equivalents)

    # weight by TOD (will come later)
    return val_per_dest


def evaluate_for_userclasses(lowest_traveltimes_by_userclass, geometry_and_dests, population_by_geom_and_userclass):
    value_sum_per_person_by_userclass = {}
    values_per_dest_by_userclass = {}
    value_sum_total_by_OD_by_userclass = {}
    for userclass in lowest_traveltimes_by_userclass.keys():
        print("summarizing results for", userclass)
        lowest_traveltimes = lowest_traveltimes_by_userclass[userclass]
        values_per_dest = lowest_traveltimes.applymap(lambda x: value_per_destination_unit(x))
        values_per_dest_by_userclass[userclass] = values_per_dest

        # Multiply each row by the lodes_jobs count for each destination (column)
        #TODO correct this to use general destinations!
        value_sum_per_person = values_per_dest.mul(
            geometry_and_dests.loc[values_per_dest.columns, 'lodes_jobs'],
            axis=1)
        # Set diagonal values (where origin equals destination) to 0
        # TODO document this behavior better. Or assume it's WALK,
        #  and set a moderate value before getting weightings?
        value_sum_per_person.values[
            pd.Index(value_sum_per_person.index).get_indexer(value_sum_per_person.columns), range(
                len(value_sum_per_person.columns))] = 0
        value_sum_per_person_by_userclass[userclass] = value_sum_per_person
        try:
            pop_by_geom = population_by_geom_and_userclass[userclass]
        except:
            import pdb; pdb.set_trace()
        value_sum_per_person = value_sum_per_person_by_userclass[userclass]
        value_sum_total_by_OD = value_sum_per_person.mul(pop_by_geom, axis=0)
        value_sum_total_by_OD_by_userclass[userclass] = value_sum_total_by_OD
    return values_per_dest_by_userclass, value_sum_per_person_by_userclass, value_sum_total_by_OD_by_userclass


def get_userclass_results(scenario_dir,
                          userclass_gtms,
                          value_sum_total_by_OD_by_userclass,
                          population_by_geom_and_userclass,
                          mode_selections_by_userclass):
    result_categories = ['total_value', 'total_pop', 'per_capita']
    result_categories += [f"percent_from_{mode}" for mode in MODES]
    result_categories += [f"value_from_{mode}" for mode in MODES]

    results_by_userclass = pd.DataFrame(index=list(userclass_gtms.keys()),
                                        columns=result_categories)
    print("summarizing results by userclass")
    for userclass in tqdm(list(userclass_gtms.keys())):
        mode_selections = mode_selections_by_userclass[userclass]
        value_sum_total_by_OD = value_sum_total_by_OD_by_userclass[userclass]
        pop_by_geom = population_by_geom_and_userclass[userclass]

        total_userclass_value = value_sum_total_by_OD.sum().sum()
        total_userclass_pop = pop_by_geom.sum().sum()

        for mode in MODES:
            mask = mode_selections != mode
            value_from_mode = value_sum_total_by_OD.mask(mask)
            sum_value_from_mode = value_from_mode.sum().sum()
            results_by_userclass.loc[userclass, f"value_from_{mode}"] = sum_value_from_mode
            results_by_userclass.loc[userclass, f"percent_from_{mode}"] = sum_value_from_mode / total_userclass_value

        # TODO also save geospatial results out?

        results_by_userclass.loc[userclass, 'total_value'] = total_userclass_value
        results_by_userclass.loc[userclass, 'total_pop'] = total_userclass_pop
        results_by_userclass.loc[userclass, 'per_capita'] = total_userclass_value / total_userclass_pop

    results_by_userclass.to_csv(f"{scenario_dir}/results/userclass_results.csv")
    return results_by_userclass

def get_geometry_results_with_viz(
    scenario_dir,
    userclass_gtms,
    geometry_and_dests,                      # GeoDataFrame indexed by geom_id (has 'geometry')
    value_sum_total_by_OD_by_userclass,      # dict[userclass] -> DataFrame (index=geom_id, columns=dest ids)
    population_by_geom_and_userclass,        # DataFrame indexed by geom_id, columns=userclasses
    mode_selections_by_userclass             # dict[userclass] -> DataFrame same shape as above values
):
    # --- columns ---
    result_categories = ['total_value', 'population', 'per_capita']
    result_categories += [f"percent_from_{mode}" for mode in MODES]
    result_categories += [f"value_from_{mode}" for mode in MODES]

    # Ensure index alignment
    geom_index = geometry_and_dests.index

    # Container for per-userclass GeoDFs
    by_geom_and_userclass = {}

    # --- per-userclass vectorized summaries ---
    per_uc_frames = []
    for uc in userclass_gtms.keys():
        values = value_sum_total_by_OD_by_userclass[uc].loc[geom_index]           # [orig x dest]
        modes  = mode_selections_by_userclass[uc].loc[geom_index]                  # same shape
        pop_s  = population_by_geom_and_userclass[uc].reindex(geom_index)          # Series by geom

        total_value = values.sum(axis=1)                                           # per geom
        # per-mode value: mask once per mode and sum rows
        per_mode_vals = {
            mode: values.where(modes.eq(mode)).sum(axis=1) for mode in MODES
        }

        # Assemble userclass frame
        df_uc = pd.DataFrame(index=geom_index)
        df_uc['total_value'] = total_value.astype('float64')
        df_uc['population']  = pop_s.astype('float64')
        df_uc['per_capita']  = df_uc['total_value'] / df_uc['population'].replace(0, np.nan)

        for mode in MODES:
            v = per_mode_vals[mode].astype('float64')
            df_uc[f"value_from_{mode}"] = v
            df_uc[f"percent_from_{mode}"] = v / df_uc['total_value'].replace(0, np.nan)

        # attach geometry (keep CRS)
        gdf_uc = gpd.GeoDataFrame(df_uc, geometry=geometry_and_dests.geometry, crs=geometry_and_dests.crs)

        by_geom_and_userclass[uc] = gdf_uc
        try:
            per_uc_frames.append(gdf_uc[result_categories])  # numeric only for later aggregation
        except:
            import pdb; pdb.set_trace()

    # --- aggregate across userclasses (vectorized) ---
    # Sum numeric columns across all userclasses, then compute derived % and per_capita
    agg_numeric = pd.concat(per_uc_frames, axis=1)
    # Build a DataFrame with summed columns
    totals = pd.DataFrame(index=geom_index)
    totals['total_value'] = agg_numeric.filter(regex=r'(^|_)total_value$').sum(axis=1)
    totals['total_pop']   = agg_numeric.filter(regex=r'(^|_)population$').sum(axis=1)

    for mode in MODES:
        totals[f"value_from_{mode}"] = agg_numeric.filter(regex=fr'(^|_)value_from_{mode}$').sum(axis=1)

    totals['per_capita'] = totals['total_value'] / totals['total_pop'].replace(0, np.nan)
    for mode in MODES:
        totals[f"percent_from_{mode}"] = totals[f"value_from_{mode}"] / totals['total_value'].replace(0, np.nan)

    # Final GeoDataFrame with geometry & CRS
    totals.rename(columns={'total_pop': 'population'}, inplace=True)
    results_by_geometry = gpd.GeoDataFrame(
        totals[result_categories], geometry=geometry_and_dests.geometry, crs=geometry_and_dests.crs
    )

    # --- write outputs & maps ---
    out_dir = f"{scenario_dir}/results"
    os.makedirs(out_dir, exist_ok=True)
    results_by_geometry.to_file(f"{out_dir}/geometry_results.gpkg", driver="GPKG")

    communication.make_radio_choropleth_map(
        scenario_dir=scenario_dir,
        in_data=results_by_geometry,
        outfile="results/geometry_results.html"
    )

    by_uc_dir = f"{out_dir}/by_userclass"
    os.makedirs(by_uc_dir, exist_ok=True)
    for uc, gdf_uc in by_geom_and_userclass.items():
        gdf_uc.to_file(f"{by_uc_dir}/results_{uc}.gpkg", driver="GPKG")
        communication.make_radio_choropleth_map(
            scenario_dir=scenario_dir,
            in_data=gdf_uc,
            outfile=f"results/by_userclass/results_{uc}.html"
        )

    return results_by_geometry


def evaluate_scenario(scenario_dir, user_classes, userclass_statistics, geometry_and_dests):
    os.makedirs(f"{scenario_dir}/results", exist_ok=True)

    #load the gtms organized by userclass
    userclass_gtms = load_gtms_for_userclasses(scenario_dir, user_classes)

    #get population by geom_id and userclass
    population_by_geom_and_userclass = get_population_by_geom_and_userclass(geometry_and_dests, userclass_statistics)

    #get traveltimes and mode selections for each userclass
    (
        lowest_traveltimes_by_userclass,
        mode_selections_by_userclass
    ) = choose_modes_for_userclasses(userclass_gtms)

    # get values for each userclass
    (
        values_per_dest_by_userclass,
        value_sum_per_person_by_userclass,
        value_sum_total_by_OD_by_userclass
    ) = evaluate_for_userclasses(lowest_traveltimes_by_userclass,
                                 geometry_and_dests,
                                 population_by_geom_and_userclass)

    # we've done all the calculations! Now sum by userclass and geometry:
    results_by_userclass = get_userclass_results(scenario_dir,
                                                  userclass_gtms,
                                                  value_sum_total_by_OD_by_userclass,
                                                  population_by_geom_and_userclass,
                                                  mode_selections_by_userclass)
    results_by_geometry = get_geometry_results_with_viz(scenario_dir,
                                                 userclass_gtms,
                                                 geometry_and_dests,
                                                 value_sum_total_by_OD_by_userclass,
                                                 population_by_geom_and_userclass,
                                                 mode_selections_by_userclass)

    print("results by geometry sum", results_by_geometry.total_value.sum(), "results by userclass sum", results_by_userclass.total_value.sum())

