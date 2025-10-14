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


def get_geometry_results_with_viz(scenario_dir,
                         userclass_gtms,
                         geometry_and_dests,
                         value_sum_total_by_OD_by_userclass,
                         population_by_geom_and_userclass,
                         mode_selections_by_userclass):
    result_categories = ['total_value', 'total_pop', 'per_capita']
    result_categories += [f"percent_from_{mode}" for mode in MODES]
    result_categories += [f"value_from_{mode}" for mode in MODES]

    results_by_geometry = gpd.GeoDataFrame(index=list(geometry_and_dests.geom_id),
                                           columns=result_categories)

    by_geom_and_userclass = {}
    for userclass in userclass_gtms.keys():
        by_geom_and_userclass[userclass] = gpd.GeoDataFrame(
            index=list(geometry_and_dests.geom_id),
            columns=result_categories)


    print("summarizing results by geometry")
    for geom_id in tqdm(list(results_by_geometry.index)):
        results_by_geometry.loc[geom_id, 'geometry'] = geometry_and_dests.loc[geom_id, 'geometry']
        total_value_tally = 0
        total_population = 0
        mode_value_tallies = {}
        for mode in MODES:
            mode_value_tallies[mode] = 0
        for userclass in userclass_gtms.keys():
            by_geom_and_userclass[userclass].loc[geom_id, 'geometry'] = geometry_and_dests.loc[geom_id, 'geometry']


            values_by_OD = value_sum_total_by_OD_by_userclass[userclass]
            mode_selections = mode_selections_by_userclass[userclass]
            population = population_by_geom_and_userclass.loc[geom_id, userclass]

            value_of_geom_and_userclass = values_by_OD.loc[geom_id, :].sum()
            total_value_tally += value_of_geom_and_userclass
            by_geom_and_userclass[userclass].loc[geom_id, 'total_value'] = value_of_geom_and_userclass
            total_population += population
            by_geom_and_userclass[userclass].loc[geom_id, 'population'] = population
            by_geom_and_userclass[userclass].loc[
                geom_id, 'per_capita'] = value_of_geom_and_userclass / population if population > 0 else np.nan

            for mode in MODES:
                mask = mode_selections != mode
                val_from_mode = values_by_OD.mask(mask).loc[geom_id, :].sum()
                mode_value_tallies[mode] += val_from_mode
                by_geom_and_userclass[userclass].loc[geom_id, f"value_from_{mode}"] = val_from_mode
                by_geom_and_userclass[userclass].loc[
                    geom_id, f"percent_from_{mode}"] = val_from_mode / value_of_geom_and_userclass if value_of_geom_and_userclass > 0 else np.nan
        results_by_geometry.loc[geom_id, 'total_value'] = total_value_tally
        results_by_geometry.loc[geom_id, 'total_pop'] = total_population
        results_by_geometry.loc[
            geom_id, 'per_capita'] = total_value_tally / total_population if total_population > 0 else np.nan

        for mode in MODES:
            results_by_geometry.loc[geom_id, f"percent_from_{mode}"] = mode_value_tallies[
                                                                           mode] / total_value_tally if total_value_tally > 0 else np.nan
            results_by_geometry.loc[geom_id, f"value_from_{mode}"] = mode_value_tallies[mode]
    
    # Ensure all numeric columns are proper numeric types before saving
    numeric_columns = result_categories
    for col in numeric_columns:
        results_by_geometry[col] = pd.to_numeric(results_by_geometry[col], errors='coerce')
    
    results_by_geometry.crs = geometry_and_dests.crs
    results_by_geometry.to_file(f"{scenario_dir}/results/geometry_results.gpkg", driver="GPKG")

    communication.make_radio_choropleth_map(
        scenario_dir=scenario_dir,
        in_data=results_by_geometry,
        outfile="results/geometry_results.html"
    )

    os.makedirs(f"{scenario_dir}/results/by_userclass/", exist_ok=True)
    for userclass in userclass_gtms.keys():
        by_geom_and_userclass[userclass].to_file(f"{scenario_dir}/results/by_userclass/results_{userclass}.gpkg", driver="GPKG")
        communication.make_radio_choropleth_map(
            scenario_dir=scenario_dir,
            in_data=by_geom_and_userclass[userclass],
            outfile=f"results/by_userclass/results_{userclass}.html"
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

