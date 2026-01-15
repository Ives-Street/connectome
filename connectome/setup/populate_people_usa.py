from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
import rasterstats
from census import Census
from pandas import DataFrame
from tqdm import tqdm
import numpy as np
from itertools import product
import os

#IF I want to add this (rather than hexagon things), i'll include the function in this file.
#from .OLDpedestriansfirst import make_patches

acs_variables = {
    'B01003_001E': 'total_pop',
    'B08201_001E': 'total_hh',
    'B08201_002E': 'hh_without_car',
    'B02001_002E': 'white',
    'B02001_003E': 'black',
    'B02001_005E': 'asian',
    'B03001_003E': 'hispanic/latino',
    'B19081_001E': 'lowest quintile hh income',
    'B19081_002E': 'second quintile hh income',
    'B19081_003E': 'middle quintile hh income',
    'B19081_004E': 'fourth quintile hh income',
    'B19081_005E': 'highest quintile hh income',
    #'B19013A_001E': 'median hh income - all',
    #'B19013A_001E': 'median hh income - white', #https://api.census.gov/data/2021/acs/acs5/groups/B19013A.html
    #'B19013B_001E': 'median hh income - black',
    #'B19013D_001E': 'median hh income - asian',
    #'B19013I_001E': 'median hh income - hisp/lat',
}


def get_acs_data_for_tracts(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("getting ACS data for tracts")
    c = Census("b55e824143791db1b0dc9dc85688cbefd0b3a04f") #TODO shouldn't be hardcoded, can be set up in python env
    acs_variable_names = list(acs_variables.keys())
    
    # Group tracts by state and county to batch API calls
    tracts_by_state_county = {}
    for idx in tracts.index:
        state = tracts.loc[idx, 'STATEFP']
        county = tracts.loc[idx, 'COUNTYFP']
        key = (state, county)
        if key not in tracts_by_state_county:
            tracts_by_state_county[key] = []
        tracts_by_state_county[key].append(idx)
    
    # Make batched API calls per state-county combination
    print(f"Fetching ACS data for {len(tracts_by_state_county)} state-county combinations")
    for (state, county), tract_indices in tqdm(tracts_by_state_county.items()):
        # Get all tracts in this county at once
        try:
            county_data = c.acs5.state_county_tract(
                acs_variable_names,
                state,
                county,
                Census.ALL  # Get all tracts in the county
            )
            
            # Create a lookup dictionary by tract code
            data_by_tract = {row['tract']: row for row in county_data}
            
            # Assign data to the tracts we're interested in
            for idx in tract_indices:
                tract_code = tracts.loc[idx, 'TRACTCE']
                if tract_code in data_by_tract:
                    vals = data_by_tract[tract_code]
                    for var_name in acs_variable_names:
                        tracts.loc[idx, var_name] = vals[var_name]
                else:
                    # Tract not found in county data, make individual call as fallback
                    vals = c.acs5.state_county_tract(
                        acs_variable_names,
                        state,
                        county,
                        tract_code
                    )[0]
                    for var_name in acs_variable_names:
                        tracts.loc[idx, var_name] = vals[var_name]
        except Exception as e:
            print(f"Error fetching data for state {state}, county {county}: {e}")
            print("Falling back to individual tract calls")
            # Fallback to individual calls for this county
            for idx in tract_indices:
                vals = c.acs5.state_county_tract(
                    acs_variable_names,
                    tracts.loc[idx, 'STATEFP'],
                    tracts.loc[idx, 'COUNTYFP'],
                    tracts.loc[idx, 'TRACTCE'],
                )[0]
                for var_name in acs_variable_names:
                    tracts.loc[idx, var_name] = vals[var_name]
    

    return tracts


def identify_bins(tracts: gpd.GeoDataFrame,
                  num_bins: int = 4,
                  return_histogram: bool = False,
                  ) -> tuple[Any, Any] | Any:
    all_income_numbers = []
    for i in range(1, 6):
        all_income_numbers += list(tracts[f'B19081_00{i}E'])
    all_income_numbers = [x for x in all_income_numbers if (x is not None and x > 0)]
    hist, bin_edges = pd.qcut(
        all_income_numbers,
        num_bins,
        retbins=True
    )
    if return_histogram:
        return bin_edges, hist
    else:
        return bin_edges


cyclist_distribution = {
    'bike_lts4': 0.04,
    'bike_lts2': 0.09,
    'bike_lts1': 0.56,
    'bike_lts0': 0.31,  #will not bike
} # higher number = greater willingness to cycle in more challenging environments


def create_userclasses(tracts: gpd.GeoDataFrame,
                              num_income_bins: int = 4,
                              save_to: str = "",
                              ) -> DataFrame:
    '''
    define user classes with factors relevant to transportation
    
    DOES NOT INCLUDE factors that are not currently relevant to how the model
    evaluates transportation (race, ethnicity) - these are currently 
    included in the userclass DF
    
    all analysis area geographies will have the same subdemographic groups
    (different numbers per analysis area, of course)
    this will make evaluation much more efficient
    
    output dataframe:
        index: user class category ID
        max_income: income ceiling
        max_cycle: cycle LTS ceiling 
    '''
    #TODO: add age, ?ability?

    if not list(acs_variables.keys())[0] in tracts.columns:
        get_acs_data_for_tracts(tracts)
    income_bin_edges = identify_bins(tracts, num_income_bins)

    # Build list of all user class combinations
    rows = []
    for max_income, max_bicycle, car_owner in product(
        income_bin_edges[1:],
        cyclist_distribution.keys(),
        ["car", "nocar"]
    ):
        user_class_id = f'{round(max_income)}_{max_bicycle[5:]}_{car_owner}'
        rows.append({
            'max_income': max_income,
            'max_bicycle': max_bicycle,
            'car_owner': car_owner,
            'user_class_id': user_class_id,
        })
    
    user_classes = pd.DataFrame(rows)
    user_classes.index = user_classes.user_class_id.values

    if save_to:
        user_classes.to_csv(save_to)

    return user_classes


def create_userclass_statistics(
    tracts: gpd.GeoDataFrame,
    user_classes: pd.DataFrame,
    min_pop_per_geom = 50, #TODO move to parameters file
    save_to: str = ""
) -> pd.DataFrame:
    """Calculate statistics for each user class.
    
    Args:
        tracts: GeoDataFrame of census tracts (should be in WGS84)
        user_classes: DataFrame of user class definitions
        save_to: Optional path to save results
        
    Returns:
        DataFrame with user class statistics
    """
    # Ensure WGS84
    if tracts.crs != "EPSG:4326":
        tracts = tracts.to_crs("EPSG:4326")
    
    # For area calculations, temporarily project to Mollweide equal area
    tracts_mw = tracts.to_crs('ESRI:54009')
    
    income_maxes = user_classes.max_income.unique()
    
    # Pre-build lookup dictionary for user_class_id to avoid repeated DataFrame filtering
    user_class_lookup = {}
    for _, row in user_classes.iterrows():
        key = (row['max_income'], row['max_bicycle'], row['car_owner'])
        user_class_lookup[key] = row['user_class_id']

    # Use list to accumulate rows instead of appending to DataFrame
    userclass_stats_rows = []

    for tract_idx in tqdm(list(tracts.index)):

        geom_id = tracts.loc[tract_idx, 'geom_id']

        tract_pop = tracts.loc[tract_idx, 'B01003_001E']
        if tract_pop <= min_pop_per_geom:
            continue

        pop_hisp_lat = tracts.loc[tract_idx, 'B03001_003E']
        not_hisp_lat = tract_pop - pop_hisp_lat
        pct_hisp_lat = pop_hisp_lat / tract_pop

        by_race = {
            'white': tracts.loc[tract_idx, 'B02001_002E'],
            'black': tracts.loc[tract_idx, 'B02001_003E'],
            'asian': tracts.loc[tract_idx, 'B02001_005E'],
        }
        by_race['other'] = tract_pop - sum([by_race['white'],
                                            by_race['black'],
                                            by_race['asian']])

        total_hh = tracts.loc[tract_idx, 'B08201_001E']
        hh_without_car = tracts.loc[tract_idx, 'B08201_002E']
        percent_with_car = 1 - (hh_without_car / total_hh)
        #note - we are assuming that car ownership is at the household level, and all
        # people within a household have a car for every trip if the household does. This is CONSERVATIVE.

        income_bin_pops = pd.Series(index=[income_maxes])
        income_bin_pops.loc[:] = 0
        for quintile_i in range(1, 6):
            quintile_income = tracts.loc[tract_idx, f'B19081_00{quintile_i}E']
            if quintile_income is None:
                print("NONEEEEE")
                continue
            if np.isnan(quintile_income): #the tract is so small that the quintile income is null
                # default to the middle income bin, rounding down
                # TODO: make this more robust, possibly in a broader synth-pop overhaul
                middle_index = len(income_maxes) // 2
                quintile_income = income_maxes[middle_index]
            try:
                bin_max = min([x for x in income_maxes
                           if x >= quintile_income])
                income_bin_pops[bin_max] += tract_pop / 5  # divide by 5 because census quintiles
            except ValueError:
                import pdb; pdb.set_trace()

        #create subgroups / user classes by nested choices
        #there must be a more elegant way to do this, but this works for now

        for income_bin in income_bin_pops.index: #1: Income
            pct_in_bin = income_bin_pops.loc[income_bin] / tract_pop
            for race in ['white', 'black', 'asian', 'other']: #2: Race
                #TODO: call census / synthpop better and disaggregate race from income
                race_pop = by_race[race] * pct_in_bin
                if race_pop <= 0:
                    continue
                for lts in [0, 1, 2, 4]: #3: Bicycle ability
                    cyclist_pct = cyclist_distribution[f'bike_lts{lts}']
                    race_cycle_pop = race_pop * cyclist_pct
                    for hisp_lat in [True, False]: #4: Hispanic/Latino
                        if hisp_lat:
                            race_cycle_hisplat_pop = race_cycle_pop * pct_hisp_lat
                        else:
                            race_cycle_hisplat_pop = race_cycle_pop * (1 - pct_hisp_lat)
                        for car_owner in ["car","nocar"]: #5: Car owner
                            if car_owner == "car":
                                race_cycle_hisplat_car_pop = race_cycle_hisplat_pop * percent_with_car
                            else:
                                race_cycle_hisplat_car_pop = race_cycle_hisplat_pop * (1 - percent_with_car)

                            # Use lookup dictionary instead of filtering DataFrame
                            lookup_key = (income_bin[0], f'bike_lts{lts}', car_owner)
                            user_class_id = user_class_lookup[lookup_key]

                            userclass_stats_rows.append({
                                'geom_id': geom_id,
                                'population': race_cycle_hisplat_car_pop,
                                'user_class_id': user_class_id,
                                'race': race,
                                'hispanic_or_latino': hisp_lat,
                                'car_owner': car_owner,
                            })
    
    # Create DataFrame once from all accumulated rows
    userclass_stats = pd.DataFrame(userclass_stats_rows)
    
    print(int(userclass_stats.population.sum()), tracts.B01003_001E.sum())
    tolerance=0.02
    totalpop_lower_bound = (1 - tolerance) * tracts.B01003_001E.sum()
    totalpop_upper_bound = (1 + tolerance) * tracts.B01003_001E.sum()
    print(totalpop_lower_bound, int(userclass_stats.population.sum()), totalpop_upper_bound)
    assert totalpop_lower_bound <= int(userclass_stats.population.sum()) <= totalpop_upper_bound

    if not save_to == False:
        userclass_stats.to_csv(save_to)
        #TODO: test below code

    return userclass_stats


def populate_people_usa(scenario_dir):

    input_dir = f"{scenario_dir}/input_data/"
    tracts = gpd.read_file(f"{input_dir}census/census_tracts.gpkg")
    tracts.index = tracts['geom_id'].values
    if not os.path.exists(f"{input_dir}/user_classes.csv"):
        num_income_bins = 4
        print("establishing user classes")
        user_classes = create_userclasses(
            tracts,
            num_income_bins,
            save_to = f"{input_dir}/user_classes.csv"
        )
    else:
        print("loading user classes from disk")
        user_classes = pd.read_csv(f"{input_dir}/user_classes.csv")
        user_classes.index = user_classes.user_class_id.values
        user_classes.fillna("",inplace=True)


    if not os.path.exists(f"{input_dir}/userclass_statistics.csv"):
        print("calculating user class statistics")
        userclass_statistics = create_userclass_statistics(
            tracts,
            user_classes,
            save_to=f"{input_dir}/census/tract_userclass_statistics.csv"
        )


