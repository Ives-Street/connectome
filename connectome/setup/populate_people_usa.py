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
from pathlib import Path
import json
import logging

#IF I want to add this (rather than hexagon things), i'll include the function in this file.
#from .OLDpedestriansfirst import make_patches

logger = logging.getLogger(__name__)


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


TRAFFIC_PARAMS_PATH = Path(__file__).parent.parent / "traffic_utils" / "traffic_analysis_parameters.json"

def load_traffic_params(path: str | Path = TRAFFIC_PARAMS_PATH):
    """Load traffic analysis parameters (functional classes + clamps)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


import re
from typing import Dict, List, Set, Tuple

_SAFE_FACILITY_RE = re.compile(r"^[A-Za-z0-9_]+$")


#### helper functions for toll exemption userclass logic

def _get_toll_exemptions_cfg(traffic_params: dict) -> dict:
    """Return toll_exemptions config block (or empty dict)."""
    return (traffic_params or {}).get("toll_exemptions", {}) or {}


def _validate_facility_keys_or_raise(facility_keys: List[str]) -> None:
    bad = [k for k in facility_keys if not _SAFE_FACILITY_RE.match(str(k))]
    if bad:
        raise ValueError(
            "Unsafe facility key(s) for exempt_{facility} columns / IDs. "
            "Expected keys to match ^[A-Za-z0-9_]+$: "
            f"{bad}"
        )


def _invert_exemptions_to_geom_sets(exemptions: Dict[str, List[Any]]) -> Dict[Any, Set[str]]:
    """
    exemptions: {facility_key: [geom_id, ...], ...}
    returns: {geom_id: {facility_key, ...}, ...}
    """
    geom_to_set: Dict[Any, Set[str]] = {}
    for facility, geom_ids in (exemptions or {}).items():
        if geom_ids is None:
            continue
        for geom_id in geom_ids:
            geom_id = str(geom_id)
            geom_to_set.setdefault(geom_id, set()).add(str(facility))
    return geom_to_set


def _exempt_set_suffix(exempt_set: Set[str]) -> str:
    """
    For IDs: 'A_B' (sorted) for multi-exempt; '' for none.
    """
    if not exempt_set:
        return ""
    return "_".join(sorted(exempt_set))

#### end helpers for toll exemption userclass logic

def get_acs_data_for_tracts(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("getting ACS data for tracts")
    try:
        census_api_key = open("census_api_key.txt").read().rstrip("\n")
    except FileNotFoundError:
        raise FileNotFoundError("census_api_key.txt not found. Get a key at https://api.census.gov/data/key_signup.html")
    c = Census(census_api_key)
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


def create_userclasses(
    tracts: gpd.GeoDataFrame,
    num_income_bins: int = 4,
    save_to: str = "",
) -> DataFrame:
    """
    Define user classes with factors relevant to transportation.

    Adds per-facility toll exemption columns `exempt_{facility}` and
    creates exempt userclasses for CAR owners only.

    ID rules:
      - Baseline (non-exempt) IDs unchanged: '{round(max_income)}_{lts}_{car_owner}'
      - Exempt CAR IDs: '{baseline_id}_{facility_or_set}exempt'
        where facility_or_set is 'FAC' or 'A_B' (sorted) for multi-exempt sets.
    """
    if not list(acs_variables.keys())[0] in tracts.columns:
        get_acs_data_for_tracts(tracts)

    income_bin_edges = identify_bins(tracts, num_income_bins)

    traffic_params = load_traffic_params()
    toll_cfg = _get_toll_exemptions_cfg(traffic_params)
    exemptions = (toll_cfg.get("exemptions", {}) or {})
    logger.info(f"toll_exemptions keys: {list((traffic_params or {}).keys())}")
    logger.info(f"exemptions type={type(exemptions)} keys={list(exemptions.keys())[:10]}")
    logger.info(f"num facilities={len(exemptions)}")

    facility_keys = list(exemptions.keys())
    _validate_facility_keys_or_raise(facility_keys)

    # Determine which exemption-sets actually occur (data-driven) so we only create needed classes.
    geom_to_exempt_set = _invert_exemptions_to_geom_sets(exemptions)
    tract_geom_ids = set(tracts["geom_id"].tolist())
    json_geom_ids = set()
    for fac, ids in exemptions.items():
        json_geom_ids.update(str(id) for id in (ids or []))

    logger.info(f"tract geom_id sample: {list(sorted(tract_geom_ids))[:5]}")
    logger.info(f"json geom_id sample: {list(sorted(json_geom_ids))[:5]}")
    logger.info(f"overlap count: {len(tract_geom_ids.intersection(json_geom_ids))}")

    # only sets that appear in tracts AND are non-empty
    unique_exempt_sets = {
        frozenset(geom_to_exempt_set.get(geom_id, set()))
        for geom_id in tracts["geom_id"].values
    }
    unique_exempt_sets.discard(frozenset())  # remove empty set (baseline)

    exempt_cols = [f"exempt_{k}" for k in facility_keys]

    rows = []

    # ---- Baseline userclasses (IDs unchanged; all exempt_* = 0) ----
    for max_income, max_bicycle, car_owner in product(
        income_bin_edges[1:],
        cyclist_distribution.keys(),
        ["car", "nocar"],
    ):
        baseline_id = f"{round(max_income)}_{max_bicycle[5:]}_{car_owner}"
        row = {
            "max_income": max_income,
            "max_bicycle": max_bicycle,
            "car_owner": car_owner,
            "user_class_id": baseline_id,
        }
        for c in exempt_cols:
            row[c] = 0
        rows.append(row)

        # ---- Exempt variants: CAR only; one row per exemption-set observed in data ----
        if car_owner == "car" and unique_exempt_sets:
            for ex_set_fs in sorted(unique_exempt_sets, key=lambda s: _exempt_set_suffix(set(s))):
                ex_set = set(ex_set_fs)
                suffix = _exempt_set_suffix(ex_set)  # e.g., 'I270' or 'A_B'
                exempt_id = f"{baseline_id}_{suffix}exempt"
                ex_row = {
                    "max_income": max_income,
                    "max_bicycle": max_bicycle,
                    "car_owner": car_owner,
                    "user_class_id": exempt_id,
                }
                for k in facility_keys:
                    ex_row[f"exempt_{k}"] = 1 if k in ex_set else 0
                rows.append(ex_row)

    user_classes = pd.DataFrame(rows)
    user_classes.index = user_classes.user_class_id.values

    if save_to:
        user_classes.to_csv(save_to, index=False)

    assert user_classes["user_class_id"].is_unique

    return user_classes

def create_userclasses_OLD(tracts: gpd.GeoDataFrame,
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

    traffic_params = load_traffic_params()# TODO finish adding toll exemption userclasses

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

    tracts["geom_id"] = tracts["geom_id"].astype(str)

    # For area calculations, temporarily project to Mollweide equal area
    tracts_mw = tracts.to_crs('ESRI:54009')

    # params / helpers for toll exempt userclass logic
    traffic_params = load_traffic_params()
    toll_cfg = _get_toll_exemptions_cfg(traffic_params)
    exemptions = (toll_cfg.get("exemptions", {}) or {})
    facility_keys = list(exemptions.keys())
    _validate_facility_keys_or_raise(facility_keys)
    geom_to_exempt_set = _invert_exemptions_to_geom_sets(exemptions)
    
    income_maxes = user_classes.max_income.unique()
    
    # Pre-build lookup dictionary for user_class_id to avoid repeated DataFrame filtering
    exempt_cols = [f"exempt_{k}" for k in facility_keys]
    user_class_lookup = {}
    for _, row in user_classes.iterrows():
        # Determine this row’s exemption-set suffix from its exempt_* columns.
        if row.get("car_owner") == "car" and exempt_cols:
            ex_set = {k for k in facility_keys if int(row.get(f"exempt_{k}", 0) or 0) == 1}
            ex_suffix = _exempt_set_suffix(ex_set)  # '' for baseline, 'A' or 'A_B' for exempt
        else:
            ex_suffix = ""  # nocar: no exemption variants

        key = (row["max_income"], row["max_bicycle"], row["car_owner"], ex_suffix)
        user_class_lookup[key] = row["user_class_id"]


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
                            tract_ex_set = geom_to_exempt_set.get(geom_id, set())
                            ex_suffix = _exempt_set_suffix(tract_ex_set) if (car_owner == "car") else ""
                            lookup_key = (income_bin[0], f"bike_lts{lts}", car_owner, ex_suffix)
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
    tracts["geom_id"] = tracts["geom_id"].astype(str)

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


