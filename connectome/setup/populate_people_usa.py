from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
import rasterstats
from census import Census
from pandas import DataFrame
from tqdm import tqdm

from .pedestriansfirst import make_patches

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
    c = Census("b55e824143791db1b0dc9dc85688cbefd0b3a04f") #TODO shouldn't be hardcoded, can be set up in python env 
    acs_variable_names = list(acs_variables.keys())
    for idx in tqdm(list(tracts.index)):
        vals = c.acs5.state_county_tract(acs_variable_names,
                                         tracts.loc[idx, 'STATEFP'],
                                         tracts.loc[idx, 'COUNTYFP'],
                                         tracts.loc[idx, 'TRACTCE'],
                                         )[0]
        for var_name in acs_variable_names:
            tracts.loc[idx, var_name] = vals[var_name]
    tracts['census_total_pop'] = tracts['B01003_001E']
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


def create_subdemo_categories(tracts: gpd.GeoDataFrame,
                              num_income_bins: int = 4,
                              save_to: str = "",
                              ) -> DataFrame:
    '''
    define subdemographic categoris with factors relevant to transportation
    
    DOES NOT INCLUDE factors that are not currently relevant to how the model
    evaluates transportation (race, ethnicity) - these are currently 
    included in the subdemo DF
    
    all analysis area geographies will have the same subdemographic groups
    (different numbers per analysis area, of course)
    this will make evaluation much more efficient
    
    output dataframe:
        index: subdemographic category ID
        max_income: income ceiling
        max_cycle: cycle LTS ceiling 
    '''
    #TODO: add age, ?ability?

    if not list(acs_variables.keys())[0] in tracts.columns:
        get_acs_data_for_tracts(tracts)
    income_bin_edges = identify_bins(tracts, num_income_bins)

    subdemo_categories = pd.DataFrame(columns=[
        'max_income',
        'max_bicycle'
    ])
    #find all permutations of relevant variables
    #TODO - replace with a more pythonic function, there must be one in a std lib
    for max_income in income_bin_edges[1:]:
        for max_bicycle in cyclist_distribution.keys():
            next_idx = subdemo_categories.shape[0] + 1
            row_data = {
                'max_income': max_income,
                'max_bicycle': max_bicycle,
            }
            subdemo_categories.loc[next_idx] = row_data

    if save_to:
        subdemo_categories.to_csv(save_to)

    return subdemo_categories


def create_subdemo_statistics(tracts: gpd.GeoDataFrame,
                              subdemo_categories: object,
                              tolerance: float = 0.02,
                              save_to: str = "",  #'existing_conditions/subdemos.csv'
                              ) -> DataFrame:
    '''
    creates a dataframe (not geo) with all the many subdemographic groups / user classes
    
    includes an ID refrence to the demographic subgroup that contains factors 
    relevant to mode choice (income, willingness to cycle)
    and includes all demographic factors NOT relevant to mode choice here
    
    output dataframe:
        geom_id: index of corresponding shape in tracts
        subdemo_category_id: index of relevant mode choice factors
        population: population
        TODO: households: households
        race: 'white'/'black'/'asian'/'other'
        hispanic_or_latino: True / False
    '''
    if not list(acs_variables.keys())[0] in tracts.columns:
        get_acs_data_for_tracts(tracts)
    income_maxes = subdemo_categories.max_income.unique()

    sub_demos = pd.DataFrame(columns=[
        'geom_id',
        'population',
        'subdemo_category_id',
        'race',
        'hispanic_or_latino',
    ])

    for tract_idx in tqdm(list(tracts.index)):

        geom_id = tracts.loc[tract_idx, 'geom_id']

        tract_pop = tracts.loc[tract_idx, 'B01003_001E']
        if tract_pop <= 0:
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

        hh_with_car = tracts.loc[tract_idx, 'B08201_001E']
        hh_without_car = tracts.loc[tract_idx, 'B08201_001E']
        percent_with_car = hh_with_car / (hh_without_car + hh_with_car)
        #note - we are assuming that car ownership is at the household level, and all
        # people within a household have a car for every trip if the household does. This is CONSERVATIVE.

        income_bin_pops = pd.Series(index=[income_maxes])
        income_bin_pops.loc[:] = 0
        for quintile_i in range(1, 6):
            quintile_income = tracts.loc[tract_idx, f'B19081_00{quintile_i}E']
            if quintile_income is None:
                print("NONEEEEE")
                continue
            bin_max = min([x for x in income_maxes
                           if x >= quintile_income])
            income_bin_pops[bin_max] += tract_pop / 5  # divide by 5 because census quintiles

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
                        for car_owner in [True, False]: #5: Car owner
                            if car_owner:
                                race_cycle_hisplat_car_pop = race_cycle_hisplat_pop * percent_with_car
                            else:
                                race_cycle_hisplat_car_pop = race_cycle_hisplat_pop * (1 - percent_with_car)

                            subdemo_category = subdemo_categories[
                                (subdemo_categories['max_income'] == income_bin[0]) &
                                (subdemo_categories['max_bicycle'] == f'bike_lts{lts}')]
                            subdemo_id = subdemo_category.index[0]

                            subgroup = pd.Series({
                                'geom_id': geom_id,
                                'population': race_cycle_hisplat_car_pop,
                                'subdemo_category_id': subdemo_id,
                                'race': race,
                                'hispanic_or_latino': hisp_lat,
                                'car_owner': car_owner,
                            })
                            sub_demos.loc[len(sub_demos)] = subgroup
    print(int(sub_demos.population.sum()), tracts.B01003_001E.sum())
    totalpop_lower_bound = (1 - tolerance) * tracts.B01003_001E.sum()
    totalpop_upper_bound = (1 + tolerance) * tracts.B01003_001E.sum()
    print(totalpop_lower_bound, int(sub_demos.population.sum()), totalpop_upper_bound)
    assert totalpop_lower_bound <= int(sub_demos.population.sum()) <= totalpop_upper_bound

    if not save_to == False:
        sub_demos.to_csv(save_to)
        #TODO: test below code

    return sub_demos


GHS_FILENAME = 'GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif'


#TODO divide into two functions? Create grid, and then populate based on census data?
def divide_tracts_to_grid(tracts,
                          sub_demos,
                          grid_size=1000,
                          min_pop=10,
                          save_patches='existing_conditions/population_geometry.gpkg',
                          save_subdemos='existing_conditions/subdemos.csv',
                          ):
    #divide the tracts into patches
    tracts_utm = ox.projection.project_gdf(tracts)
    tracts_mw = tracts.to_crs('ESRI:54009')
    patches = make_patches(tracts,
                           tracts_utm.crs,
                           patch_length=grid_size,
                           buffer=0)[0]
    patches_utm = ox.projection.project_gdf(patches)
    patches['area_sqm'] = patches_utm.area
    patches_mw = patches.to_crs('ESRI:54009')
    for idx in patches.index:
        x = rasterstats.zonal_stats(
            patches_mw.loc[idx, 'geometry'],
            GHS_FILENAME,
            stats=['mean'],
            all_touched=True
        )
        assert len(x) == 1
        pop_per_km2 = x[0]['mean']
        print(pop_per_km2)
        total_pop = pop_per_km2 * (patches.loc[idx, 'area_sqm'] / 1000000)
        patches.loc[idx, 'population'] = total_pop

        if total_pop > min_pop:
            #find 'parent' tract ID
            patch_reppoint = patches.loc[idx, 'geometry'].representative_point()
            try:
                parent_id = tracts[tracts.contains(patch_reppoint)].index[0]
            except:
                import pdb;
                pdb.set_trace()
            patches.loc[idx, 'parent_id'] = parent_id

    patches = patches[patches.population > min_pop]  #todo reindex?

    #find all the patches within each tract
    #and get the total "ghsl population" within each tract
    #because we're now using two different sources of population
    #and this will be the denominator when we divide the subgroups
    for tract_idx in tracts.index:
        select_within = patches.representative_point().within(
            tracts.loc[tract_idx, 'geometry'])
        #tracts.loc[tract_idx, 'children'] = select_within.index not working, trying something else.
        total_ghs_pop = patches[select_within].population.sum()
        tracts.loc[tract_idx, 'total_ghs_pop'] = total_ghs_pop

    #divide the subgroups
    new_subdemos = []
    for p_idx in tqdm(list(patches.index)):
        parent_id = patches.loc[p_idx, 'parent_id']
        parent_pop = tracts.loc[parent_id, 'total_ghs_pop']
        fraction_of_parent_pop = patches.loc[p_idx, 'population'] / parent_pop
        copied_subdemos = sub_demos[sub_demos.geom_id == parent_id].copy()
        copied_subdemos.geom_id = p_idx
        copied_subdemos.population = copied_subdemos.population * fraction_of_parent_pop
        new_subdemos.append(copied_subdemos)
    out_subdemos = pd.concat(new_subdemos)

    if not save_patches == False:
        patches.to_file(save_patches, driver='gpkg')

    if not save_subdemos == False:
        out_subdemos.to_csv(save_subdemos)

    return patches, out_subdemos

#test adding new line in Cursor