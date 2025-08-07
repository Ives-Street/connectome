from connectome.setup.define_analysis_areas import usa_tracts_from_address
from connectome.setup.populate_people_usa import (
    create_subdemo_categories, create_subdemo_statistics)
import os
import geopandas as gpd

if __name__=='__main__':
    run_name = "test_run_burlington"
    os.makedirs(run_name,exist_ok=True)
    os.chdir(run_name)
    os.makedirs("existing_conditions", exist_ok=True)
    os.makedirs("existing_conditions/input_data", exist_ok=True)

    #get tracts
    if not os.path.exists("existing_conditions/input_data/analysis_geometry.gpkg"):
        print("fetching census tracts")
        states = ["VT"]
        address = "26 University Pl, Burlington, VT 05405"
        buffer = 3000 #up to 10000
        burlington_tracts = usa_tracts_from_address(
            states,
            address,
            buffer=3000,
            save_to = "existing_conditions/input_data/analysis_geometry.gpkg"
        )
    else:
        print("loading tracts from disk")
        burlington_tracts = gpd.read_file("existing_conditions/input_data/analysis_geometry.gpkg")

    #get tract info
    num_income_bins = 4
    print("establishing subdemo categories")
    subdemo_categories = create_subdemo_categories(
        burlington_tracts,
        num_income_bins,
        save_to = "existing_conditions/input_data/subdemo_categories.csv"
    )

    print("calculating subdemo membership statistics")
    subdemo_statistics = create_subdemo_statistics(
        burlington_tracts,
        subdemo_categories,
        save_to="existing_conditions/input_data/subdemo_statistics.csv"
    )


def setup_pop_usa_from_address(states,
                               address,
                               buffer=10000,  #meters
                               grid=1000,  #False to return tracts, otherwise meters per grid cell side
                               min_pop_per_grid_cell=10,
                               num_income_bins=4,
                               folder_name='existing_conditions/',
                               analysis_area_filename='analysis_geometry.gpkg',
                               subdemo_categories_filename='subdemo_categories.csv',
                               subdemo_statistics_filename='subdemo_statistics.csv',
                               ):
    #TODO test

    subdemo_categories = create_subdemo_categories(
        tracts,
        num_income_bins,
        save_to=folder_name + subdemo_categories_filename,
    )

    subdemo_statistics = create_subdemo_statistics(
        tracts,
        subdemo_categories,
        save_to=(folder_name + subdemo_statistics_filename if grid == False else False)
    )

    if grid:
        divide_tracts_to_grid(
            tracts,
            subdemo_categories,
            grid_size=grid,
            min_pop=min_pop_per_grid_cell,
            save_patches=folder_name + analysis_area_filename,
            save_subdemos=folder_name + subdemo_categories_filename,
        )
