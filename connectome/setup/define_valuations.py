import numpy as np
#maybe eventually this will have some fancy means of dynamically generating functions for various pairs
#for now let's keep it simple

# Definition of valuation relationships between people and destinations (gravity model & exponent, dual/n-th access, etc) 
# by destination class and potentially by origin subgroup, also by time of day


#assume the average Place is visited 10x/day, the average job 1x/day, and the average person 0.05x/day
# what categories does Overture Places include? all retail, food, entertainment, etc?
VISIT_FREQ = {
    "overture_places":5,
    "lodes_jobs":1,
    "total_pop":0.05,
}

# read in overture places categories and assign overture json to appropriate category
# ^^ probably not necessary since json includes categories
# How many categories we thinking?

#TODO have this funciton take the dest_type
def value(subdemo, dest_type, minute_equivalents, time_of_day) -> float:
    base_val = np.exp(-0.05 * minute_equivalents)

    # weight by visit frequency/possibly weighted by size? - include this as an input?
    val = base_val * VISIT_FREQ[dest_type]

    # weight by TOD (will come later)


    return val