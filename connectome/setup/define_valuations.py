import numpy as np
#maybe eventually this will have some fancy means of dynamically generating functions for various pairs
#for now let's keep it simple

#assume the average Place is visited 10x/day, the average job 1x/day, and the average person 0.05x/day
VISIT_FREQ = {
    "overture_places":10,
    "lodes_jobs":1,
    "total_pop":0.05,
}


def value(subdemo, dest_type, minute_equivalents, time_of_day) -> float:
    base_val = np.exp(-0.05 * minute_equivalents)
    return val