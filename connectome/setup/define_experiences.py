import os
import shutil
import pandas as pd
import osmium
import json

MODES = [
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
    #"RIDESHARE",
    #"SHARED_BICYCLE",
]

class SimplestLTSAdder(osmium.SimpleHandler):
    def __init__(self, writer, max_lts):
        osmium.SimpleHandler.__init__(self)
        self.writer = writer
        self.n_modified_ways = 0
        self.max_lts = max_lts  # if 0, set all

    def node(self, n):
        self.writer.add_node(n)

    def way(self, way):
        if 'highway' in way.tags:
            if self.max_lts == 0:
                newtags = dict(way.tags)
                newtags['lts'] = '4'
                self.writer.add_way(way.replace(tags=newtags))
            else:
                if way.tags.get('highway') == 'cycleway':  # LTS 1
                    newtags = dict(way.tags)
                    lts_to_assign = 1
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway') == 'track':  # LTS 1
                    newtags = dict(way.tags)
                    lts_to_assign = 1
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway:left') == 'track':  # LTS 1
                    newtags = dict(way.tags)
                    lts_to_assign = 1
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway:right') == 'track':  # LTS 1
                    newtags = dict(way.tags)
                    lts_to_assign = 1
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway') == 'lane' and way.tags.get('highway') in ['tertiary',
                                                                                        'residential']:  # LTS 2
                    newtags = dict(way.tags)
                    lts_to_assign = 2
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway:left') == 'lane' and way.tags.get('highway') in ['tertiary',
                                                                                             'residential']:  # LTS 2
                    newtags = dict(way.tags)
                    lts_to_assign = 2
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway:right') == 'lane' and way.tags.get('highway') in ['tertiary',
                                                                                              'residential']:  # LTS 2
                    newtags = dict(way.tags)
                    lts_to_assign = 2
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway') == 'lane' and way.tags.get('highway') in ['primary',
                                                                                        'secondary']:  # LTS 3
                    newtags = dict(way.tags)
                    lts_to_assign = 3
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway:left') == 'lane' and way.tags.get('highway') in ['primary',
                                                                                             'secondary']:  # LTS 3
                    newtags = dict(way.tags)
                    lts_to_assign = 3
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                elif way.tags.get('cycleway:right') == 'lane' and way.tags.get('highway') in ['primary',
                                                                                              'secondary']:  # LTS 3
                    newtags = dict(way.tags)
                    lts_to_assign = 3
                    writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
                    newtags['lts'] = str(writeval)
                    self.writer.add_way(way.replace(tags=newtags))
                    if writeval != 4: self.n_modified_ways += 1
                else:
                    newtags = dict(way.tags)
                    newtags['lts'] = '4'
                    self.writer.add_way(way.replace(tags=newtags))


def add_lts_tags(osm_filename: str, out_filename: str, max_lts: int = 3) -> None:
    """
    Adds tags to OSM data indicating LTS for cyclists. Adds tags for levels 1, 2, 3, to cycle-able roads,
    and adds 4 to everything else.
    If max_lts is provided (as 0, 1, 2, 3, or 4), treats roads with higher LTS than max_lts as un-cycleable (lts 4)
    Args:
        osm_filename: input path
        out_filename: output path
        max_lts: the highest level of LTS tags to add (0, 1, 2, 3, or 4)
    """
    assert max_lts in [0, 1, 2, 3, 4]
    writer = osmium.SimpleWriter(out_filename)
    ltsadder = SimplestLTSAdder(writer, max_lts)
    ltsadder.apply_file(osm_filename)
    print(f'With max_lts {max_lts}, added lts=1,2, or 3 to {ltsadder.n_modified_ways}')

def define_experiences(input_dir, scenario_dir) -> pd.DataFrame:
    """
    Creates a new directory "routing environment" with OSM/GTFS files reflecting conditions as experienced
    by each combination of user class and mode
    Also assign each class/mode combination to a routing environment.
    """

    input_osm_filename = f"{input_dir}/osm_study_area.pbf"
    input_gtfs_dir = f"{input_dir}/GTFS/"
    input_traffic_dir = f"{input_dir}/traffic/"
    user_classes = pd.read_csv(f"{input_dir}/user_classes.csv")
    destination_dir = f"{scenario_dir}/routing/"
    save_userclasses_to = f"{scenario_dir}/routing/user_classes_with_routeenvs.csv"

    os.makedirs(destination_dir, exist_ok=True)

    # for now, we're going to assume the most basic version possible:
    # all subgroups experience the city the same way for CAR, TRANSIT, and WALK,
    # and there are only four different options for BIKE

    # TODO - this is where we'll have to adjust link-level driving impedances for toll roads

    universal_routing_env_id = "universal_re"
    # first let's do CAR, TRANSIT, and WALK:
    # 1) make the directory
    os.makedirs(f"{destination_dir}/{universal_routing_env_id}", exist_ok=True)
    # 2) copy the osm, traffic, and gtfs files
    shutil.copy(input_osm_filename,f"{destination_dir}/{universal_routing_env_id}/osm_file.pbf")
    shutil.copytree(input_traffic_dir, f"{destination_dir}/{universal_routing_env_id}/traffic/",dirs_exist_ok=True)
    shutil.copytree(input_gtfs_dir, f"{destination_dir}/{universal_routing_env_id}/gtfs_files/", dirs_exist_ok=True)

    # 3) create a parameters file for r5 routing
    r5_params = {
    }
    with open(f"{destination_dir}/{universal_routing_env_id}/r5_params.json", 'w') as f:
        json.dump(r5_params, f)

    # 4) assign that ID to the user_classes dataframe
    user_classes.loc[:, "routeenv_CAR"] = universal_routing_env_id
    # even userclasses without a car will use universal routing env to route car,
    # they just will only have the choice of ridehail later
    user_classes.loc[:, "routeenv_TRANSIT"] = universal_routing_env_id
    user_classes.loc[:, "routeenv_WALK"] = universal_routing_env_id

    # next let's do the bike options

    LTSs = [1,2,4] # by definition, we don't need to create a routing environment for lts=0
    for lts in LTSs:
        bike_env_id = f"bike_re_lts{lts}"
        # 1) make the directory
        os.makedirs(f"{destination_dir}/{bike_env_id}/", exist_ok=True)
        # 2) copy OSM, adding LTS tags
        add_lts_tags(input_osm_filename,
                     f"{destination_dir}/{bike_env_id}/osm_file.pbf",
                     lts)
        # copy GTFS, just for the heck of it, we won't actually use it for routing
        # maybe someday we'll route bike-to-transit options
        shutil.copytree(input_gtfs_dir, f"{destination_dir}/{bike_env_id}/gtfs_files/")
        # 3) create a parameters file for r5 routing
        r5_params = {
            "max_bicycle_traffic_stress": lts,
            "speed_walking": 0.5, #km/h TODO: formalize this somewhere
        }
        with open(f"{destination_dir}/{bike_env_id}/r5_params.json", 'w') as f:
            json.dump(r5_params, f)

        # 4) assign IDs

        selector = user_classes['max_bicycle'] == f'bike_lts{lts}'
        user_classes.loc[selector, "routeenv_BICYCLE"] = bike_env_id
    if not save_userclasses_to == "":
        user_classes.to_csv(save_userclasses_to)
    return user_classes