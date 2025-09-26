import pandas as pd

def evaluate_scenario(scenario_dir):
    user_classes = pd.read_csv(f"{scenario_dir}/user_classes_with_routeenvs.csv")