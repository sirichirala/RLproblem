import os
import json
import logging
import typing
import math
import csv
import pandas as pd

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    optimal_action_file_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'milp_results', 'SAA_result_instance_action.csv'))
    num_instances = 200
    saa_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'SAA')
    saa_result_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'milp_results'))

class Data(typing.NamedTuple):
    config: Config
    P: typing.List[int]
    T: typing.List[int]
    D: typing.Dict[str, int]
    L: int
    S: typing.Dict[int, float]
    V: typing.Dict[int, float]
    Vmax: float
    R: typing.Dict[str, int]
    F: typing.Dict[str, float]
    H: typing.Dict[str, float]
    G: typing.Dict[str, float]
    alpha: int
    init_inv: typing.Dict[int, int]
    N: int
    A: int
    B: int
    C: typing.List[typing.Set[str]]

    @classmethod 
    def build(cls, cfg: Config, initial_condition_instance_path, static_instance_path):
        """
        1. Read the static_instance.json file.
        2. Collect all the static data:
            - P, T, D, L, S, V, Vmax, R, F, H, G, alpha
        """
        with open(static_instance_path, 'r') as file:
            static_instance_data = json.load(file)

        product_ids = [product['id'] for product in static_instance_data.get('products', [])]
        week_number = [week for week in range(static_instance_data.get('config', {})['num_time_points'])]
        demand = static_instance_data.get('demand_data',{})
        lead_time = static_instance_data['config']['lead_time_in_weeks']
        safety_stock_product = {product['id']: math.floor(product['safety_stock_in_weeks']) for product in static_instance_data.get('products', [])}
        volume_product = {product['id']: product['volume'] for product in static_instance_data.get('products', [])}
        volume_max = static_instance_data['config']['container_volume']
        ramping_units = static_instance_data.get('ramping_factor', {})
        F = static_instance_data.get('fail_demand_cost', {})
        H = static_instance_data.get('overstocking_cost', {})
        G = static_instance_data.get('reward_recommended_stock', {})
        alpha = static_instance_data.get('proportionality_cost')

        """
        1. Read the ic_instance.json file.
        2. Collect all initial condition info.
             - i_p,0
        """
        with open(initial_condition_instance_path, 'r') as file:
            ic_data = json.load(file)

        initial_inventory = {product['id']: product['on_hand_inventory'] for product in ic_data.get('products', [])}

        """
        1. Create rest of the model parameters.
            - N, A, B, C
        """
        num_containers = 50
        large_constant_1 = 10000
        large_constant_2 = 10000


    
        # Get the set of distinct safety stock values
        distinct_safety_stock_values = set(safety_stock_product.values())
        # Create the collection of sets C
        C = [set(product_id for product_id, safety_stock_value in safety_stock_product.items() if safety_stock_value == s)
            for s in distinct_safety_stock_values]
        

        return cls(config = cfg,
            P = product_ids,
            T = week_number,
            D = demand,
            L = lead_time,
            S = safety_stock_product,
            V = volume_product,
            Vmax = volume_max,
            R = ramping_units,
            F = F,
            H = H,
            G = G,
            alpha = alpha,
            init_inv = initial_inventory,
            N = num_containers,
            A = large_constant_1,
            B = large_constant_2,
            C = C
            )

class CalculateObjective:
    def __init__(self, data, cfg, optimal_action):
        self.P = data.P
        self.T = data.T
        self.D = data.D
        self.L = data.L
        self.S = data.S
        self.V = data.V
        self.Vmax = data.Vmax
        self.F = data.F
        self.H = data.H
        self.G = data.G
        self.alpha = data.alpha
        self.init_inv = data.init_inv
        self.N = data.N
        self.A = data.A
        self.B = data.B
        self.R = data.R
        self.C = data.C

        self.optimal_action = optimal_action
        self.result_path = cfg.saa_result_path

        added_units = {}
        unmet_units = {}
        for p in self.P:
            for t in self.T:
                key = f"{p}_{t}"
                added_units[key], unmet_units[key] = 0, 0
        
        penalty_F = 0
        penalty_H = 0
        reward_G = 0
        
        for t in self.T:
            for p in self.P:
                if t == 0:
                    if self.init_inv[p] + self.optimal_action[f"{p}_{t}"] > self.D[f"{p}_{t}"]:
                        added_units[f"{p}_{t}"] = (self.init_inv[p] + self.optimal_action[f"{p}_{t}"]) - self.D[f"{p}_{t}"]
                        unmet_units[f"{p}_{t}"] = 0
                    else:
                        added_units[f"{p}_{t}"] = 0
                        unmet_units[f"{p}_{t}"] = self.D[f"{p}_{t}"] - (self.init_inv[p] + self.optimal_action[f"{p}_{t}"])
                else:
                    if added_units[f"{p}_{t-1}"] + self.optimal_action[f"{p}_{t}"] > self.D[f"{p}_{t}"]:
                        added_units[f"{p}_{t}"] = (added_units[f"{p}_{t-1}"] + self.optimal_action[f"{p}_{t}"]) - self.D[f"{p}_{t}"]
                        unmet_units[f"{p}_{t}"] = 0
                    else:
                        added_units[f"{p}_{t}"] = 0
                        unmet_units[f"{p}_{t}"] = self.D[f"{p}_{t}"] - (added_units[f"{p}_{t-1}"] + self.optimal_action[f"{p}_{t}"])

        for t in self.T:
            for p in self.P:
                if added_units[f"{p}_{t}"] == 0:
                    penalty_F += self.F[f"{p}"]*unmet_units[f"{p}_{t}"]
                elif added_units[f"{p}_{t}"] > 0:
                    if added_units[f"{p}_{t}"] > self.D[f"{p}_{t}"]*self.S[p]:
                        penalty_H += (self.H[f"{p}"]*(added_units[f"{p}_{t}"] - (self.D[f"{p}_{t}"]*self.S[p])))
                    elif added_units[f"{p}_{t}"] == self.D[f"{p}_{t}"]*self.S[p]:
                        reward_G -= self.G[f"{p}"]*added_units[f"{p}_{t}"]
                    elif added_units[f"{p}_{t}"] < self.D[f"{p}_{t}"]*self.S[p]:
                        reward_G -= self.G[f"{p}"]*added_units[f"{p}_{t}"]
        self.reward = penalty_F + penalty_H + reward_G
    
    def write_to_csv(self):

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        result_file = os.path.join(self.result_path, 'SAA_result_instances.csv')
        result_data = [self.reward]
        
        file_exists = os.path.isfile(result_file)

        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            # Write header if the file doesn't exist
            if not file_exists:
                writer.writerow(["Instance name", "Objective Function"])
        file.close()
        with open(result_file, "r") as file:
            no_lines = sum(1 for _ in file)
        # Write result data
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([no_lines] + result_data)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    cfg = Config()

    df = pd.read_csv(cfg.optimal_action_file_path)
    optimal_action = {}
    for i, row in df.iterrows():
        for j, col in enumerate(df.columns[1:]):
            key = f"{col}_{i}"
            value = row[col] 
            optimal_action[key] = value

    
    log.info('Starting data read from both json files for static data and initial conditions.')
    for i in range(cfg.num_instances):
        ic_filename = os.path.join(cfg.saa_instance_path, f"ic_instance_{i}.json")
        static_filename = os.path.join(cfg.saa_instance_path, f"static_instance_{i}.json")
        data = Data.build(cfg, ic_filename, static_filename)
        log.info('Data read complete.')
        result = CalculateObjective(data, cfg, optimal_action)
        result.write_to_csv()
    

if __name__ == '__main__':
    main()