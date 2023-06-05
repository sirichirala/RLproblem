import os
import json
import logging
import typing
import random
import math
import csv
import time
from docplex.mp.model import Model
from docplex.mp.solution import *

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    static_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'static_instance.json')
    initial_condition_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'ic_instance.json')
    #result_path: str = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results_milp')), 'results.csv')
    result_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results_milp'))

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
    init_inv: typing.Dict[int, int]
    C: int
    A: int
    B: int

    @classmethod 
    def build(cls, cfg: Config):
        """
        1. Read the static_instance.json file.
        2. Collect all the static data:
            - P, T, D, L, S, V, Vmax, R, F, H, G
        """
        with open(cfg.static_instance_path, 'r') as file:
            static_instance_data = json.load(file)

        product_ids = [product['id'] for product in static_instance_data.get('products', [])]
        week_number = [week for week in range(static_instance_data.get('config', {})['num_time_points'])]
        demand = static_instance_data.get('demand_data',{})
        lead_time = static_instance_data['config']['lead_time_in_weeks']
        safety_stock_product = {product['id']: math.ceil(product['safety_stock_in_weeks']) for product in static_instance_data.get('products', [])}
        volume_product = {product['id']: product['volume'] for product in static_instance_data.get('products', [])}
        volume_max = static_instance_data['config']['container_volume']
        ramping_units = static_instance_data.get('ramping_factor', {})
        F = static_instance_data.get('fail_demand_cost', {})
        H = static_instance_data.get('overstocking_cost', {})
        G = static_instance_data.get('reward_recommended_stock', {})

        """
        1. Read the ic_instance.json file.
        2. Collect all initial condition info.
             - i_p,0
        """
        with open(cfg.initial_condition_instance_path, 'r') as file:
            ic_data = json.load(file)

        initial_inventory = {product['id']: product['on_hand_inventory'] for product in ic_data.get('products', [])}

        """
        1. Create rest of the model parameters.
            - C, A, B
        """
        num_containers = 50
        large_constant_1 = 10000
        large_constant_2 = 10000

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
            init_inv = initial_inventory,
            C = num_containers,
            A = large_constant_1,
            B = large_constant_2
            )

class OptimizationModel:
    def __init__(self, data, cfg):
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
        self.init_inv = data.init_inv
        self.C = data.C
        self.A = data.A
        self.B = data.B
        self.R = data.R

        self.model = None
        self.solve_time = None

        self.result_path = cfg.result_path

    def build_model(self):
        
        
        """
        Create Model
        """
        self.model = Model()

        """
        Define Decision Variables
        """
        self.x = {(p, t): self.model.integer_var(lb=0, name=f'x_{p}_{t}') for p in self.P for t in self.T}
        self.n = {t: self.model.integer_var(lb=0, name=f'n_{t}') for t in self.T}

        self.i = {(p, t): self.model.integer_var(lb=0, name=f'i_{p}_{t}') for p in self.P for t in self.T}
        self.u = {(p, t): self.model.integer_var(lb=0, name=f'u_{p}_{t}') for p in self.P for t in self.T}
        self.y = {(p, t): self.model.binary_var(name=f'y_{p}_{t}') for p in self.P for t in self.T}
        self.e = {(p, t): self.model.integer_var(lb=0, name=f'e_{p}_{t}') for p in self.P for t in self.T}
        self.z = {(p, t): self.model.binary_var(name=f'z_{p}_{t}') for p in self.P for t in self.T}
        self.o = {(p, t): self.model.integer_var(lb=0, name=f'o_{p}_{t}') for p in self.P for t in self.T}
        self.j = {(p, t): self.model.binary_var(name=f'j_{p}_{t}') for p in self.P for t in self.T}
        self.q = {(p, t): self.model.integer_var(lb=0, name=f'q_{p}_{t}') for p in self.P for t in self.T}
        self.k = {(p, t): self.model.binary_var(name=f'k_{p}_{t}') for p in self.P for t in self.T}
        self.m = {(p, t): self.model.binary_var(name=f'm_{p}_{t}') for p in self.P for t in self.T}

        """
        Add constraints
            - initial inventory constraints
            - Demand and inventory constraints
            - Shipping constraint
            - Ramping constraint
            - Variable restrictions
        """
        #initial inventory constraints
        for p in self.P:
            self.model.add_constraint(
                self.i[(p, 0)] == self.init_inv[p],
                ctname=f"initial_inventory_constraint_{p}"
            ) 

        #Demand and inventory constraints

        for p in self.P:
            for t in self.T:
                self.model.add_constraint(
                    self.x[(p, t)] + self.i[(p, t)] + self.u[(p, t)] == self.D[f"{p}_{t}"] + self.e[(p, t)],
                    ctname=f"demand_constraint3a_{p}_{t}"
                )
                self.model.add_constraint(
                    self.u[(p, t)] <= self.A * self.y[(p, t)],
                    ctname=f"demand_constraint3b_{p}_{t}"
                )
                self.model.add_constraint(
                    self.e[(p, t)] <= self.A * self.z[(p, t)],
                    ctname=f"demand_constraint3c_{p}_{t}"
                )
                self.model.add_constraint(
                    self.y[(p, t)] + self.z[(p, t)] <= 1,
                    ctname=f"demand_constraint3d_{p}_{t}"
                )
                if t+1 < len(self.T):
                    self.model.add_constraint(
                        self.e[(p, t)] == self.i[(p, t + 1)],
                        ctname=f"demand_constraint3e_{p}_{t}"
                    )
                self.model.add_constraint(
                    self.e[(p, t)] == self.S[p] * self.D[f"{p}_{t}"] + self.o[(p, t)] - self.q[(p, t)],
                    ctname=f"demand_constraint3f_{p}_{t}"
                )
                self.model.add_constraint(
                    self.o[(p, t)] <= self.B * self.j[(p, t)],
                    ctname=f"demand_constraint3g_{p}_{t}"
                )
                self.model.add_constraint(
                    self.q[(p, t)] <= self.B * self.k[(p, t)],
                    ctname=f"demand_constraint3h_{p}_{t}"
                )
                self.model.add_constraint(
                    self.j[(p, t)] + self.k[(p, t)] <= 1,
                    ctname=f"demand_constraint3i_{p}_{t}"
                )
                self.model.add_constraint(
                    self.j[(p, t)] + self.k[(p, t)] + self.m[(p, t)] == 1,
                    ctname=f"demand_constraint3j_{p}_{t}"
                )

        #Shipping constraint
        
        for t in self.T:
            self.model.add_constraint(
                self.n[t] <= self.C,
                ctname=f"shipping_constraint_{t}"
            )
            self.model.add_constraint(
                self.model.sum(self.V[p] * self.x[(p, t)] for p in self.P) <= self.Vmax * self.n[t],
                ctname=f"shipping_constraint1_{t}"
            )
            self.model.add_constraint(
                0.98 * self.Vmax * self.n[t] <= self.model.sum(self.V[p] * self.x[(p, t)] for p in self.P),
                ctname=f"shipping_constraint2_{t}"
            )
        

        #Ramping constraint
        for p in self.P:
            for t in self.T:
                if t > 0:
                    self.model.add_constraint(
                        self.x[(p, t)] - self.x[(p, t - 1)] <= self.R[f"{p}"],
                        ctname=f"ramping_constraint1_{p}_{t}"
                    )
                    self.model.add_constraint(
                        -self.R[f"{p}"] <= self.x[(p, t)] - self.x[(p, t - 1)],
                        ctname=f"ramping_constraint2_{p}_{t}"
                    )

        #Variable restrictions

        for p in self.P:
            for t in self.T:
                self.model.add_constraint(self.x[(p, t)] >= 0)
                self.model.add_constraint(self.i[(p, t)] >= 0)
                self.model.add_constraint(self.u[(p, t)] >= 0)
                self.model.add_constraint(self.e[(p, t)] >= 0)
                self.model.add_constraint(self.o[(p, t)] >= 0)
                self.model.add_constraint(self.q[(p, t)] >= 0)
                self.model.add_constraint(self.x[(p, t)].is_integer())
                self.model.add_constraint(self.i[(p, t)].is_integer())
                self.model.add_constraint(self.u[(p, t)].is_integer())
                self.model.add_constraint(self.e[(p, t)].is_integer())
                self.model.add_constraint(self.o[(p, t)].is_integer())
                self.model.add_constraint(self.q[(p, t)].is_integer())
                self.model.add_constraint(self.y[(p, t)].is_binary())
                self.model.add_constraint(self.z[(p, t)].is_binary())
                self.model.add_constraint(self.j[(p, t)].is_binary())
                self.model.add_constraint(self.k[(p, t)].is_binary())
                self.model.add_constraint(self.m[(p, t)].is_binary())

        # Define the objective function expression
        objective_expr = self.model.sum(self.F[f"{p}"] * self.u[p, t] for p in self.P for t in self.T) + \
                            self.model.sum(self.H[f"{p}"] * self.o[p, t] for p in self.P for t in self.T) - \
                             self.model.sum(self.G[f"{p}"] * self.m[p, t] for p in self.P for t in self.T)

        # Set the objective function
        self.model.minimize(objective_expr)



    def optimize(self):
        start_time = time.time()  # Capture the start time

        self.model.export_as_lp("optimization_model.lp")
        self.model.solve()

        end_time = time.time()  # Capture the end time
        self.solve_time = end_time - start_time  # Calculate the solve time
    
        if self.model.solution.is_valid_solution:
            logging.debug(f"Notify end solve, status=JobSolveStatus.OPTIMAL_SOLUTION, solve_time={self.solve_time}")
            logging.info(f"Notify end solve, status=JobSolveStatus.OPTIMAL_SOLUTION, solve_time={self.solve_time}")
        else:
            log.warning('Optimization did not result in an optimal solution.')
        
    
    def write_to_csv(self):
        """
        1. create a results.csv file in the results_milp folder which has the optimality status, objective function values, solve times for instances run.
        2. creata a separate result_instance_# file for each instance with the decision variables.
        """
        result_file = os.path.join(self.result_path, 'results.csv')
        result_data = [self.model.solve_status, self.model.solution.objective_value, self.solve_time]
        
        file_exists = os.path.isfile(result_file)

        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            # Write header if the file doesn't exist
            if not file_exists:
                writer.writerow(["Instance name", "Optimality Status", "Objective Function", "Solve Time"])
        file.close()
        with open(result_file, "r") as file:
            no_lines = sum(1 for _ in file)
        # Write result data
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([no_lines] + result_data)

        #need self.x[(p, t)], self.n[t]
        result_instance_file = os.path.join(self.result_path, 'result_instance_' + str(no_lines) + '.csv')
        with open(result_instance_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time period", "n_t"] + self.P)
            for t in self.T:
                row_values = [t, self.model.solution.get_value(self.n[t])]
                for p in self.P:
                     row_values.append(self.model.solution.get_value(self.x[p,t]))
                writer.writerow(row_values)



def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    cfg = Config()
    log.info('Starting data read from both json files for static data and initial conditions.')
    data = Data.build(cfg)
    log.info('Data read complete.')
    log.info('Building optimization model.')
    model = OptimizationModel(data, cfg)
    model.build_model()
    log.info('Optimization model build complete.')
    log.info('Running Optimization model.')
    model.optimize()
    log.info('results to csv.')
    model.write_to_csv()
    

if __name__ == '__main__':
    main()
