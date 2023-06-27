import os
import json
import logging
import typing
import math
import csv
import time

from scm_env import *

#import gymnasium for open ai gym compatible env
import gym
from gym import spaces
import gymnasium

#using stable_baselines3 for RL algorithms
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C



log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    static_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'static_instance.json')
    initial_condition_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'ic_instance.json')
    result_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results_rl'))

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




def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    
    cfg = Config()
    
    log.info('Starting data read from both json files for static data and initial conditions.')
    data = Data.build(cfg)
    log.info('Data read complete.')
    
    log.info('Setting up reinforcement learning states, actions and environment.')
    environment = ScmEnv(data)
    log.info('Environment setup complete.')
    
    log.info('check env compatibility with OpenAI gym')
    compatible = isinstance(environment, gymnasium.Env)
    if compatible:
        print("The ScmEnv is OpenAI Gym compatible.")
    else:
        print("The ScmEnv is not OpenAI Gym compatible.")
    
    log.info('running reinforcement learning algorithms using stable baselines.')
    #Register your custom environment
    #gym.register(id='ScmEnv-v0', entry_point=ScmEnv(data))
    
    log.info('checking environment using stable baselines3')
    check_env(environment)

    model = A2C("MlpPolicy", environment).learn(10000)
    print(model)
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    

if __name__ == '__main__':
    main()
