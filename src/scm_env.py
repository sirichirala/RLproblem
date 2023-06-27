import math
import numpy as np

import gymnasium
from gymnasium import spaces    

class ScmEnv(gymnasium.Env):

    def __init__(self, data):
        """
        1. Import all necessary parameters.
        2. Define self.observation_space:
            - observation: The number of products of type 'p' present in the warehouse after considering demand.
        3. Define self.action_space:
            - action: The number of products of type 'p' ordered to the warehouse in container 'c'.
        4. Create additional instance variables.
        """
        self.P = data.P # List of product id's
        self.T = len(data.T) # Episode length in days
        self.D = data.D # Dictionary productid_weeknum : Demand
        self.L = data.L # int, lead time (weeks)
        self.S = data.S # Dictionary productid : safety stock weeks
        self.V = data.V #Dictionary productid : volume
        self.Vmax = 20 #float, volume max
        self.F = data.F #Dictionary productid : cost
        self.H = data.H #Dictionary productid : cost
        self.G = data.G #Dictionary productid : cost
        self.init_inv = data.init_inv #Dictionary productid : initial inventory
        self.C = 3 # int, max containers
        self.R = data.R #Dictionary productid : ramping units

        # Define the state/observation space 
        self.observation_space = self.create_observation_space()
        #print("observation sample")
        #print(self.observation_space.sample())

        # Define the action space
        self.action_space = self.create_action_space()
        #print("action sample")
        #print(self.action_space.sample())

        # Additional instance variables
        self.current_obs = None
        self.week_num = 0 #Time step for the episode


    def create_observation_space(self):
        LB = np.array([0 for p in self.P], dtype=np.int32) # Lower bound of 0 for each product
        UB_obs = np.array([np.floor(self.Vmax/self.V[p])*self.C + self.init_inv[p] for p in self.P], dtype=np.int32) # Upper bound for each product for observation
        return spaces.Box(low=LB, high=UB_obs, dtype=np.int32)

    def create_action_space(self):
        # Calculate the upper bound for each product and container
        ub = np.zeros((self.C, len(self.P)), dtype=np.int32)
        for c in range(self.C):
            for i, p in enumerate(self.P):
                ub[c][i] = np.floor(self.Vmax / self.V[p])

        action_space = spaces.Box(low=0, high=ub, shape=(self.C, len(self.P)), dtype=np.int32)
        return action_space

    def reset(self, seed=None):
        """
        Reset environment to initial state/first observation to start new episode.
        Must return observation of the initial state.
        """
        self.current_obs = np.array([self.init_inv[p] for p in self.P], dtype=np.int32)
        self.week_num = 0

        return self.current_obs, {}

    def step(self, action):
        """
        Takes action as argument and performs one transition step.
        Given current observation, returns next observation, reward obtained in transition, whether current obs is terminal state, truncated state.
        Also: some additional info, check documentation.  
        """

        #method parameters
        next_obs = np.zeros(len(self.P), dtype=np.int32)
        added_demand_units = np.zeros(len(self.P), dtype=np.int32)
        unmet_demand_units = np.zeros(len(self.P), dtype=np.int32)
        penalty_F = 0
        penalty_H = 0
        reward_G = 0
        reward = 0

        # Compute next observation
            # 1. Take the current observation and action argument.
            # 2. Perform action. 
            # 3. Return next state (total products left after considering demand.)
        for index, element in np.ndenumerate(self.current_obs):
            p_id = self.P[index[0]]
            added_demand_units[index] = np.sum(action[:, index])
            if (element + added_demand_units[index]) >  self.D[f"{p_id}_{self.week_num}"]:
                next_obs[index] = (element + added_demand_units[index]) - self.D[f"{p_id}_{self.week_num}"]
                unmet_demand_units[index] = 0
            else:
                next_obs[index] = 0
                unmet_demand_units[index] = self.D[f"{p_id}_{self.week_num}"] - (element + added_demand_units[index])

        # Compute reward
            # 1. Penalty for failing to meet demand. (F)
            # 2. Penalty for overstocking inventory. (H)
            # 3. Reward for maintaining recommended safety stock. (G)
            
        for index, element in np.ndenumerate(next_obs):
            p_id = self.P[index[0]]
            if element == 0:
                penalty_F -= self.F[f"{p_id}"]*unmet_demand_units[index]
            elif element > 0:
                if element > self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]:
                    penalty_H -= self.H[f"{p_id}"]*(element - (self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]))
                elif element == self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]:
                    reward_G += self.G[f"{p_id}"]
        reward = penalty_F + penalty_H + reward_G

        # Adding high penalty for taking an action that is infeasible.
        #compute episode truncated conditions.
        episode_truncated = False
        for action_list in action:
            sum_values = 0
            for index, element in enumerate(self.V):
                sum_values += action_list[index] * self.V[element]
            if sum_values <= 0.98 * self.Vmax or sum_values >= self.Vmax:
                episode_truncated = True
                reward -= 100000
                break

        # Compute episode done conditions.
        self.week_num += 1
        episode_done = False
        if self.week_num >= self.T:
            episode_done = True
            self.week_num = 0
        
        # update current_obs.
        self.current_obs = next_obs

        # info must be dictionary.
        return self.current_obs, reward, episode_done, episode_truncated, {}


    
    def render(self, mode="human"):
        """
        Displays current env state: e.g. graphical window in 'CartPole v1'.
        Method must be implemented, however Can leave empty if visualization is not important.
        Returns: None
        """
        pass

    def close(self):
        """
        Optional method for implementation.
        Used to clean up all resources (threads, graphical windows, etc.)
        Returns: None
        """
        pass

    def seed(self, seed= None):
        """
        Optional method for implementation.
        Set seed for environments random number generator for obtaining deterministic behavior.
        Returns: List of seeds.
        """
        return

    