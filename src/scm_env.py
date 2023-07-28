import math
import numpy as np
import scipy.stats as stats

import gymnasium
from gymnasium import spaces    

class ScmEnv(gymnasium.Env):

    def __init__(self, data):
        """
        1. Import all necessary parameters.
        2. Define self.observation_space:
            - observation (Box): The number of products of type 'p' present in the warehouse after considering demand (np.array(len(self.P)))+
                           The Demand of products of type 'p' for next 'data.L' weeks (np.array(self.L*len(self.P)))+
                           If shipping container limits have been violated for action in previous step (int(0 or 1)) +
                           Number of ramping violations for products for action in previous step (int(0, len(self.P)))
        3. Define self.action_space:
            - action (Box): The number of products of type 'p' ordered to the warehouse (np.array(len(self.P))).
        4. Create additional instance variables.
        """
        self.P = data.P # List of product id's
        self.N = data.N # int, max containers
        self.V = data.V # Dictionary productid : volume
        self.Vmax = data.Vmax # float, volume max
        self.S = data.S # Dictionary productid : safety stock weeks
        self.R = data.R # Dictionary productid : ramping units
        self.F = data.F # Dictionary productid : cost
        self.H = data.H # Dictionary productid : cost
        self.G = data.G # Dictionary productid : cost
        
        self.L = 3 # int, Number of weeks of future demands to add to observation space
        self.T = data.T # Episode length in weeks
        
        self.init_inv = data.init_inv # Dictionary productid : initial inventory
        self.gamma_params = data.gamma_params # Dictionary{productid : Dictionary{alpha, loc, scale}}
        self.Dmax_p200640680 = 50 
        self.Dmax_p200527730  = 15
        self.D = self.create_demands_episode() # Dictionary productid_weeknum : Demand
        self.D_pred = data.D # Demands for prediciton.

        # Define the state/observation space 
        self.observation_space = self.create_observation_space()

        # Define the action space
        self.action_space = self.create_action_space()
        #print("action sample")
        #print(self.action_space.sample())
        #print(self.action_space.high)
        #print(self.action_space.low)
        

        # Additional instance variables
        self.current_obs = None
        self.previous_action = None
        self.week_num = 0 #Time step for the episode
        

    def create_demands_episode(self):
        demand_data = {}        
        for p in self.P:
            demand = stats.gamma.rvs(self.gamma_params[f"{p}"]['alpha'], self.gamma_params[f"{p}"]['loc'], self.gamma_params[f"{p}"]['scale'], size= len(self.T) + self.L)
            demand = np.ceil(demand).astype(int)
            demand += np.random.randint(0, 6, size=len(demand))
            for week in range(len(self.T) + self.L):
                key = f"{p}_{week}"  # String representation of the key since key cant be tuple
                demand_data[key] = int(abs(demand[week]))
        return demand_data
    
    def create_observation_space(self):
        # Calculate the upper bounds for products and demands.
        UB_num_products = np.array([len(self.T)*self.Dmax_p200640680*(1 + 2*value) if i < len(self.S) - 8 else len(self.T)*self.Dmax_p200527730*(1 + 2*value) for i, value in enumerate(self.S.values())], dtype=np.int32) # Upper bound for each product for observation
        UB_Demand = np.full(len(self.P), self.Dmax_p200640680, dtype=np.int32)
        UB_Demand[-8:] = self.Dmax_p200527730
        UB_shipping = np.array([1], dtype=np.int32)
        UB_ramping = np.array([len(self.P)], dtype=np.int32)
        #UB_ramping_abs = np.array([self.Dmax_p200640680*(1 + 2*value) if i < len(self.S) - 8 else self.Dmax_p200527730*(1 + 2*value) for i, value in enumerate(self.S.values())], dtype=np.int32)
        UB = np.hstack((UB_num_products, np.tile(UB_Demand, self.L), UB_shipping, UB_ramping))
        return spaces.Box(low=0, high=UB, dtype=np.int32)

    def create_action_space(self):
        # Calculate the upper bound for number of orders for each product
        UB = np.array([self.Dmax_p200640680*(1 + 2*value) if i < len(self.S) - 8 else self.Dmax_p200527730*(1 + 2*value) for i, value in enumerate(self.S.values())], dtype=np.int32)
        action_space = spaces.Box(low=0, high=UB, dtype=np.float32)
        return action_space

    def prediction_reset(self, seed=None):
        """
        Reset environment to initial state/first observation to start new episode.
        """
        self.week_num = 0

        current_products = np.array([self.init_inv[p] for p in self.P], dtype=np.int32)

        self.D = self.D_pred
        for p in self.P:
            demand = stats.gamma.rvs(self.gamma_params[f"{p}"]['alpha'], self.gamma_params[f"{p}"]['loc'], self.gamma_params[f"{p}"]['scale'], size= self.L)
            demand = np.ceil(demand).astype(int)
            demand += np.random.randint(0, 6, size=len(demand))
            for week in range(self.L):
                key = f"{p}_{week+len(self.T)}"  # String representation of the key since key cant be tuple
                self.D[key] = int(abs(demand[week]))

        demand_products = []
        for week in range(1, self.L+1):
            for p in self.P:
                demand_products.append(self.D[f"{p}_{week}"])
        demand_products = np.array(demand_products, dtype=np.int32)

        shipping = np.array([0])
        ramping = np.array([0])
        #ramping_abs = np.array([0]*len(self.P))
        
        self.current_obs = np.hstack((current_products,demand_products, shipping, ramping)) # Observation is number of products left after considering demand followed by the demands for products for next data.L weeks.
        return self.current_obs, {}

    def reset(self, seed=None):
        """
        Reset environment to initial state/first observation to start new episode.
        """
        self.week_num = 0

        current_products = np.array([self.init_inv[p] for p in self.P], dtype=np.int32)

        demand_products = []
        self.create_demands_episode()
        for week in range(1, self.L+1):
            for p in self.P:
                demand_products.append(self.D[f"{p}_{week}"])
        demand_products = np.array(demand_products, dtype=np.int32)

        shipping = np.array([0])
        ramping = np.array([0])
        #ramping_abs = np.array([0]*len(self.P))

        self.current_obs = np.hstack((current_products,demand_products, shipping, ramping)) # Observation is number of products left after considering demand followed by the demands for products for next data.L weeks.
        return self.current_obs, {}

    def step(self, action):
        """
        Takes action as argument and performs one transition step.
        Given current observation, returns next observation, reward obtained in transition, whether current obs is terminal state, truncated state.
        Also: some additional info, check documentation.  
        """
        action = np.floor(action)
    
        #method parameters
        #next_obs = np.zeros(len(self.P)*(1+self.L) + 2 + len(self.P), dtype=np.int32)
        next_obs = np.zeros(len(self.P)*(1+self.L) + 2, dtype=np.int32)

        added_demand_units = np.zeros(len(self.P), dtype=np.int32)
        unmet_demand_units = np.zeros(len(self.P), dtype=np.int32)
        penalty_F = 0
        penalty_H = 0
        reward_G = 0
        reward = 0
        milp_reward = 0
        shipping = 0
        ramping = 0
        #ramping_abs = np.zeros(len(self.P), dtype=np.int32)

        # Compute next observation
            # 1. Take the current observation and action argument.
            # 2. Perform action. 
            # 3. Return next state (total products left after considering demand.)
        for index, element in np.ndenumerate(self.current_obs[:len(self.P)]):
            p_id = self.P[index[0]]
            added_demand_units[index] = action[index]
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
            
        for index, element in np.ndenumerate(next_obs[:len(self.P)]):
            p_id = self.P[index[0]]
            if element == 0:
                penalty_F -= self.F[f"{p_id}"]*unmet_demand_units[index]
            elif element > 0:
                if element > self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]:
                    penalty_H -= self.H[f"{p_id}"]*(element - (self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]))
                elif element == self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]:
                    reward_G += self.G[f"{p_id}"]
        reward = penalty_F + penalty_H + reward_G

        # Compute milp equivalent reward
        milp_reward = reward

        """
        Adding high penalty for taking an action that is infeasible.
        1. Violating Vmax - shipping constraints
        2. Violating ramping constraints
        """

        
        # shipping:
        sum_values = 0
        for index, element in enumerate(self.V.values()):
            sum_values += action[index] * element
        if sum_values >= self.Vmax * self.N:
            reward -= 100000
            shipping = 1
        else:
            reward += 100000
            #print(shipping)
                
        # Ramping:
        if self.week_num != 0:
            for index, (curr, nxt, r) in enumerate(zip(action, self.previous_action, self.R.values())):
                #ramping_abs[index]=(abs(curr-nxt))
                if abs(curr - nxt) >= r:
                    reward -= 100000
                    ramping +=1
            if ramping == 0:
                reward += 2500000
    
        # update current_obs.
        index=0
        for week in range(self.week_num+1, self.week_num+self.L+1):
            for p in self.P:
                next_obs[len(self.P)+index] = self.D[f"{p}_{week}"]
                index +=1 

        next_obs[-2] = shipping 
        next_obs[-1] = ramping
        #next_obs[-len(self.P)-2] = shipping 
        #next_obs[-len(self.P)-1] = ramping
        #next_obs[-len(self.P):] = ramping_abs
        self.current_obs = next_obs

        #update previous action
        self.previous_action = action

        # Compute episode done and truncated conditions.
        self.week_num += 1
        episode_done = False
        episode_truncated = False
        if self.week_num >= len(self.T):
            episode_done = True
            self.week_num = 0
    
        # info must be dictionary.
        return self.current_obs, reward, episode_done, episode_truncated, {'milp_reward':milp_reward, 'shipping_violated':shipping, 'ramping_violated':ramping}

    
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

    def seed(self, seed=None):
        """
        Optional method for implementation.
        Set seed for environments random number generator for obtaining deterministic behavior.
        Returns: List of seeds.
        """
        return

    
    