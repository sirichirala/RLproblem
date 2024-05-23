import random
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler


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
        self.R = data.R # Dictionary productid : ramping units
        self.S = data.S # Dictionary productid : safety stock weeks
        self.F = data.F # Dictionary productid : cost
        self.H = data.H # Dictionary productid : cost
        self.G = data.G # Dictionary productid : cost
        self.L = 3 # int, Number of weeks of future demands to add to observation space
        self.T = data.T # Episode length in weeks
        
        self.init_inv = data.init_inv # Dictionary productid : initial inventory
        self.inv_max = 5
        self.gamma_params = data.gamma_params # Dictionary{productid : Dictionary{alpha, loc, scale}}
        self.Dmax_p200640680 = 15 
        self.Dmax_p200527730  = 15
        self.D = 0 # Dictionary productid_weeknum : Demand
        self.D_pred = data.D # Demands for prediciton.


        # Define the state/observation space 
        self.observation_space = self.create_observation_space()
        """
        print("observation sample")
        print(self.observation_space.sample()) 
        print(self.observation_space.high)
        print(self.observation_space.low)
        """
        


        # Define the action space
        self.action_space = self.create_action_space()
        """
        print("action sample")
        print(self.action_space.sample())
        print(self.action_space.high)
        print(self.action_space.low)
        """

        self.current_obs = None
        self.previous_action = np.zeros(len(self.P), dtype=np.int32)
        self.week_num = 0 #Time step for the episode
        self.train_same_instance_number = 1
        self.train_same_instance_counter = 0 #Counter for training same instances
        self.train_instance_type = 3


        
    def create_demands_episode(self):
        demand_data = {}        
        for p in self.P:
            demand = [random.randint(5,15) for _ in range(len(self.T) + self.L)]
            #demand = stats.gamma.rvs(self.gamma_params[f"{p}"]['alpha'], self.gamma_params[f"{p}"]['loc'], self.gamma_params[f"{p}"]['scale'], size= len(self.T) + self.L)
            #demand = np.ceil(demand).astype(int)
            #demand += np.random.randint(0, 3, size=len(demand))
            for week in range(len(self.T)):
                key = f"{p}_{week}"  # String representation of the key since key cant be tuple
                demand_data[key] = int(abs(demand[week]))
            demand_data[f"{p}_{len(self.T)}"] = 0
        return demand_data


    def create_observation_space(self):
        # Calculate the bounds for products and demands.
        self.obs_ub_num_products = np.array([len(self.T)*self.Dmax_p200527730*(1 + value) + self.inv_max for i, value in enumerate(self.S.values())], dtype=np.int32)
        self.obs_ub_Demand = np.array([self.Dmax_p200527730 for i, value in enumerate(self.S.values())], dtype=np.int32)
        self.obs_ub_shipping = np.array([1], dtype=np.int32)
        self.obs_ub_ramping = np.array([len(self.P)], dtype=np.int32)
        UB=np.hstack((self.obs_ub_num_products, self.obs_ub_Demand, self.obs_ub_shipping, self.obs_ub_ramping))
        observation_space = spaces.Box(low=0, high=UB, dtype=np.int32)
        return observation_space
    
    def create_action_space(self):
        # Calculate the upper bound for number of orders for each product
        self.action_ub = np.array([self.Dmax_p200527730*(1 + value) for i, value in enumerate(self.S.values())], dtype=np.int32)
        action_space = spaces.Box(low=0, high=self.action_ub, dtype=np.int32)
        #action_space = spaces.MultiDiscrete(self.action_ub)
        return action_space


    def prediction_reset(self, seed=None):
        """
        Reset environment to initial state/first observation to start new episode.
        """
        self.week_num = 0


        current_products = np.array([self.init_inv[p] for p in self.P], dtype=np.int32)


        self.D = self.D_pred
        for p in self.P:
            self.D[f"{p}_{len(self.T)}"] = 0


        demand_products = []
        for p in self.P:
            demand_products.append(self.D[f"{p}_{0}"])
        demand_products = np.array(demand_products, dtype=np.int32)
        
        shipping = np.array([0])
        ramping = np.array([0])

        self.current_obs = np.hstack((current_products, demand_products, shipping, ramping))
        return self.current_obs, {}


    def reset(self, seed=None):
        """
        Reset environment to initial state/first observation to start new episode.
        """
        self.week_num = 0
        
        if self.train_same_instance_counter == self.train_same_instance_number:
            self.train_same_instance_counter = 0


        if self.train_same_instance_counter == 0:
            self.D = self.create_demands_episode()
            if self.train_instance_type == 3:
                self.current_products = np.array([random.randint(0,self.inv_max) for i in range (0, len(self.P))], dtype=np.int32)
                self.train_instance_type -=1
            elif self.train_instance_type == 2:
                self.current_products = np.array([random.randint(0,self.inv_max) for i in range (0, len(self.P))], dtype=np.int32)
                self.train_instance_type -=1
            elif self.train_instance_type == 1:
                self.current_products = np.array([random.randint(0,self.inv_max) for i in range (0, len(self.P))], dtype=np.int32)
                self.train_instance_type = 3
        self.train_same_instance_counter += 1
        """
        self.D = self.D_pred
        for p in self.P:
            self.D[f"{p}_{len(self.T)}"] = 0
        """
        
        demand_products = []
        for p in self.P:
            demand_products.append(self.D[f"{p}_{0}"])
        demand_products = np.array(demand_products, dtype=np.int32)
        
        shipping = np.array([0])
        ramping = np.array([0])
        
        self.current_obs = np.hstack((self.current_products, demand_products, shipping, ramping)) # Observation is number of products left after considering demand followed by the demands for products for this week.
        
        return self.current_obs, {}


    def step(self, action):
        """
        Takes action as argument and performs one transition step.
        Given current observation, returns next observation, reward obtained in transition, whether current obs is terminal state, truncated state.
        Also: some additional info, check documentation.  
        """
        action = np.floor(action)
        
        #method parameters
        next_obs = np.zeros(len(self.P)*(2) + 1 + 1, dtype=np.int32)
        added_demand_units = np.zeros(len(self.P), dtype=np.int32)
        unmet_demand_units = np.zeros(len(self.P), dtype=np.int32)


        penalty_F = 0
        penalty_H = 0
        reward_G = 0

        reward = 0
        milp_reward = 0


        shipping = 0
        ramping = 0


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
                penalty_F += -1*(self.F[f"{p_id}"]*unmet_demand_units[index])
            elif element > 0:
                if element > self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]:
                    penalty_H += -1*(self.H[f"{p_id}"]*(element - (self.D[f"{p_id}_{self.week_num}"]*self.S[p_id])))
                elif element == self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]:
                    reward_G += self.G[f"{p_id}"]*(element)
                #added
                elif element < self.D[f"{p_id}_{self.week_num}"]*self.S[p_id]:
                    reward_G += self.G[f"{p_id}"]*(element)
        reward = penalty_F + penalty_H + reward_G
        


        # Compute milp equivalent reward
        milp_reward = reward


        """
        Adding high penalty for taking an action that is infeasible.
        1. Violating Vmax - shipping constraints
        """
        # shipping:
        sum_values = 0
        for index, element in enumerate(self.V.values()):
            sum_values += action[index] * element
        if sum_values > self.Vmax * self.N:
            episode_truncated = False
            reward += -10
            shipping = 1
        else:
            episode_truncated = False
            reward += 0

        #ramping
        if self.week_num != 0:
            for curr, nxt, r in zip(self.current_obs[:len(self.P)], next_obs[:len(self.P)], self.R.values()):
                if abs(curr - nxt) >= r:
                    reward -= 10
                    ramping += 1

        # update current_obs.
        index=0
        for p in self.P:
            next_obs[len(self.P)+index] = self.D[f"{p}_{self.week_num+1}"]
            index +=1 


        next_obs[-2] = shipping
        next_obs[-1] = ramping
        self.current_obs = next_obs
        
        #update previous action
        self.previous_action = action


        # Compute episode done and truncated conditions.
        self.week_num += 1
        episode_done = False
        #episode_truncated = False
        if self.week_num >= len(self.T):
            episode_done = True
            self.week_num = 0


        #reward must be float.
        # info must be dictionary.
        return self.current_obs, reward, episode_done, episode_truncated, {'milp_reward':milp_reward, 'shipping_violated':shipping, 'ramping_violated':ramping, 'inventory':self.current_obs, 'unmet': unmet_demand_units} #'ramping_violated':ramping}


    
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