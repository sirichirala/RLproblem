#using stable_baselines3 for RL algorithms
import os
import csv
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, SAC

class RLAlgorithms():

    def __init__(self, environment, cfg):
        self.P = environment.P #Number of products.
        self.T = environment.T # Number of time steps per episode.
        self.N = int(10*len(self.T)) # Number of episodes used for training rl algorithms.
        self.environment = environment # SCM Environment.
        #self.optimal_actions = [] # Place holder optimal actions.
        self.result_path = cfg.result_path # Results path.


    def checkenv(self):

        return check_env(self.environment)

    def checkfeasibility(self):

        pass

    def A2C_algorithm(self):
        #Train the model
        model = A2C("MlpPolicy", self.environment)
        model.learn(total_timesteps=self.N, progress_bar=True)

        # Test the model for one episode
        obs, info = self.environment.reset()
        episode_done = False
        optimal_actions = []

        while not episode_done:
            # Predict the action using the trained model
            action, _states = model.predict(obs)
            optimal_actions.append(np.sum(action, axis=0).tolist())

            # Take the action in the environment
            obs, reward, episode_done, episode_truncated, info = self.environment.step(action)

        # Write optimal action results to file.
        self.write_to_csv(reward, optimal_actions)

        return optimal_actions

    
    def PPO_algorithm(self):
        #Train the model
        model = PPO("MlpPolicy", self.environment)
        model.learn(total_timesteps=self.N, progress_bar=True)

        # Test the model for one episode
        obs, info = self.environment.reset()
        episode_done = False
        optimal_actions = []

        while not episode_done:
            # Predict the action using the trained model
            action, _states = model.predict(obs)
            optimal_actions.append(np.sum(action, axis=0).tolist())

            # Take the action in the environment
            obs, reward, episode_done, episode_truncated, info = self.environment.step(action)

        # Write optimal action results to file.
        self.write_to_csv(reward, optimal_actions)

        return optimal_actions


    def write_to_csv(self, reward, optimal_actions):
        """
        1. create a results.csv file in the results_milp folder which has the total reward collected.
        2. creata a separate result_instance_# file for each instance with the actions.
        """
        
        result_file = os.path.join(self.result_path, 'results.csv')
        result_data = [reward]
        
        file_exists = os.path.isfile(result_file)

        # Writing reward
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            # Write header if the file doesn't exist
            if not file_exists:
                writer.writerow(["Instance name", "Reward"])
        file.close()
        with open(result_file, "r") as file:
            no_lines = sum(1 for _ in file)
        # Write result data
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([no_lines] + result_data)
        

        # Writing actions
        result_instance_file = os.path.join(self.result_path, 'result_instance_' + str(no_lines) + '.csv')
        with open(result_instance_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time period"] + self.P)
            for t in self.T:
                row_values = [t]
                for p in range(len(self.P)):
                    row_values.append(optimal_actions[t][p])
                writer.writerow(row_values)
