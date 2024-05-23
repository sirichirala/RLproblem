#using stable_baselines3 for RL algorithms
import os
import csv
import numpy as np

from rl_zoo3 import linear_schedule

from scm_env import ScmEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, SAC

import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 1000 episodes
              mean_reward = np.mean(y[-10000:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class RLAlgorithms():

    def __init__(self, environment, cfg):
        self.P = environment.P #Number of products.
        self.T = environment.T # Number of time steps per episode.
        self.N = int(4000000*len(self.T)) # Number of episodes used for training rl algorithms.
        self.environment = environment # SCM Environment.
        self.algo_name = PPO
        self.model_path = cfg.model_path #RL model path.
        self.model = None
        self.result_path = cfg.result_path # Results path.


    def checkenv(self):

        return check_env(self.environment)

    def algorithm_train(self):
        self.environment = Monitor(self.environment, self.model_path)
        self.model = self.algo_name("MlpPolicy", self.environment, ent_coef=0.01, verbose=0, tensorboard_log=self.model_path) #policy_kwargs=dict(net_arch=[256, 256]), learning_rate=0.03,
        callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.model_path)
        self.model.learn(total_timesteps=self.N, progress_bar=True, callback=callback)

        plot_results([self.model_path], self.N, results_plotter.X_TIMESTEPS, "ScmEnv")
        plt.savefig(str(self.model_path)+"\\training_reward_plot.png")
        plt.close()

        return
    

    def algorithm_predict(self):
        # Load the saved model
        self.model = self.algo_name.load(os.path.join(self.model_path, "best_model.zip"))
        # Test the model for one episode
        obs, info = self.environment.prediction_reset()
        episode_done = False
        optimal_actions = []
        shipping_violated = []
        ramping_violated = []
        milp_episode_reward = 0
        rl_episode_reward = 0

        while not episode_done:
            # Predict the action using the trained model
            action, _states = self.model.predict(obs, deterministic=True)
            action = np.floor(action)
            optimal_actions.append(action.tolist())

            # Take the action in the environment
            obs, reward, episode_done, episode_truncated, info = self.environment.step(action)
            rl_episode_reward += reward
            milp_episode_reward += info['milp_reward']
            shipping_violated.append(info['shipping_violated'])
            ramping_violated.append(info['ramping_violated'])
            #if shipping_violated[-1] == 0:
            #    rl_episode_reward -= 0
            
            #inv = 0 
            #for p in range(0,5):
            #    inv += info['inventory'][p]
            #print("inv: " + str(inv))
            #print(sum(info['unmet']))
            
        # Write optimal action results to file.
        self.write_to_csv(rl_episode_reward, milp_episode_reward, optimal_actions, shipping_violated, ramping_violated) #,ramping_violated)

        return optimal_actions


    def write_to_csv(self, reward, milp_reward, optimal_actions, shipping_violated, ramping_violated):
        """
        1. create a results.csv file in the results_milp folder which has the total reward collected.
        2. creata a separate result_instance_# file for each instance with the actions.
        """
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        result_file = os.path.join(self.result_path, 'rl_results.csv')
        result_data = [reward, milp_reward] 
        
        file_exists = os.path.isfile(result_file)

        # Writing reward
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            # Write header if the file doesn't exist
            if not file_exists:
                writer.writerow(["Instance name", "RL Reward", "MILP Reward"])
        file.close()
        with open(result_file, "r") as file:
            no_lines = sum(1 for _ in file)
        # Write result data
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([no_lines] + result_data)
        

        # Writing actions
        result_instance_file = os.path.join(self.result_path, 'result_instance_action_' + str(no_lines) + '.csv')
        with open(result_instance_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time period"] + self.P + ["shipping"] + ["ramping"])
            for t in self.T:
                row_values = [t]
                for p in range(len(self.P)):
                    row_values.append(optimal_actions[t][p])
                row_values.append(shipping_violated[t])
                row_values.append(ramping_violated[t])
                writer.writerow(row_values)
