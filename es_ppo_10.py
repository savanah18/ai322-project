import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from environments.es_env import EnergySavingEnv
import pickle
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure

from datetime import datetime


def initialize_norm_stats(stats_pickle = "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/explore/data-engineering/data_stats/normalization_stats.pkl"):
    with open(stats_pickle,"rb") as f:
        data_stats = pickle.load(f)
        return data_stats['obs_mean'], data_stats['obs_std'], data_stats['reward_mean'], data_stats['reward_std']

obs_mean, obs_std, reward_mean, reward_std = initialize_norm_stats()
# data_stats
obs_mean,obs_mean

config_paths = [
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_0_0_0.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_1_0_0.json",    
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_2_0_0.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_3_0_0.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_0_0_1.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_1_0_1.json",    
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_2_0_1.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_110/eval_110_500_3_0_1.json",
]

ns3_path = "/home/gagluba/ns-3-mmwave-oran/"
output_folder = "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/output"
optimized = True

num_envs = 8



class NormalizedEnv(gym.Env):
    def __init__(self, env, obs_mean, obs_std, reward_mean, reward_std, hold_steps=1):
        super().__init__()
        self.env = env
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        # Expose standard gym attributes
        self.action_space = gym.spaces.Discrete(128)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(61,), dtype=np.float32)
        self.hold_steps = hold_steps
        self._hold_counter = 0
        self._last_action = None

    def unpack_action(self, action):
        width = 7
        binary_str = np.binary_repr(action, width=width)
        binary_array = np.array([int(b) for b in binary_str])
        return binary_array

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        obs = np.asarray(obs).squeeze() 

        # Reset the penalty decay
        self._hold_counter = 0
        self._last_action = None

        return (obs - self.obs_mean) / (self.obs_std + 1e-8) , info

    def step(self, action):
        # Only accept a new action every self.hold_steps steps
        if self._hold_counter == 0 or self._last_action is None:
            self._last_action = action
        self._hold_counter = (self._hold_counter + 1) % self.hold_steps

        unpacked_action = self.unpack_action(self._last_action)
        obs, reward, terminated, truncated, info = self.env.step(unpacked_action)
        obs = np.asarray(obs).squeeze()

        print("++++++++++++++++++++++++++")
        print("Action (possibly held): ", self._last_action)
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        print("Truncated: ", truncated)
        print("Info: ", info)
        print("++++++++++++++++++++++++++")

        return (obs - self.obs_mean) / (self.obs_std + 1e-8), (reward - self.reward_mean)/(reward_std + 1e-8), terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
    
class SaveEveryNTimestepsCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(f"{self.save_path}/ppo_step_{self.num_timesteps}")
        return True


class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            # Example: log episode reward and length if available
            if len(self.locals.get("infos", [])) > 0:
                info = self.locals["infos"][-1]
                if "episode" in info:
                    ep_info = info["episode"]
                    self.logger.record("custom/episode_reward", ep_info.get("r", 0))
                    self.logger.record("custom/episode_length", ep_info.get("l", 0))

            # Log loss if available
            if "loss" in self.locals:
                self.logger.record("custom/loss", self.locals["loss"])
            elif hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                # Try to get loss from logger if available
                loss = self.model.logger.name_to_value.get("train/loss", None)
                if loss is not None:
                    self.logger.record("custom/loss", loss)

        return True

class StepLoggerCallback(BaseCallback):
    def __init__(self, log_freq=32, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        print("++++++++++++++++++++++++++")
        reward = self.locals["rewards"]
        action = self.locals["actions"]
        #observation = self.locals["obs"]
        # Log step-wise metrics
        self.logger.record("step/reward", reward)
        self.logger.record("step/action", action)
        print("++++++++++++++++++++++++++")        

        return True

    def _on_rollout_end(self):
        """Executed after each rollout ends"""
        approx_kl = self.model.logger.name_to_value.get("train/approx_kl", None)
        clip_fraction = self.model.logger.name_to_value.get("train/clip_fraction", None)
        clip_range = self.model.logger.name_to_value.get("train/clip_range", None)
        entropy_loss = self.model.logger.name_to_value.get("train/entropy_loss", None)
        explained_variance = self.model.logger.name_to_value.get("train/explained_variance", None)
        loss = self.model.logger.name_to_value.get("train/loss", None)
        n_updates = self.model.logger.name_to_value.get("train/n_updates", None)
        policy_gradient_loss = self.model.logger.name_to_value.get("train/policy_gradient_loss", None)
        value_loss = self.model.logger.name_to_value.get("train/value_loss", None)

        print(self.model.logger.name_to_value.keys())
        print(self.model.logger.name_to_value)


        print(f"\n--- Rollout Completed ---")
        print(f"Approx KL: {approx_kl}")
        print(f"Clip Fraction: {clip_fraction}")
        print(f"Clip Range: {clip_range}")
        print(f"Entropy Loss: {entropy_loss}")
        print(f"Explained Variance: {explained_variance}")
        print(f"Loss: {loss}")
        print(f"Number of Updates: {n_updates}")
        print(f"Policy Gradient Loss: {policy_gradient_loss}")
        print(f"Value Loss: {value_loss}")
        print("-------------------------")


        # Log to Tensorboard
        self.logger.record("rollout/approx_kl", approx_kl)
        self.logger.record("rollout/clip_fraction", clip_fraction)
        self.logger.record("rollout/clip_range", clip_range)
        self.logger.record("rollout/entropy_loss", entropy_loss)
        self.logger.record("rollout/explained_variance", explained_variance)
        self.logger.record("rollout/loss", loss)
        self.logger.record("rollout/n_updates", n_updates)
        self.logger.record("rollout/policy_gradient_loss", policy_gradient_loss)
        self.logger.record("rollout/value_loss", value_loss)
        # Optionally, you can also log to stdout or a file
        return True  # Continue training

def make_env(config_path, ns3_path, output_folder, optimized, obs_mean, obs_std, reward_mean, reward_std):
    def _init():
        with open(config_path) as f:
            scenario_configuration = json.load(f)
        env = EnergySavingEnv(
            ns3_path=ns3_path,
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized
        )
        env = NormalizedEnv(env, obs_mean, obs_std, reward_mean, reward_std)
        return env
    return _init


if __name__ == "__main__":
    save_callback = SaveEveryNTimestepsCallback(save_freq=10, save_path="./ppo-12-checkpoints")
    # tensorboard_callback = TensorboardLoggingCallback(log_freq=1)
    env_fns = [make_env(config_paths[i], ns3_path, output_folder, optimized, obs_mean, obs_std, reward_mean, reward_std) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Load model from checkpoint 
    log_dir = "./ppo-12-tensorboard"
    model = PPO.load(
        "./ppo-10-checkpoints/ppo_step_3240", 
        env=vec_env, 
        tensorboard_log=log_dir, 
        batch_size=32, 
        learning_rate=3e-4, 
        n_steps=32, 
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    new_logger = configure("./ppo-11-logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)


    # model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100_000, callback=[save_callback, StepLoggerCallback()])

