import os
import json
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from environments.es_env import EnergySavingEnv
import pickle
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from pathlib import Path
import shutil



def initialize_norm_stats(stats_pickle = "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/explore/data-engineering/data_stats/normalization_stats.pkl"):
    with open(stats_pickle,"rb") as f:
        data_stats = pickle.load(f)
    
    return data_stats

data_stats = initialize_norm_stats()
obs_mean = data_stats["obs_mean"]
obs_std = data_stats["obs_std"]
next_obs_mean = data_stats["next_obs_mean"]
next_obs_std = data_stats["next_obs_std"]
reward_mean = data_stats["reward_mean"]
reward_std = data_stats["reward_std"]

config_paths = [
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_0_0_0.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_1_0_0.json",    
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_2_0_0.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_3_0_0.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_0_0_1.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_1_0_1.json",    
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_2_0_1.json",
    "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_eval_202/eval_202_500_3_0_1.json",
]

ns3_path = "/home/gagluba/ns-3-mmwave-oran/"
output_folder = "/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/output"
optimized = True

num_envs = 8


def load_memmap_directories(basepath):
    memmap_dirs = []
    for root, dirs, files in os.walk(basepath):
        for dir in dirs:
            memmap_dirs.append(os.path.join(root, dir))
    return memmap_dirs



def initialize_replay_buffer():
    raw_dir = Path("/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/explore/data-engineering/raw")
    memmap_dirs = load_memmap_directories(raw_dir/"0") + \
        load_memmap_directories(raw_dir/"2") 

    #print(memmap_dirs)
    loaded_data = [TensorDict.load_memmap(d) for d in memmap_dirs]
    #print(loaded_data)

    replay_buffer_dir = Path("/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/explore/data-engineering/replay_buffer")
    if os.path.exists(replay_buffer_dir):
        shutil.rmtree(replay_buffer_dir)

    replay_buffer = ReplayBuffer(
        storage=LazyMemmapStorage(1000000, scratch_dir=replay_buffer_dir),
    )
    for data in loaded_data:
        try:
            data['observation'] = (data['observation'] - obs_mean) / obs_std
            data['next_observation'] = (data['next_observation'] - next_obs_mean) / next_obs_std
            data['reward'] = (data['reward'] - reward_mean) / reward_std        
            replay_buffer.extend(data)
        except Exception as e:
            # print(f"Error extending replay buffer with data from {data}: {e}")
            continue

    return replay_buffer




class MemmapRLDataset(Dataset):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.length = len(replay_buffer)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.replay_buffer[idx]
        # Return as tuple for DataLoader compatibility
        return (
            sample["observation"],
            sample["action"],
            sample["reward"],
            sample["next_observation"],
            sample["done"]
        )



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
            self.model.save(f"{self.save_path}/dqn_step_{self.num_timesteps}")
        return True


class StepLoggerCallback(BaseCallback):
    def __init__(self, log_freq=1, verbose=0):
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
        train_loss = self.model.logger.name_to_value.get("train/loss", None)


        print(self.model.logger.name_to_value.keys())
        print(self.model.logger.name_to_value)
        print(train_loss)
        self.logger.record("rollout/loss", train_loss)
        print("++++++++++++++++++++++++++")        
        
        self.logger.dump(self.num_timesteps)
        return True

    # def _on_rollout_end(self):
    #     """Executed after each rollout ends"""
    #     approx_kl = self.model.logger.name_to_value.get("train/approx_kl", None)
    #     clip_fraction = self.model.logger.name_to_value.get("train/clip_fraction", None)
    #     clip_range = self.model.logger.name_to_value.get("train/clip_range", None)
    #     entropy_loss = self.model.logger.name_to_value.get("train/entropy_loss", None)
    #     explained_variance = self.model.logger.name_to_value.get("train/explained_variance", None)
    #     loss = self.model.logger.name_to_value.get("train/loss", None)
    #     n_updates = self.model.logger.name_to_value.get("train/n_updates", None)
    #     policy_gradient_loss = self.model.logger.name_to_value.get("train/policy_gradient_loss", None)
    #     value_loss = self.model.logger.name_to_value.get("train/value_loss", None)

    #     print(self.model.logger.name_to_value.keys())
    #     print(self.model.logger.name_to_value)


    #     print(f"\n--- Rollout Completed ---")
    #     print(f"Approx KL: {approx_kl}")
    #     print(f"Clip Fraction: {clip_fraction}")
    #     print(f"Clip Range: {clip_range}")
    #     print(f"Entropy Loss: {entropy_loss}")
    #     print(f"Explained Variance: {explained_variance}")
    #     print(f"Loss: {loss}")
    #     print(f"Number of Updates: {n_updates}")
    #     print(f"Policy Gradient Loss: {policy_gradient_loss}")
    #     print(f"Value Loss: {value_loss}")
    #     print("-------------------------")


    #     # Log to Tensorboard
    #     self.logger.record("rollout/approx_kl", approx_kl)
    #     self.logger.record("rollout/clip_fraction", clip_fraction)
    #     self.logger.record("rollout/clip_range", clip_range)
    #     self.logger.record("rollout/entropy_loss", entropy_loss)
    #     self.logger.record("rollout/explained_variance", explained_variance)
    #     self.logger.record("rollout/loss", loss)
    #     self.logger.record("rollout/n_updates", n_updates)
    #     self.logger.record("rollout/policy_gradient_loss", policy_gradient_loss)
    #     self.logger.record("rollout/value_loss", value_loss)
    #     # Optionally, you can also log to stdout or a file
    #     return True  # Continue training

    def _on_rollout_end(self):
        # ... your existing code ...
        # After recording all metrics:
        self.logger.dump(self.num_timesteps)  # <-- This is required for TensorBoard!
        return True

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

def binary_array_to_discrete(binary_array):
    """
    Convert a binary array (e.g., [1,0,1,1,0,0,1]) to a discrete integer.
    """
    return int("".join(str(int(b)) for b in binary_array), 2)


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(61,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(128)
    def reset(self, *args, **kwargs):
        return np.zeros(61), {}
    def step(self, action):
        return np.zeros(61), 0.0, True, False, {}


TRAINING_MODE = "online"
if __name__ == "__main__":
    # Initialize replay buffers
    match TRAINING_MODE:
        case "offline":
            replay_buffer = initialize_replay_buffer()
            dataset = MemmapRLDataset(replay_buffer)
            # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

            # Define spaces
            # env_fns = [make_env(obs_mean, obs_std, reward_mean, reward_std) for i in range(1)]
            env = DummyVecEnv([lambda: DummyEnv()])

            model = DQN(
                "MlpPolicy", 
                env, 
                buffer_size=1000000, 
                learning_starts=1000, 
                verbose=1
            )
            model.replay_buffer.handle_timeout_termination = False
            for obs, action, reward, next_obs, done in dataset:
                print(
                    f"Obs: {obs.shape}, Action: {action.shape}, Reward: {reward.shape}, Next Obs: {next_obs.shape}, Done: {done.shape}"
                )
                action =  binary_array_to_discrete(action)
                action = torch.tensor(action, dtype=torch.int64)
                model.replay_buffer.add(obs,next_obs, action, reward, done, {})

            new_logger = configure("./dqn-11-logs", ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)

            save_callback = SaveEveryNTimestepsCallback(save_freq=100, save_path="./dqn-11-checkpoints")
            model.learn(total_timesteps=100000, callback=save_callback)
        case "online":
            save_callback = SaveEveryNTimestepsCallback(save_freq=10, save_path="./dqn-11-eval-checkpoints")
            # tensorboard_callback = TensorboardLoggingCallback(log_freq=1)
            env_fns = [make_env(config_paths[i], ns3_path, output_folder, optimized, obs_mean, obs_std, reward_mean, reward_std) for i in range(num_envs)]
            vec_env = SubprocVecEnv(env_fns)

            # Load model from checkpoint 
            log_dir = "./dqn-11-eval-tensorboard"
            model = DQN.load(
                "./dqn-10-eval-checkpoints/dqn_step_840", 
                env=vec_env, 
                tensorboard_log=log_dir, 
                batch_size=32, 
                learning_rate=3e-4, 
                learning_starts=1,
                train_freq=1,
                verbose=9
            )
            new_logger = configure("./dqn-11-eval-logs", ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)

            # print(model.learning_starts)
            # print(model.batch_size)
            # print(model.gradient_steps)

            # model = PPO("MlpPolicy", vec_env, verbose=1)
            model.learn(total_timesteps=100_000, callback=[save_callback, StepLoggerCallback()])