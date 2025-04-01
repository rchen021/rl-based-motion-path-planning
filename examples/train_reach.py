import gymnasium as gym
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
import sys
import os
local_panda_gym_path = os.path.abspath("C:/NTU/panda-gym")  # Replace with your actual path
sys.path.insert(0, local_panda_gym_path)
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


env_id = "PandaReach-v3"
env = gym.make(env_id, render_mode='rgb_array') 
env = Monitor(env)  # Wrap the env to log episode stats

# env = make_vec_env(env_id, n_envs=1)
# env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Logging directory
log_dir = "./logs/"
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True) 

new_logger = configure(log_dir, ["csv"])  # Save logs as CSV

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG(policy="MultiInputPolicy", env=env, verbose=1, device="cuda",
            action_noise=action_noise,
            replay_buffer_class=HerReplayBuffer)

model.set_logger(new_logger)

model.learn(total_timesteps=8000)

model_path = os.path.join(model_dir, "ddpg_wo_normalized_customized_dense_reward10_HRB_AN_2obj42")
model.save(model_path)

print(f"Logs saved in {log_dir}")
print(f"Model saved in {model_dir}")