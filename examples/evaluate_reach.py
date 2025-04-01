import gymnasium as gym
from stable_baselines3 import DDPG, SAC
import sys
import os
import time
# local_panda_gym_path = os.path.abspath("C:/NTU/panda-gym")
# sys.path.insert(0, local_panda_gym_path)
import panda_gym
import numpy as np
from gymnasium.wrappers import RecordVideo

def evaluate_success_rate(model, env, num_episodes=100):
    success_count = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if "is_success" in info and info["is_success"]:
                success_count += 1
                break
    
    success_rate = success_count / num_episodes
    return success_rate

# Load trained model
env = gym.make("PandaReach-v3")
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda episode_id: True)
model = SAC.load("./models/sac_default_without_normalization_20_000", env=env)  # Ensure you have saved your model with model.save("ddpg_panda_reach")

success_rate = evaluate_success_rate(model, env)
print(f"Success rate over {100} episodes: {success_rate * 100:.2f}%")

env.close()