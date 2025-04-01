import gymnasium as gym
import sys
import os
local_panda_gym_path = os.path.abspath("C:/NTU/panda-gym")  # Replace with your actual path
sys.path.insert(0, local_panda_gym_path)
import panda_gym

env = gym.make("PandaReach-v3", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
