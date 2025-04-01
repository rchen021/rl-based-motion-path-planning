import gymnasium as gym
import panda_gym


env = gym.make("PandaReach-v3")
print(env.spec.max_episode_steps)
print(env._max_episode_steps)
print("Time limit before truncation:", env.spec.max_episode_steps)