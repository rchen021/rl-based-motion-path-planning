import gymnasium as gym
from stable_baselines3 import DDPG, SAC
import sys
import os
import time
from PIL import Image
import numpy as np
local_panda_gym_path = os.path.abspath("C:/NTU/panda-gym")  # Replace with your actual path
sys.path.insert(0, local_panda_gym_path)
import panda_gym
from gymnasium.wrappers import RecordVideo

def save_frame_as_image(frame, episode_id, step_id):
    image_path = f"./images/episode_{episode_id}_step_{step_id}.png"
    image = Image.fromarray(frame)
    image.save(image_path)
    print(f"Saved frame as image: {image_path}")

def evaluate_success_rate(model, env, num_episodes=15):
    success_count = 0
    
    for episode_id in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_id = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Capture and save the frame as an image
            frame = env.render()  # Render the current frame as RGB array
            save_frame_as_image(frame, episode_id, step_id)  # Save the image
            step_id += 1
            
            if "is_success" in info and info["is_success"]:
                success_count += 1
                break
    
    success_rate = success_count / num_episodes
    return success_rate

# Load trained model
env = gym.make("PandaReach-v3")
model = DDPG.load("./models/td3_wo_normalized_customized_dense_reward5_2obj", env=env)

# Create a folder to save images if it doesn't exist
if not os.path.exists('./images'):
    os.makedirs('./images')

success_rate = evaluate_success_rate(model, env)
print(f"Success rate over 15 episodes: {success_rate * 100:.2f}%")

env.close()

