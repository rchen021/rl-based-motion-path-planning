import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3 import SAC, TD3
import sys
import os
import time
import panda_gym

def evaluate_success_rate_and_record(model, env, video_path, num_episodes=100, fps=30):
    success_count = 0

    # Access the base environment
    raw_env = env.unwrapped  

    # Get frame size
    frame = raw_env.render()
    frame_size = (frame.shape[1], frame.shape[0])

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            frame = raw_env.render()  # Render frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR
            video_writer.write(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if "is_success" in info and info["is_success"]:
                success_count += 1
                break

    video_writer.release()  # Save video
    success_rate = success_count / num_episodes
    return success_rate

# Load trained model
env = gym.make("PandaReach-v3")  
model = TD3.load("./models/td3_wo_normalized_customized_dense_reward5_2obj", env=env)
video_path="td3_2obj_output43.mp4"

success_rate = evaluate_success_rate_and_record(model, env, video_path)
print(f"Success rate over {100} episodes: {success_rate * 100:.2f}%")

env.close()
