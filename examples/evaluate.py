import gymnasium as gym
import numpy as np
from numpngw import write_apng
from IPython.display import Image, display
from stable_baselines3 import DDPG, SAC, PPO, TD3
from customReach import MyRobotTaskEnv

# Load trained model
# model = DDPG.load("DDPG_custom_panda_reach")
model = SAC.load("SAC_custom_panda_reach2")
# model = TD3.load("TD3_custom_panda_reach")

# Number of obstacle instances
num_trials = 10
successful_cases = 0
all_frames = []  # Store frames from all obstacle instances

for i in range(num_trials):
    print(f"Running instance {i + 1} / {num_trials}")

    env = MyRobotTaskEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []  # Store frames for this instance

    done = False
    max_steps = 100
    step_count = 0

    while not done and step_count < max_steps:
        action, _ = model.predict(obs)
        step_result = env.step(action)

        # Handle new Gym API (5 values instead of 4)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result
            done = done or truncated  # Stop if either is True

        # Check if the agent succeeded (modify this based on your environment)
        if "is_success" in info and info["is_success"]:
            successful_cases += 1
            print(f"Success in instance {i + 1}!")

        # Render and store frame
        frame = env.render()
        frames.append(frame)

        step_count += 1

    env.close()  # Close environment
    all_frames.extend(frames)  # Store frames from this instance

# Calculate success rate
success_rate = (successful_cases / num_trials) * 100
print(f"\nTotal Successful Cases: {successful_cases} / {num_trials}")
print(f"Success Rate: {success_rate:.2f}%")