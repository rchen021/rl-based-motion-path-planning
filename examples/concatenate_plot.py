import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Folder containing the CSV files
folder_path = ''


# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Create a color map to get different colors for each line
colors = plt.cm.get_cmap('tab10', len(csv_files))

# Initialize plot
plt.figure(figsize=(8, 5))

# Loop through each CSV file and plot its data
for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(os.path.join(folder_path, csv_file))

    df = df[df["time/total_timesteps"] <= 10000]
    
    # Extract relevant data from the current CSV
    total_timesteps = df["time/total_timesteps"]
    # success_rate = df["rollout/success_rate"]
    mean_reward = df["rollout/ep_rew_mean"]

    label = csv_file.replace('.csv', '')

    # Plot with a different color for each file
    plt.plot(total_timesteps, mean_reward, linestyle="-", markersize=3, color=colors(i), label=label)

# Customize plot
plt.xlabel("Total Timesteps")
plt.ylabel("Mean Reward")
# plt.title("")
# Performance Comparison of Various Reward Functions
# DDPG Success Rate Across Different Hyperparameters
plt.grid(True)
plt.ylim(0, 1.1)

# Add legend with CSV filenames
plt.legend()

# Show plot
plt.show()
