import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./logs/progress.csv")
# df = df[df["time/total_timesteps"]<10000]

# Extract relevant data
total_timesteps = df["time/total_timesteps"]
success_rate = df["rollout/success_rate"]
ep_rew_mean = df["rollout/ep_rew_mean"]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(total_timesteps, success_rate, marker="o", linestyle="-", markersize=3)
plt.xlabel("Total Timesteps")
plt.ylabel("Success Rate")
plt.title("Success Rate over Total Timesteps")
plt.grid(True)
plt.ylim(0, 1.1)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(total_timesteps, ep_rew_mean, marker="o", linestyle="-", markersize=3, color="r")
plt.xlabel("Total Timesteps")
plt.ylabel("Episode Reward Mean")
plt.title("Episode Reward Mean over Total Timesteps")
plt.grid(True)
plt.show()