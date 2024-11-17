import pandas as pd
from matplotlib import pyplot as plt

# baseline
df_baseline = pd.read_csv("res/ra10000_baseline.csv")
filter_col = [col for col in df_baseline if col.endswith("nusselt")]
filter_col.append("sim_time")
df_baseline = df_baseline[filter_col].set_index("sim_time")

mean_baseline = df_baseline.mean(axis=1)
std_baseline = df_baseline.std(axis=1)

# pd
df_pd = pd.read_csv("res/ra10000_pd.csv")
filter_col = [col for col in df_pd if col.endswith("nusselt")]
filter_col.append("sim_time")
df_pd = df_pd[filter_col].set_index("sim_time")

mean_pd = df_pd.mean(axis=1)
std_pd = df_pd.std(axis=1)

fig, ax = plt.subplots(figsize=(5, 3))
# plot baseline
mean_baseline.plot(ax=ax, label="Baseline")
mean = mean_baseline.to_numpy()
std = std_baseline.to_numpy()
ax.fill_between(
    range(1, 50),
    mean + std,
    mean - std,
    facecolor="blue",
    alpha=0.1,
)

# plot pd
mean_pd.plot(ax=ax, label="PD")
mean = mean_pd.to_numpy()
std = std_pd.to_numpy()
ax.fill_between(
    range(1, 50),
    mean + std,
    mean - std,
    facecolor="orange",
    alpha=0.1,
)

# show
ax.set_ylim(2, 3)
ax.set_ylabel("Nusselt number")
ax.set_xlabel("Time")
ax.legend()
plt.tight_layout()
plt.show()
