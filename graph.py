# importing packages
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder_path = "../datasets/spec2006-train/astar/astar-big/astar_astar-big.csv"

# apply the default theme
sns.set_theme()
sns.set(font_scale=1.45)
plt.rcParams['axes.labelsize'] = 22

# loading dataset
data = pd.read_csv(f"{folder_path}")
data = data.rename(columns={'llc_accesses': "llc accesses", 'instr': "instructions retired"})
data = data.drop(['llc_misses', 'cycles', 'bus_accesses', 'cluster', 'phase'], axis=1)
data = data.apply(filter_data)

# draw plotline
ax = sns.lineplot(data=data)
ax.set(xlabel='Epoch', ylabel='Counter Value')
fig = ax.get_figure()
fig.set_size_inches(11.7, 8.27)
fig.savefig("astar-finegrained-filtered.png", dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()