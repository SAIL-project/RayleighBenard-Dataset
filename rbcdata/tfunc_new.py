# Goal: implement a function that takes a temperature profile computed by a controller and converts it to

#%%
import sim.tfunc as tfunc
import sympy
import importlib
import numpy as np
from sympy.abc import y
from spb import plot_piecewise
import matplotlib.pyplot as plt
%matplotlib qt
# %%
domain = [[-1, 1], [0, 2*sympy.pi]]
max_fluctuation = 0.75
nr_segments = 10
tfunc_in = tfunc.Tfunc(nr_segments, domain, max_fluctuation)
# %%
# play around with tfunc to understand what it does
# action_agent = np.linspace(-1, 1, 10)
action_agent = np.zeros(10)
action_agent[0:5] = -1 
action_agent[5:] = 1
# action_agent = 500 * np.concatenate([np.ones(nr_segments // 2) * -1, np.ones(nr_segments // 2) * 1])
# action_agent = np.random.rand(nr_segments) *  
action_effective = tfunc_in.apply_T(action_agent, y, [3, 1])
fig, ax = plt.subplots()
plot_piecewise(action_effective, (y, float(domain[1][0]), float(domain[1][1])), ax=ax)
# ax.set_xlim((float(domain[1][0]), float(domain[1][1])))
plt.show()
# tfunc.apply_T()
# %%
