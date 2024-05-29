# Goal: implement a function that takes a temperature profile computed by a controller and converts it to

#%%
import sim.tfunc as tfunc
import sympy
# %%
domain = [[-1, 1], [0, sympy.pi]]
max_fluctuation = 0.75
tfunc = tfunc.Tfunc(10, domain, max_fluctuation)
# %%
# play around with tfunc to understand what it does
tfunc.apply_T()