#%%
import numpy as np
import scipy.interpolate as spi
import scipy.optimize as spo


#%%

from parameters import *
from households import *


#%%

p = Parameters(
    T=3,
    R=2,
    tau=0.5,
    eta=.2,
    chi=1,
    beta=.9,
    z_grid=np.array([.3, 1.0]),
    age_eff=np.ones(3),
    Pi=np.array([[.9, .1], [.1, .9]]),
    theta=.1,
    Delta=np.array([[.8, 1], [1, .8]]),
    b_grid=np.linspace(0, 20, 50)
)


# %%

P_c = np.array([1, 1])
P_n = np.array([1, 1])
P_h = np.array([1, 1])
w = np.array([1, 1])
i = .05
f = .5

v, b_policy, x_policy, r_probs, EV = solve_household(p, P_c, P_n, P_h, w, i, f)


# %%

import matplotlib.pyplot as plt

# %%

plt.plot(p.b_grid, v[-1, 0, 0, 0, 0, :])

# %%
