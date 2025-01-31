#%%
import numpy as np
import scipy.interpolate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
plt.rcParams['figure.dpi'] = 600

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
    beta=.72,
    z_grid=np.array([.3, 1.0]),
    age_eff=np.ones(3),
    Pi=np.array([[.9, .1], [.1, .9]]),
    theta=.1,
    Delta=np.array([[.8, 1], [1, .8]]),
    b_grid=np.linspace(1.5, 20, 100)
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
'''
Look at the saving policy functions for educated and non-educated 
'''
plt.plot(p.b_grid, b_policy[-2, 0, 0, 0, 0, :], color = (0/255, 147/255, 245/255), label='No College')
plt.plot(p.b_grid, b_policy[-2, 0, 0, 0, 1, :], color = (247/255, 129/255, 4/255),label='College')
plt.xlabel(r'$k$')
plt.ylabel(r'$\hat{k}$')
plt.xticks(fontsize = 9)
plt.legend()
sns.despine()
plt.savefig('../output/graphs/savingpolicy.pdf', dpi=600, transparent=True,
            bbox_inches = 'tight')

# %%
'''
Look at the out-migration policies
'''
plt.plot(p.b_grid, r_probs[-2, 0, 1, 0, 0, :], color = (0/255, 147/255, 245/255), label='No College')
plt.plot(p.b_grid, r_probs[-2, 0, 1, 0, 1, :], color = (247/255, 129/255, 4/255),label='College')
plt.xlabel(r'$k$')
plt.ylabel(r'$\gamma_{g}^{jr}$')
plt.legend()
sns.despine()
plt.savefig('../output/graphs/outmigration.pdf', dpi = 600, transparent= True,
            bbox_inches='tight')
