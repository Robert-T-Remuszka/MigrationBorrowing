'''
This file contains the model parameters class. 
'''
import numpy as np

class Parameters:
    def __init__(self,
                 T,         # length of life (int)
                 R,         # number of regions (int)
                 tau,       # consumption weight (float)
                 eta,       # nontrables weight (float)
                 chi,       # location preference shape parameter (float)
                 alpha_r,   # location preference scale parameter (array)
                 beta,      # discount factor (float)
                 z_grid,  # grid for productivity shocks (array)
                 age_eff,   # age effect on productivity (array)
                 Pi,        # productivity transition matrix (array)
                 theta,     # education augementation parameter (float)
                 Delta,     # symmetric moving cost matrix (array)
                 b_grid,    # grid for assets (array)
    ):
        self.T = T
        self.R = R
        self.tau = tau
        self.eta = eta
        self.chi = chi
        self.alpha_r = alpha_r
        self.beta = beta
        self.z_grid = z_grid
        self.age_eff = age_eff
        self.Pi = Pi
        self.theta = theta
        self.Delta = Delta
        self.b_grid = b_grid
        self.b_min = np.min(b_grid)
        self.b_max = np.max(b_grid)
        self.nb = len(b_grid)
        self.nz = len(z_grid)
        self.x_grid = np.array([0, 1])
        self.nx = len(self.x_grid)
    
    def U(self, I, P_c, P_n, P_h, eps=1e-4):
        denom = (P_c ** self.tau) * (P_n ** self.eta) * (P_h ** (1 - self.tau - self.eta))
        x = I / denom
        if x < eps: # use a Taylor expansion around epsilon to avoid log(0)
            return np.log(eps) + (x - eps) / eps
        else:
            return np.log(x)
        
    def extr_val_prob(self, v_flat, r=None):
        if r is None:
            return np.exp(self.chi * v_flat) / np.sum(np.exp(self.chi * v_flat))
        else:
            return np.exp(self.chi * v_flat[r]) / np.sum(np.exp(self.chi * v_flat))
    
    def calc_EV_A(self, v_flat):
        return (np.euler_gamma + np.log(np.sum(np.exp(self.chi * v_flat)))) / self.chi 
