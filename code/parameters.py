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
                 z_grid,    # grid for productivity (array)
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
        self.Pi = Pi
        self.theta = theta
        self.Delta = Delta
        self.b_grid = b_grid
        self.b_min = np.min(b_grid)
        self.b_max = np.max(b_grid)
        self.nb = len(b_grid)
        self.nz = len(z_grid)





                 
