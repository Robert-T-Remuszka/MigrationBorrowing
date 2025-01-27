import numpy as np


def solve_household(p, P_c, P_n, P_h, w, i):
    '''
    This function solves the household problem.
    
    Args:
        p (Parameters): model parameters
        P_C (array): consumption prices
        P_N (array): non-tradable prices
        P_h (array): housing prices
        w (array): wages
        i (float): interest rate
    
    Returns:
        V (array): value function
        psi (array): policy function
    '''

    V = np.zeros(p.T, p.R, p.nb, p.nz, p.nx)
    b_policy = np.zeros(p.T, p.R, p.nb, p.nz, p.nx)
    r_probs = np.zeros(p.T, p.R, p.R, p.nb, p.nz, p.nx)

    # start at the end of life - assume V_{t+1} = 0
    I = np.zeros(p.R, p.R, p.nb, p.nz, p.nx)
    v = np.zeros(p.R, p.R, p.nb, p.nz, p.nx)
    for j in range(p.R):
        for r in range(p.R):
            for (b_index, b) in enumerate(p.b_grid):
                for (z_index, z) in enumerate(p.z_grid):
                    for x in p.x_grid: # uses that x = 0 or 1 (x index == x)
                        I[j, r, b_index, z_index, x] = w[r] * np.exp(p.theta * x) * z * p.Delta[j, r] + b * (1 + i)
                        v[j, r, b_index, z_index, x] = p.U(I[j, r, b_index, z_index, x], P_c[r], P_n[r], P_h[r])

    for j in range(p.R):
        for r in range(p.R):
            for (b_index, b) in enumerate(p.b_grid):
                for (z_index, z) in enumerate(p.z_grid):
                    for x in p.x_grid: # uses that x = 0 or 1 (x index == x)
                        r_probs[-1, j, r, b_index, z_index, x] = np.exp(v[j, r, b_index, z_index, x] / np.sum(v[j, :, b_index, z_index, x]))
