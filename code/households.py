import numpy as np
import scipy.interpolate as spi
import scipy.optimize as spo


def solve_household(p, P_c, P_n, P_h, w, i, f):
    '''
    This function solves the household problem.
    
    Args:
        p (Parameters): model parameters
        P_C (array): consumption prices
        P_N (array): non-tradable prices
        P_h (array): housing prices
        w (array): wages
        i (float): interest rate
        f (float): fixed cost of education
    
    Returns:
        V (array): value function
        b_policy (array): saving policy function
        x_policy (array): education policy function
        r_probs (array): location choice probabilities
    '''
    v_tilde = np.zeros((p.T, p.R, p.R, p.nz, p.nx, p.nx, p.nb))   # indirect utility given education and location choice
    v = np.zeros((p.T, p.R, p.R, p.nz, p.nx, p.nb))               # indirect utility given location choice
    EV = np.zeros((p.T, p.R, p.nz, p.nx, p.nb))                   # expected value over location choice
    b_policy_tilde = np.zeros_like(v_tilde)                       # saving policy given education and location choice
    b_policy = np.zeros_like(v)                                   # saving policy given location choice
    x_policy = np.zeros_like(v, dtype=int)                        # education policy given location choice
    r_probs = np.zeros_like(v)                                    # location choice probabilities


    # start at the end of life - assume V_{t+1} = 0
    for j in range(p.R):
        for (z_index, z) in enumerate(p.z_grid):
            for x in p.x_grid: # uses that x = 0 or 1 (x index == x)
                for (b_index, b) in enumerate(p.b_grid):
                    for r in range(p.R):
                        x_policy[-1, j, r, z_index, x, b_index] = x
                        budget = w[r] * p.age_eff[-1] * np.exp(p.theta * x) * z * p.Delta[j, r] + (1 + i)*b
                        v[-1, j, r, z_index, x, b_index] = p.U(budget, P_c[r], P_n[r], P_h[r])
                    
                    r_probs[-1, j, :, z_index, x, b_index] = p.extr_val_prob(v[-1, j, :, z_index, x, b_index])
                    EV[-1, j, z_index, x, b_index] = p.calc_EV_A(v[-1, j, :, z_index, x, b_index])

    # iterate backwards
    for t in range(p.T - 2, -1, -1):
        for j in range(p.R):
            for (z_index, z) in enumerate(p.z_grid):
                for x in p.x_grid:
                    for (b_index, b) in enumerate(p.b_grid):
                        for r in range(p.R):
                            for x_prime in p.x_grid:
                                EV_ipt = spi.CubicSpline(p.b_grid, 
                                                         p.Pi[z_index,:] @ EV[t+1, r, : , x_prime, :])
                                budget = w[r] * p.age_eff[t] * np.exp(p.theta * x) * z * p.Delta[j, r] + (1 + i)*b + f*(x_prime - x)
                                obj = lambda b_prime: -p.U(budget - b_prime, P_c[r], P_n[r], P_h[r]) - p.beta * EV_ipt(b_prime)
                                res = spo.minimize_scalar(obj, bounds=(p.b_min, p.b_max))
                                if res.success:
                                    b_policy_tilde[t, j, r, z_index, x, x_prime, b_index] = res.x
                                    v_tilde[t, j, r, z_index, x, x_prime, b_index] = -res.fun
                                else:
                                    raise ValueError('Optimization failed')
                                
                            if x==1:
                                x_policy[t, j, r, z_index, x, b_index] = 1
                                b_policy[t, j, r, z_index, x, b_index] = b_policy_tilde[t, j, r, z_index, x, 1, b_index]
                                v[t, j, r, z_index, x, b_index] = v_tilde[t, j, r, z_index, x, 1, b_index]
                            else: # check which x_prime is optimal
                                x_policy[t, j, r, z_index, x, b_index] = np.argmax(v_tilde[t, j, r, z_index, x, :, b_index])
                                b_policy[t, j, r, z_index, x, b_index] = b_policy_tilde[t, j, r, z_index, x, x_policy[t, j, r, z_index, x, b_index], b_index]
                                v[t, j, r, z_index, x, b_index] = np.max(v_tilde[t, j, r, z_index, x, :, b_index])
                        
                        r_probs[t, j, :, z_index, x, b_index] = p.extr_val_prob(v[t, j, :, z_index, x, b_index])
                        EV[t, j, z_index, x, b_index] = p.calc_EV_A(v[t, j, :, z_index, x, b_index])

    return v, b_policy, x_policy, r_probs, EV
