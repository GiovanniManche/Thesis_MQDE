################################################
# constraints.py
# Contains constraint functions used during ATSM estimation.
#
# - rho1_cap_value: inequality constraint imposing an upper bound on one element of the short-rate factor-loading vector rho1.
# - long_run_mean_value: equality constraint imposing a target long-run mean for inflation under the physical measure P.
################################################

import numpy as np

def rho1_cap_value(rho1_P: np.ndarray, factor: int, max_value: float) -> float:
    """
    Inequality constraint on the short-term rate autoregressive coefficient rho1_P
        rho1_P(factor) - max_value <= 0

    Inputs:
    -----------
        :rho1_P: vector of short-rate loadings on the observable factors.
        :factor: one-based index of the factor to constrain. Ex: factor=3 contains the third element of rho1_P (so would create
                    a constraint on the curvature loading)
        :max_value: constraint 
            
    Output:
    ----------
        Left hand side of the above equation
    """
    rho1_P = np.asarray(rho1_P, dtype=float).reshape(-1)
    index = int(factor) - 1
    if index < 0 or index >= len(rho1_P):
        raise IndexError("rho1 cap factor is outside the state dimension.")
    return float(rho1_P[index] - max_value)


def long_run_mean_value(rho0_P: float, rho1_P: np.ndarray, K0P_P: np.ndarray, K1P_P: np.ndarray, target: float) -> float:
    """
    Equality constraint on the long-run mean of inflation under P. With Delta P_t = K0P_P + K1P_P P_{t-1} + eps_t, the unconditional factor mean satisfies
        P_bar = (-K1P_P)^(-1) K0P_P.
    Since inflation is affine in the factors, pi_t = rho0_P + rho1_P' P_t, its long-run mean is rho0_P + rho1_P' P_bar.
    The constraint imposes
        rho0_P + rho1_P' (-K1P_P)^(-1) K0P_P - target = 0.
    
    Inputs:
    -----------
        :rho0_P, rho1_P: intercept and factor loadings of the short-rate equation
        :K0P_P, K1P_P: intercept vector and drift matrix of the factor dynamics under P
        :target: target long-run mean of inflation
            
    Output:
    ----------
        Left hand side of the above equation
    """
    long_run_factors = np.linalg.solve(-K1P_P, K0P_P)  # solve (-K1P_P) P_bar = K0P_P <==> P_bar = (-K1P_P)^(-1) K0P_P
    long_run_inflation = float(rho0_P + rho1_P @ long_run_factors)
    return long_run_inflation - float(target)
