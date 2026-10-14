################################################
# recursion.py
# Contains three functions that compute affine yield loadings.
# The state dynamics are written in variation, X_{t+1} - X_t = K0 + K1 X_t + eps_{t+1},
# while bond-pricing recursions depend on the transition matrix Phi = I + K1.
#
# - compute_ab: recursive computation when the transition matrix Phi is supplied directly.
# - compute_ab_drift: same recursion when the drift matrix K1 is supplied; Phi = I + K1 is applied implicitly.
# - compute_ab_drift_diagonal: closed-form computation when K1 is diagonal
################################################

import numpy as np

def compute_ab(maturities: np.ndarray, rho0_P: float,
    rho1_P: np.ndarray, muQ: np.ndarray,
    PhiQ: np.ndarray, Sigma_P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes A and B matrices in the recursive pricing equations
    A_{p,n+1} = A_n + B_n'muQ + 0.5*B_n'*Sigma_P*B_n - rho_{0,p}
    B_{p,n+1} = PhiQ'*B_n - rho1_P
    A_0, B_0 = 0
    
    What we want are the yield loadings: in a exponential affine setting, 
        price of the bond of horizon n at date t Price(t,n) = exp(A_n + B_n'P_t) with P_t the observabe factors
    The maturity-n YtM is thus -1/n log(Price(t,n)) = -1/n(A_n + B_n'P_t)
    Inputs:
    -----------
        :maturities: array containing all the ILS maturities
        :muQ: K0Q_P
        :PhiQ: I + K1Q_P (dynamics are written in variation)
        :rho0_P, rho1_P, Sigma_P: corresponding values in the estimated ATSM.

    Output:
    ----------
        Tuple containing the yield loadings (a_n, b_n) = (-A_n/n, -B_n/n)
    """
    # Convert in case of bad type + standard checks
    maturities = np.asarray(maturities, dtype=int)
    rho1_P = np.asarray(rho1_P, dtype=float).reshape(-1)
    muQ = np.asarray(muQ, dtype=float).reshape(-1)
    PhiQ = np.asarray(PhiQ, dtype=float)
    Sigma_P = np.asarray(Sigma_P, dtype=float)

    if maturities.ndim != 1 or len(maturities) == 0:
        raise ValueError("maturities must be a non-empty one-dimensional array.")
    if np.any(maturities <= 0) or np.any(np.diff(maturities) < 0):
        raise ValueError("maturities must be positive and sorted.")
    if len(np.unique(maturities)) != len(maturities):
        raise ValueError("maturities must be unique.")

    M = len(maturities)
    N = len(muQ)

    # Initialisation
    A = 0.0
    B = np.zeros(N)
    A_loadings = np.full(M, np.nan)
    B_loadings = np.full((N, M), np.nan)
    curr_mat = 0

    # Compute the yield loadings for each maturity
    for i in range(1, int(maturities[-1]) + 1):
        A = A + muQ @ B + 0.5 * B @ Sigma_P @ B - float(rho0_P)
        B = PhiQ.T @ B - rho1_P
        if curr_mat < M and i == maturities[curr_mat]:
            A_loadings[curr_mat] = -A / maturities[curr_mat]
            B_loadings[:, curr_mat] = -B / maturities[curr_mat]
            curr_mat += 1

    return A_loadings, B_loadings


def compute_ab_drift(maturities: np.ndarray,
    K0d: np.ndarray, K1d: np.ndarray,
    Sigma_d: np.ndarray, rho0d: float, rho1d: np.ndarray,
    timestep: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Same purpose as compute_ab, but takes the drift matrix K1Q_d rather than
    the transition matrix PhiQ_d = I + K1Q_d (d = {P,X})
    A_{d,n+1} = A_n + B_n'K0Q_d + 0.5*B_n'*Sigma_d*B_n - rho_{0,d}
    B_{d,n+1} = (I+K1Q_d)'*B_n - rho1_d = B_n + K1Q_d*B_n - rho1_d
    
    Inputs:
    -----------
        :maturities: array containing all the ILS maturities
        :K0d: K0Q_P or K0Q_X
        :K1d: K1Q_P or K1Q_X
        :rho0d, rho1d, Sigma_d: corresponding values in the estimated ATSM.

    Output:
    ----------
        Tuple containing the yield loadings (b_n, a_n) = (-B_n/n, -A_n/n)
    
    """
    maturities = np.asarray(maturities, dtype=int)
    K0d = np.asarray(K0d, dtype=float).reshape(-1)
    K1d = np.asarray(K1d, dtype=float)
    Sigma_d = np.asarray(Sigma_d, dtype=float)
    rho1d = np.asarray(rho1d, dtype=float).reshape(-1)
    M = len(maturities)
    N = len(K0d)
    A = 0.0
    B = np.zeros(N)
    A_loadings = np.full(M, np.nan)
    B_loadings = np.full((N, M), np.nan)
    curr_mat = 0

    for i in range(1, int(maturities[-1]) + 1):
        A = A + K0d @ B + 0.5 * B @ Sigma_d @ B - float(rho0d)
        B = B + K1d.T @ B - rho1d    # We reconstruct the persistence matrix I + K1d here directly

        if curr_mat < M and i == maturities[curr_mat]:
            A_loadings[curr_mat] = -A / maturities[curr_mat]
            B_loadings[:, curr_mat] = -B / maturities[curr_mat]
            curr_mat += 1

    if timestep is not None:
        A_loadings = A_loadings / timestep
        B_loadings = B_loadings / timestep

    return B_loadings, A_loadings


def compute_ab_drift_diagonal(maturities: np.ndarray,
    K0d: np.ndarray, K1d_diag: np.ndarray,
    Sigma_d: np.ndarray, rho0d: float,
    rho1d: np.ndarray, timestep: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Same purpose as compute_ab_drift, but with K1Q_d diagonal.
    Let G = I + K1, with diagonal entries g_i. One can then show that the sums entering
    the n-period bond price can be written:
        mu1_i = sum_{j=0}^{n-1} g_i^j = (1 - g_i^n) / (1 - g_i)
        mu0_i = K0_i * sum_{j=0}^{n-2} (n-1-j) g_i^j = -((n - mu1_i) / K1_i) * K0_i
        Sigma0_ab = Sigmad_ab * sum_{j=1}^{n-1} [(1 - g_a^j)/(1 - g_a)] [(1 - g_b^j)/(1 - g_b)]

    The corresponding n-period yield is affine: y_t^n = A_n + B_n' X_t
    with
        B_n = rho1 * mu1 / n
        A_n = rho0 + (rho1' mu0 - 0.5 * rho1' Sigma0 rho1) / n

    These formulas are equivalent to the recursive affine-pricing equations, 
    but exploit diagonal K1 to replace matrix recursions with geometric-series closed forms
    Allows for faster computation. Requires K1_i != 0 and 1 - g_a g_b != 0 for all relevant pairs,
    so the current closed form does not handle unit roots.
    Inputs/Outpus -> see compute_ab_drift
    """
    maturities = np.asarray(maturities, dtype=int)
    K0d = np.asarray(K0d, dtype=float).reshape(-1)
    K1d_diag = np.asarray(K1d_diag, dtype=float).reshape(-1)
    Sigma_d = np.asarray(Sigma_d, dtype=float)
    rho1d = np.asarray(rho1d, dtype=float).reshape(-1)

    M = len(maturities)
    N = len(K0d)

    A = np.zeros(M)
    B = np.zeros((N, M))

    ones_n = np.ones(N)
    G = K1d_diag + 1.0
    GG = np.outer(G, G)

    for m, mat in enumerate(maturities):
        if mat == 1:
            B[:, m] = rho1d
            A[m] = rho0d
            continue

        i = int(mat - 1)
        # If division by 0 -> error is raised
        # If not, we apply the closed-form formulas
        with np.errstate(divide="raise", invalid="raise"):
            mu1 = ones_n - (G - G ** (i + 1)) / K1d_diag
            mu0 = -(((i + 1) * ones_n - mu1) / K1d_diag) * K0d
            B[:, m] = rho1d * mu1 / mat
            sigma_term1 = i * Sigma_d
            sigma_term2 = np.outer(mu1 - 1.0, ones_n) * Sigma_d
            sigma_term3 = sigma_term2.T
            ratio = (GG - GG ** (i + 1)) / (1.0 - GG)
            sigma_term4 = ratio * Sigma_d
            denom = np.outer(K1d_diag, K1d_diag)
            Sigma0 = (sigma_term1 - sigma_term2 - sigma_term3 + sigma_term4) / denom
            A[m] = rho0d + (rho1d @ mu0 - 0.5 * rho1d @ Sigma0 @ rho1d) / mat
            
    if timestep is not None:
        B = B / timestep
        A = A / timestep

    return B, A
