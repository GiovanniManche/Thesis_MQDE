
################################################
# var.py
# VAR estimation and finite-sample bias-correction used by the ATSM.
# - estimate_var: estimates an unrestricted VAR(1) with intercept by OLS
# - estimate_var_calibrated_mean: estimates the VAR(1) slope matrix by OLS and recalibrates the
#     intercept so that the model-implied unconditional mean equals the empirical mean of the factors.
# - analytical_var_bias: computes the analytical O(1/T) finite-sample bias approximation for the VAR(1) transition matrix, following Pope (1990).
# - estimate_var_bias_adjusted_kilian: applies the analytical bias correction to the VAR dynamics,
#     adjusts the intercept consistently, and applies a Killian-like persistence shrinkage procedure.
################################################

import numpy as np

def estimate_var(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a standard VAR X[t+1] = alpha + Gamma @ X[t] + eps[t+1] by OLS

    Inputs:
    ----------
        :X: matrix of variables (here the observed pricing portfolio P).
    
    Output:
    ----------
        Tuple (Gamma_hat, alpha_hat, Omega_hat) with the estimated autoregressive coefficients, intercepts and covariance matrix
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a T x N matrix with T >= 2.")
    T, N = X.shape
    Yt = X[:-1]
    Ytp1 = X[1:]
    Z = np.column_stack([np.ones(T - 1), Yt])  # (T-1) x (N+1)

    # Equivalent to A = Y*Z'/(Z*Z') but numerically preferable.
    coeff, *_ = np.linalg.lstsq(Z, Ytp1, rcond=None)
    # coeff is (N+1) x N;
    A = coeff.T
    alpha_hat = A[:, 0]
    Gamma_hat = A[:, 1:]
    residuals = Ytp1 - Z @ coeff
    Omega_hat = residuals.T @ residuals / (T - 1)
    return Gamma_hat, alpha_hat, Omega_hat


def estimate_var_calibrated_mean(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    VAR estimated by OLS but we force the intercept to be such that the unconditional mean of the VAR(1) is equal to the
    empirical mean of the factors (P_bar). 
    Unconditional mean of a VAR(1) = (I - Gamma)^{-1}*alpha so by imposing alpha = (I-Gamma)P_bar we have the unconditional 
    mean = P_bar.   
    Omega_hat is computed from the residuals of the unrestricted OLS regression, i.e before replacing alpha by the calibrated intercept.

    Inputs:
    ----------
        :X: matrix of variables (here the observed pricing portfolio P).
    
    Output:
    ----------
        Tuple (Gamma_hat, alpha_hat, Omega_hat) with the estimated autoregressive coefficients, calibrated intercept and covariance matrix
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a T x N matrix with T >= 2.")
    T, N = X.shape
    Yt = X[:-1]
    Ytp1 = X[1:]
    Z = np.column_stack([np.ones(T - 1), Yt])
    coeff, *_ = np.linalg.lstsq(Z, Ytp1, rcond=None)
    A = coeff.T
    Gamma_hat = A[:, 1:]
    alpha_hat = (np.eye(N) - Gamma_hat) @ X.mean(axis=0)
    residuals = Ytp1 - Z @ coeff
    Omega_hat = residuals.T @ residuals / (T - 1)
    return Gamma_hat, alpha_hat, Omega_hat


def analytical_var_bias(Gamma: np.ndarray, Sigma: np.ndarray, T: int) -> np.ndarray:
    """
    Analytical finite-sample VAR(1) bias approximation based on Pope (1990). Especially relevant since ILS are peristent.
    Refer to Pope (1990) or the associated paper for all the details / equations.
    The goal is to compute Bias(Gamma_hat) = E[Gamma_hat - Gamma] approx - B/T

    Inputs:
    ----------
        :Gamma: VAR(1) transition matrix
        :Sigma: innovation covariance matrix
        :T: number of observation 
    
    Output:
    ----------
        Array of analytical bias approximation
    """
    # Imports and standard checks
    Gamma = np.asarray(Gamma, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if Gamma.ndim != 2 or Gamma.shape[0] != Gamma.shape[1]:
        raise ValueError("Gamma must be a square matrix.")
    k = Gamma.shape[0]
    if Sigma.shape != (k, k):
        raise ValueError(f"Sigma must have shape {(k, k)}, got {Sigma.shape}.")
    if T <= 0:
        raise ValueError("T must be strictly positive.")
    Omega = Sigma

    # We first compute the unconditional covariance of the factors:
    # Var(X_t) = Gamma_0 satisfies Gamma_0 = Gamma Gamma_0 Gamma' + Omega which after vectorization yields
    # vec(Gamma_0) = [I-Gamma ox Gamma]^-1 vec(Omega)

    system = np.eye(k * k) - np.kron(Gamma, Gamma)
    omega_vec = Omega.reshape(k * k,order="F")
    gamma0_vec = np.linalg.solve(system,omega_vec)
    Gamma0 = gamma0_vec.reshape((k, k), order="F")

    # Now we compute S = sum_{i=1}^N lambda_i (I - lambda_i Gamma')^-1
    # lambdas are the eigenvalues of Gamma_hat'
    lambdas = np.linalg.eigvals(Gamma.T)
    S = np.zeros((k, k),dtype=complex)
    I = np.eye(k)
    for lam in lambdas:
        S += lam * np.linalg.solve(I - lam * Gamma.T, I)
    # A real non-symmetric matrix may have complex-conjugate eigenvalues. Their contributions should combine into a real-valued S.
    # real_if_close removes negligible imaginary parts due to numerical noise; a materially complex final result signals a numerical problematic case.
    S = np.real_if_close(S, tol=1000)
    if np.iscomplexobj(S):
        raise ValueError("analytical_var_bias produced materially complex values.")

    # Computation of b_Gamma
    term1 = np.linalg.solve(I - Gamma.T, I)
    term2 = Gamma.T @ np.linalg.solve(I - Gamma.T @ Gamma.T,I)
    middle = (term1 + term2 + S)
    numerator = Omega @ middle
    b = np.linalg.solve(Gamma0.T, numerator.T).T
    b = np.real_if_close(b, tol=1000)
    if np.iscomplexobj(b):
        raise ValueError("Analytical bias calculation produced materially complex values.")
    bias = -np.asarray(b, dtype=float) / T
    return bias


def estimate_var_bias_adjusted_kilian(X: np.ndarray, Omega: np.ndarray,
                                      apply_killian: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates the adjusted for persistence bias VAR. It:
        1. estimates the VAR by OLS
        2. applies the analytical finite-sample correction
        3. adjusts the intercept consistently with that correction
        4. applies Kilian-style shrinkage if the largest corrected eigenvalue is >= 0.99. That is,
        if Gamma_hat_corrected has eigenvalues too close to 1, to ensure stationarity we do not apply the 
        entire bias correction but only a fraction kappa of it.  
    
    Inputs:
    ----------
        :X: observed variables (here the factors)
        :Omega: Innovation covariance matrix supplied externally. ATSM -> called Sigma_P and evaluated in the likelihood
        :T: number of observation 
        :apply_killian: if False, doesn't apply the Killian shrinkage. Useful in expanding setting because for some dates (2022)
                        it is actually impossible to get eigenvalues <1.
    
    Output:
    ----------
        Tuple (Gamma_bias_adj, alpha_hat_bias_adj, Omega_hat) with the adjusted autoregressive coefficients,
        intercepts and covariance matrix
    """
    X = np.asarray(X, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    apply_killian = apply_killian
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a T x N matrix with T >= 2.")
    T, N = X.shape
    if Omega.shape != (N, N):
        raise ValueError(f"Omega must have shape {(N, N)}, got {Omega.shape}.")

    # We estimate the unrestricted OLS
    Gamma_hat, alpha_hat, _ = estimate_var(X)

    # We compute the analytical correction
    bias = analytical_var_bias(Gamma_hat, Omega, T)
    bias_Gamma_hat = -bias * T

    # Autoregressive coefficients adjustment
    Gamma_bias_adj = Gamma_hat + bias_Gamma_hat / T

    # We apply the intercept correction
    # bias_alpha is not equal to B -> see Pope and paper

    long_run_mean_hat = np.linalg.solve(np.eye(N) - Gamma_hat, alpha_hat)
    bias_alpha = -bias_Gamma_hat @ long_run_mean_hat
    alpha_hat_bias_adj = alpha_hat + bias_alpha / T
    
    # Kilian correction if the corrected VAR becomes too persistent.
    
    def _max_eigenvalue(matrix: np.ndarray) -> float:
        """
        - if eigenvalues are real, max() selects the largest value;
        - if the vector is complex, max() selects the eigenvalue
            with the largest magnitude;
        - the subsequent >= comparison uses the real part.
        """
        eigvals = np.linalg.eigvals(matrix)
        # Treat negligible numerical imaginary parts as real.
        if np.all(np.abs(eigvals.imag) < 1e-12):
            return float(np.max(eigvals.real))

        magnitudes = np.abs(eigvals)
        max_magnitude = np.max(magnitudes)

        candidate_indices = np.flatnonzero(np.isclose(magnitudes,max_magnitude,rtol=1e-12,atol=1e-15))
        if len(candidate_indices) == 1:
            selected = eigvals[candidate_indices[0]]
        else:
            phases = np.angle(eigvals[candidate_indices])
            selected = eigvals[candidate_indices[np.argmax(phases)]]

        return float(selected.real)

    max_eig = _max_eigenvalue(Gamma_bias_adj)
    # Killian-type correction shrinkage
    if apply_killian:
        kappa = 0.995
        while max_eig >= 0.99:
            alpha_hat_bias_adj = alpha_hat + kappa * bias_alpha / T
            Gamma_bias_adj = Gamma_hat + kappa * bias_Gamma_hat / T
            max_eig = _max_eigenvalue(Gamma_bias_adj)
            kappa -= 0.005
            # Under a stationary OLS VAR the loop should stop well before this.
            # But for instance in 2022 not the case!!
            if kappa < 0:
                raise RuntimeError("Kilian correction failed to obtain a maximum eigenvalue below 0.99.")

    # We can recompute innovations covariance using the bias adjusted dynamics
    Xdem = X - X.mean(axis=0, keepdims=True)
    eps_tilde = Xdem[1:].T - Gamma_bias_adj @ Xdem[:-1].T
    Omega_hat = eps_tilde @ eps_tilde.T / (T - 1)

    return (Gamma_bias_adj, alpha_hat_bias_adj, Omega_hat)