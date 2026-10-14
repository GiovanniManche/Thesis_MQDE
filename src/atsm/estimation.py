################################################
# estimation.py
# Estimates both the standard and persistence-bias-adjusted specifications through a common estimation routine.
# The optimization parameters are:
#   - dlamQ: differences between the ordered canonical Q eigenvalues;
#   - the lower-triangular Cholesky elements of Sigma_P;
#   - K0P_P: intercept of the physical dynamics of the pricing portfolios.
# For each candidate parameter vector, we:
#   1. reconstructs K1Q_X, Sigma_P and K0P_P;
#   2. evaluates the concentrated negative log-likelihood;
#   3. imposes bounds and model restrictions;
#   4. imposes optional restrictions on rho1_P and the long-run mean;
#   5. repeats the constrained optimization several times;
#   6. evaluates the final model and stores parameters and diagnostics.
# In the bias-adjusted specification, Gamma_P depends on the candidate Sigma_P and must therefore also be recomputed 
# inside the long-run-mean nonlinear constraint.
################################################

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from .constraints import long_run_mean_value, rho1_cap_value
from .jsz import compute_loadings_orchestrator, jsz_llk, rotation_canonical_to_observable
from .model import ATSMModel
from .var import estimate_var, estimate_var_calibrated_mean, estimate_var_bias_adjusted_kilian


def estimate_atsm(yields: np.ndarray, W: np.ndarray, maturities: np.ndarray,
                  dt: float, config: dict, random_seed: int = 12345, verbose: bool = True,
                  bias_adjusted: bool = False, apply_killian: bool = True) -> ATSMModel:
    """
    Estimate the JSZ Gaussian ATSM. Setting bias_adjusted=True replaces the unrestricted
    OLS estimate of the physical transition matrix Gamma_P by its Pope/Kilian bias-adjusted counterpart. 
    The Q-pricing block, concentration of kinf, cross-sectional likelihood, parameterization of Sigma_P, optimizer bounds and other model
    restrictions are common to both specifications.

    Inputs:
    ----------
        :yields: observed matrix of ILS rates
        :W: pricing-portfolio weights 
        :maturities: vector of the J ILS maturities expressed in years.
        :dt: model period expressed in years
        :config: dictionary containing optimizer and constraint settings.
        :random_seed: seed used to generate candidate initial Q eigenvalues.
        :verbose: if True, optimizer and initial-seed information is printed.
        :bias_adjusted: if True, use the Pope/Kilian bias-adjusted physical transition matrix in the likelihood and relevant nonlinear constraint.
        :apply_killian: if False, doesn't apply the Killian shrinkage. Useful in expanding setting because for some dates (2022)
                                it is actually impossible to get eigenvalues <1.

    Output:
    ----------
        ATSMModel containing the estimated P- and Q-dynamics, affine yield loadings, short-rate coefficients, likelihood contributions,
        optimizer information and model diagnostics.
    """
    # Imports and basic checks
    yields = np.asarray(yields, dtype=float)
    W = np.asarray(W, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    N, J = W.shape
    if yields.shape[1] != J:
        raise ValueError("W and yields have incompatible maturity dimensions.")

    # ESTIMATION CONFIGURATION 
    # The optimizer jointly estimates: theta = [dlamQ, vech(L), K0P_P],
    #   dlamQ parameterizes the ordered canonical Q eigenvalues;
    #   L = Cholesky factor of Sigma_P, Sigma_P = L L';
    #   K0P_P = physical-measure intercept.

    est_cfg = config["estimation"]
    constraints_cfg = config["constraints"]
    n_seeds = int(est_cfg.get("n_random_seeds", 50))
    n_repeats = int(est_cfg.get("n_repeats", 3))
    method = est_cfg.get("optimizer", "SLSQP")
    ftol = float(est_cfg.get("ftol", 1e-8))
    maxiter = int(est_cfg.get("maxiter", 2000))
    default_llk = float(est_cfg.get("default_bad_likelihood", 1000.0))
    min_total_eigenvalue = float(est_cfg.get("q_min_total_eigenvalue", -0.95))
    first_lb = float(est_cfg.get("q_first_difference_lower", -2.0))
    first_ub = float(est_cfg.get("q_first_difference_upper", 0.5))
    other_ub = float(est_cfg.get("q_other_difference_upper", 0.0))
    chol_diag_lb = float(est_cfg.get("cholesky_diag_lower", 1e-7))
    k0p_lb = float(est_cfg.get("k0p_lower", -1.0))
    k0p_ub = float(est_cfg.get("k0p_upper", 1.0))

    # Pricing ptf
    P = yields @ W.T

    # INITIAL VALUES
    # We initialize Sigma_P and K0P_P from a standard VAR, with K0P_P calibrated so that the VAR-implied unconditional mean equals
    # the empirical mean of the pricing portfolios.
    _, alpha_hat, Omega_hat = estimate_var_calibrated_mean(P)

    Sigma_P0 = Omega_hat
    L0 = np.linalg.cholesky(Sigma_P0)

    # Cholesky decomposition
    lower_mask_f = np.tril(np.ones((N, N), dtype=bool)).flatten(order="F")
    cholSigma_P0 = L0.flatten(order="F")[lower_mask_f]

    # Store the corresponding (row, col) coordinates. These are used
    # below to impose positivity only on the diagonal elements of L
    flat_indices = np.flatnonzero(lower_mask_f)
    lower_rows, lower_cols = np.unravel_index(flat_indices, (N, N), order="F")
    packed_diag = lower_rows == lower_cols
    K0P_P0 = alpha_hat.copy()
    n_chol = N * (N + 1) // 2

    # PARAMETER UNPACKING

    def unpack_parameters(theta: np.ndarray):
        """
        Unpack the optimizer parameter vector theta = [dlamQ, vech(L), K0P_P].
        The canonical Q drift matrix is reconstructed as K1Q_X = diag(cumsum(dlamQ)), while the covariance matrix 
        is guaranteed positive semidefinite through Sigma_P = L L'
        Returns a Tuple (dlamQ, K1Q_X, Sigma_P, K0P_P).
        """
        dlamQ = theta[:N]
        chol_vector = theta[N : N + n_chol]
        K0P_P = theta[N + n_chol :]
        # Recover the ordered canonical Q eigenvalues from their successive differences.
        K1Q_X = np.diag(np.cumsum(dlamQ))
        # Reconstruct the lower-triangular Cholesky
        L_flat = np.zeros(N * N)
        L_flat[lower_mask_f] = chol_vector
        L = L_flat.reshape((N, N), order="F")
        # Candidate covariance matrix of the observable pricing portfolios.
        Sigma_P = L @ L.T

        return dlamQ, K1Q_X, Sigma_P, K0P_P

    def objective(theta: np.ndarray) -> float:
        """
        Defines the function to minimize. jsz_llk returns per-period log-likelihood contributions,
        and what we minimize is their sample mean. If bias_adjusted = True, Gamma_P is recomputed using 
        the Pope/Kilian correction for the current candidate Sigma_P.
        """
        try:
            _, K1Q_X, Sigma_P, K0P_P = unpack_parameters(theta)
            result = jsz_llk(yields_o=yields, W=W, K1Q_X=K1Q_X,
                             Sigma_P=Sigma_P, maturities=maturities, dt=dt, K0P_P=K0P_P,
                             bias_adjusted=bias_adjusted)
            value = float(np.mean(result.llk))
            return value if np.isfinite(value) else default_llk
        except Exception:
            return default_llk

    # RANDOM INITIAL VALUES FOR Q EIGENVALUES
    # We generate several candidate vectors dlamQ and retain the one producing the lowest initial objective.
    # This mitigates the possibility of starting in a very bad likelihood area
    # For i >= 1, dlamQ[i] <= 0 ensures that the reconstructed
    # eigenvalues remain weakly decreasing.
    rng = np.random.default_rng(random_seed)
    best_llk = np.inf
    best_dlam = None
    for _ in range(n_seeds):
        dlam = np.empty(N)
        # Initial value of the first canonical Q eigenvalue.
        dlam[0] = 0.01 * rng.standard_normal()
        if N > 1:
            random_values = rng.random(N - 1)
            ordered = np.sort(np.concatenate(([dlam[0]], random_values)))
            dlam[1:] = -np.diff(ordered)
        theta_seed = np.concatenate([dlam, cholSigma_P0, K0P_P0])
        llk = objective(theta_seed)
        if llk < best_llk:
            best_llk = llk
            best_dlam = dlam.copy()
            if verbose:
                print(f"Improved random seed objective to {llk:.8g}")
    if best_dlam is None:
        raise RuntimeError("Could not construct a finite initial parameter vector.")
    x0 = np.concatenate([best_dlam, cholSigma_P0, K0P_P0])

    # PARAMETER BOUNDS

    lower = np.full_like(x0, -np.inf, dtype=float)
    upper = np.full_like(x0, np.inf, dtype=float)
    lower[0] = first_lb
    upper[0] = first_ub
    if N > 1:
        upper[1:N] = other_ub
    chol_start = N
    chol_end = N + n_chol
    lower[chol_start:chol_end][packed_diag] = chol_diag_lb
    k0_start = chol_end
    lower[k0_start:] = k0p_lb
    upper[k0_start:] = k0p_ub
    bounds = Bounds(lower, upper)

    # MODEL CONSTRAINTS

    scipy_constraints = []

    #       sum(dlamQ) >= -0.95.
    linear_row = np.zeros_like(x0)
    linear_row[:N] = 1.0
    scipy_constraints.append(LinearConstraint(linear_row[None, :], lb=np.array([min_total_eigenvalue]), ub=np.array([np.inf])))

    # We then need the physical K1P_P matrix for other constraints
    # Baseline specification -> estimated by a VAR on its own
    # Bias-adjusted specification -> the corrected K1P_P depends on the current candidate Sigma_P 
    
    K1PplusI, _, _ = estimate_var(P)
    K1P_P_for_constraints = K1PplusI - np.eye(N)

    # Quantities common to nonlinear constraints
    def constraint_rho_quantities(theta: np.ndarray):
        """
        Reconstruct the short-rate coefficients required by the nonlinear
        restrictions for the current optimizer parameter vector. For the current K1Q_X and Sigma_P, the function reproduces the
        same concentration of kinf as the likelihood (see jsz_llk). Sigma_P is also returned because, in the bias-adjusted model,
        the long-run-mean constraint needs the current Sigma_P to recompute the corrected physical transition matrix.
        """
        _, K1Q_X, Sigma_P, K0P_P = unpack_parameters(theta)
        # We build the affine-loading components alpha0, alpha1 and B for the current candidate Q dynamics and Sigma_P.
        try:
            loadings = compute_loadings_orchestrator(W=W, K1Q_X=K1Q_X, rho0_P=0.0,
                                                     Sigma_P=Sigma_P, maturities=maturities, dt=dt)
        except np.linalg.LinAlgError:
            print("Singular candidate dlamQ:", theta[:N])
            print("Q eigenvalues:", np.diag(K1Q_X))
            raise

        # Pricing errors can only live in Null(W), since P = W y are exact observable pricing portfolios.
        V = np.linalg.svd(W, full_matrices=True)[2][N:, :]
        P_orth = V.T @ V

        # Sample means entering the analytical computation of kinf
        mean_yields = yields[1:].mean(axis=0)
        mean_P = P[1:].mean(axis=0)
        residual_mean = mean_yields - loadings.alpha1_P - loadings.B_P.T @ mean_P

        # Concentrate out kinf = [r' P_orth alpha0_P] / [alpha0_P' P_orth alpha0_P].
        numerator = residual_mean @ (P_orth @ loadings.alpha0_P)
        denominator = loadings.alpha0_P @ (P_orth @ loadings.alpha0_P)
        kinf = float(numerator / denominator)

        # Final canonical affine intercept A_X = alpha0_X * kinf + alpha1_X.
        A_X = loadings.alpha0_X * kinf + loadings.alpha1_X
        # Canonical Q intercept K0Q_X = kinf * e_m1.
        K0Q_X = np.zeros(N)
        K0Q_X[loadings.m1 - 1] = kinf

        # Change coordinates from X basis to P basis
        _, _, rho0_P, rho1_P = rotation_canonical_to_observable(W=W, K1Q_X=loadings.K1Q_X, K0Q_X=K0Q_X,
                                                                rho1_X=np.ones(N), rho0_X=0.0, B_X=loadings.B_X, A_X=A_X)

        return rho0_P, rho1_P, K0P_P, Sigma_P

    # Upper bound on a selected rho1_P loading
    rho_cap_cfg = constraints_cfg.get("rho1_cap", {})
    if bool(rho_cap_cfg.get("enabled", False)):
        factor = int(rho_cap_cfg.get("factor", 3))
        cap = float(rho_cap_cfg.get("max_value", 5.0))
        def rho_cap_fun(theta):
            """
            Inequality constraint on one selected element of rho1_P.           
            """
            try:
                _, rho1, _, _ = constraint_rho_quantities(theta)
                return rho1_cap_value(rho1, factor=factor, max_value=cap)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                return 1.0e6
        scipy_constraints.append(NonlinearConstraint(rho_cap_fun, -np.inf, 0.0))

    # Long-run inflation mean constraint
    # Delta P_t = K0P_P + K1P_P P_t + eps_t, so the unconditional factor mean is
    #       E[P] = (-K1P_P)^(-1) K0P_P.
    # Hence long-run inflation is rho0_P + rho1_P' (-K1P_P)^(-1) K0P_P.

    lr_cfg = constraints_cfg.get("long_run_mean", {})
    if bool(lr_cfg.get("enabled", False)):
        target = float(lr_cfg.get("target", 0.02))
        def lr_mean_fun(theta):
            try:
                rho0, rho1, K0P, Sigma_P_current = constraint_rho_quantities(theta)
                return long_run_mean_value(rho0_P=rho0, rho1_P=rho1, K0P_P=K0P, K1P_P=K1P_P_for_constraints, target=target)
            except np.linalg.LinAlgError:
                return 1.0e6
        scipy_constraints.append(NonlinearConstraint(lr_mean_fun, 0.0, 0.0))

    # CONSTRAINED OPTIMIZATION
    # The optimizer is run repeatedly, restarting each pass from the solution of the previous pass. 

    result = None
    x = x0.copy()
    for repeat in range(n_repeats):
        result = minimize(objective, x, method=method,
                          bounds=bounds, constraints=scipy_constraints,
                          options={"ftol": ftol, "maxiter": maxiter, "disp": bool(verbose)})
        x = result.x.copy()
        if verbose:
            print(f"Optimization pass {repeat + 1}/{n_repeats}: objective={result.fun:.10g}, success={result.success}")
    assert result is not None

    # FINAL MODEL EVALUATION
    # Reconstruct the optimal parameters and evaluate jsz_llk once more so that all returned ATSM quantities 
    # correspond exactly to the optimizer solution
    _, K1Q_X, Sigma_P, K0P_P = unpack_parameters(result.x)
    final = jsz_llk(yields_o=yields, W=W, K1Q_X=K1Q_X,
        Sigma_P=Sigma_P, maturities=maturities, dt=dt, K0P_P=K0P_P,
        bias_adjusted=bias_adjusted, apply_killian=apply_killian)

    # DIAGNOSTICS
    eig_P = np.linalg.eigvals(np.eye(N) + final.K1P_P)
    eig_Q = np.linalg.eigvals(np.eye(N) + final.K1Q_P)
    # Model-implied physical long-run factor mean and inflation mean
    try:
        long_factor_mean = np.linalg.solve(-final.K1P_P, final.K0P_P)
        long_run_inflation = float(final.rho0_P + final.rho1_P @ long_factor_mean)
    except np.linalg.LinAlgError:
        long_run_inflation = np.nan
    metadata = {
        "bias_adjusted": bias_adjusted,
        "bias_adjustment_method": "analytical_Pope_Kilian" if bias_adjusted else None,
         "eigenvalues_P_real": eig_P.real,
        "eigenvalues_P_imag": eig_P.imag,
        "eigenvalues_Q_real": eig_Q.real,
        "eigenvalues_Q_imag": eig_Q.imag,

        "max_abs_eigenvalue_P": float(np.max(np.abs(eig_P))),
        "max_abs_eigenvalue_Q": float(np.max(np.abs(eig_Q))),

        "long_run_factors": long_factor_mean,
        "long_run_inflation": long_run_inflation,

        "constraints": constraints_cfg,
    }

    return ATSMModel(
        W=W,
        maturities=maturities.copy(),
        dt=float(dt),
        K0P_P=final.K0P_P,
        K1P_P=final.K1P_P,
        K0Q_P=final.K0Q_P,
        K1Q_P=final.K1Q_P,
        Sigma_P=Sigma_P,
        rho0_P=final.rho0_P,
        rho1_P=final.rho1_P,
        AP=final.A_P,
        BP=final.B_P,
        AX=final.A_X,
        BX=final.B_X,
        kinfQ=final.kinf,
        K0Q_X=final.K0Q_X,
        K1Q_X=final.K1Q_X,
        rho0_X=final.rho0_X,
        rho1_X=final.rho1_X,
        sigma_e=final.sigma_e,
        llk_series=final.llk,
        llkP=final.llkP,
        llkQ=final.llkQ,
        P_monthly=final.P,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        objective_value=float(result.fun),
        optimizer_parameters=result.x.copy(),
        metadata=metadata,
    )