################################################
# jsz.py
# Core implementation of the JSZ Gaussian ATSM.
# The module links the canonical factor representation X used for JSZ identification to the observable pricing-portfolio representation P = W y.
# It constructs affine yield loadings implied by candidate Q-dynamics, maps model quantities between the canonical and observable bases,
# and evaluates the cross-sectional and time-series components of the log-likelihood.
#
# - _right_solve: does A @ inv(B) without explicitly computing the inverse.
# - jordan_type_if_bad_diagonal: computes smooth superdiagonal weights when adjacent canonical roots
#                                are too close for the standard diagonal representation.
#                                (could lead to numerical difficulties if we don't do that)
# - adjust_diagonal_matrix: orders and, when necessary, adjusts the canonical Q drift matrix
#                           K1Q_X to handle near-unit or nearly repeated roots, and determines
#                           the canonical intercept position m1.
# - rotation_canonical_to_observable: maps Q-dynamics and short-rate coefficients from the canonical
#                                     factor basis X to the observable pricing-portfolio basis P.
# - compute_loadings_orchestrator: constructs affine yield loadings in the canonical and observable
#                                  representations and recovers the associated Q parameters.
# - reduced_rank_free_intercept_regress: estimates a multivariate regression with a free intercept and an optional reduced-rank restriction.
# - jsz_llk: evaluates the concentrated JSZ negative log-likelihood
################################################


from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space, eig, cholesky
from .recursion import compute_ab_drift, compute_ab_drift_diagonal
from .var import estimate_var, estimate_var_bias_adjusted_kilian

@dataclass(slots=True)
class JSZLoadingsResult:
    """
    Intermediate affine-pricing quantities produced by the JSZ loading construction.
    Contains yield loadings and Q-dynamics in both the canonical JSZ factor basis X and the observable pricing-portfolio basis P.
    Also stores the decomposition of the affine intercept into its deterministic and covariance-related components, together with
    the canonical factor associated with the Q intercept.
    """
    # Yield loadings and ATSM coefficients - observable pricing potfolios
    B_P: np.ndarray
    A_P: np.ndarray
    K0Q_P: np.ndarray
    K1Q_P: np.ndarray
    rho0_P: float
    rho1_P: np.ndarray
    # Yield loadings and ATSM coefficients - normalised basis
    K0Q_X: np.ndarray
    K1Q_X: np.ndarray
    A_X: np.ndarray
    B_X: np.ndarray
    Sigma_X: np.ndarray
    alpha0_P: np.ndarray
    alpha1_P: np.ndarray
    alpha0_X: np.ndarray
    alpha1_X: np.ndarray
    m1: int


@dataclass(slots=True)
class JSZLikelihoodResult:
    """
    Output of one evaluation of the JSZ likelihood.
    Stores:
        - total, P and Q negative log-likelihood contributions;
        - affine yield loadings;
        - P- and Q-dynamics in the observable pricing-portfolio basis;
        - Q-dynamics in the canonical basis;
        - short-rate coefficients;
        - pricing portfolios;
        - concentrated parameters such as kinf and sigma_e.
    """
    llk: np.ndarray
    A_P: np.ndarray
    B_P: np.ndarray
    A_X: np.ndarray
    B_X: np.ndarray
    kinf: float
    K0P_P: np.ndarray
    K1P_P: np.ndarray
    sigma_e: float
    K0Q_P: np.ndarray
    K1Q_P: np.ndarray
    rho0_P: float
    rho1_P: np.ndarray
    P: np.ndarray
    llkP: np.ndarray
    llkQ: np.ndarray
    K0Q_X: np.ndarray
    K1Q_X: np.ndarray
    rho0_X: float
    rho1_X: np.ndarray


def _right_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return A / B but we prefer using that instead of inv(B) because of computational issues.
    """
    return np.linalg.solve(B.T, A.T).T  #B'x = A' ==> x = (B")^-1A'


def jordan_type_if_bad_diagonal(x: np.ndarray, eps1: float = 1e-3) -> np.ndarray:
    """
    Smooth log-scale cutoff used when adjusting the canonical Q drift matrix.
    Returns values between 0 and 1:
        - 1 for zero or very small positive arguments (diagonal coefs are virtually equal)
        - smoothly decreasing values for intermediate arguments
        - 0 once the argument exceeds eps1. (acceptable differences)
    It is used by adjust_diagonal_matrix to introduce a superdiagonal link when
    adjacent canonical eigenvalues are too close for the standard diagonal representation.
    If the diagonal form is problematic because diagonal coefs are too close, we transform it into a Jordan-type
    upper-triangular matrix (thus does not affect the eigenvalues of the matrix but in the code we use the standard recursions 
    instead of the closed-form for diagonal matrices)

    Inputs:
        -----------
            :x: vector of absolute differences between the diagonal coefficients of a matrix.
            :eps1: Threshold under which two diagonal elements are considered too close    
    Output:
        ----------
            Array of weights to put in the superdiagonal 
    """
    x = np.asarray(x, dtype=float)
    eps0 = 1e-5
    out = np.zeros_like(x)
    out[x == 0] = 1.0
    positive = x > 0
    low = positive & (x < eps0)
    mid = positive & (x >= eps0) & (x < eps1)
    out[low] = 1.0
    out[mid] = 1.0 - ((np.log(x[mid]) - np.log(eps0)) / (np.log(eps1) - np.log(eps0)))
    return out


def adjust_diagonal_matrix(K1Q_X: np.ndarray, eps1: float = 1e-3) -> tuple[np.ndarray, bool, int]:
    """
    Adjust the canonical Q drift matrix K1Q_X used in the JSZ representation.
    If K1Q_X is diagonal, its diagonal elements are first ordered, then we check for roots close to zero in K1Q_X
    (= near-unit root in the transition matrix I+K1Q_X) and nearly repeated canonical roots.
    When the standard diagonal representation becomes numerically problematic (nearly repeated eigenvalues),
    superdiagonal elements are introduced using jordan_type_if_bad_diagonal.
    When near-UR is detected, this function prevents the use of the closed-form formulas without introducing any superdiagonal.
    The function also determines m1, the one-based canonical index used to locate the non-zero element of K0Q_X.

    Inputs:
    -----------
        :K1Q_X: Canonical Q drift matrix.
        :eps1: Threshold used both for near-unit-root detection and for determining whether adjacent roots are sufficiently distinct.   
    
    Output:
    ----------
        Tuple (K1Q_X, diagonal_closed_form_safe, m1) with K1Q_X the adjusted canonical Q drift matrix,
        diagonal_closed_form_safe = True when closed-form for yield loadings under Q can be used, and 
        m1 the one-based location of the canonical Q intercept.
    """
    K1Q_X = np.asarray(K1Q_X, dtype=float).copy()
    diag_values = np.diag(K1Q_X).copy()

    # We first make sure that the K1Q_X matrix is indeed diagonal. If not -> directly rejects the use of closed-form formula
    is_diagonal = np.array_equal(K1Q_X, np.diag(diag_values))
    if is_diagonal:
        # Sort the diagonal elements
        diag_values = np.sort(diag_values)[::-1]
        K1Q_X = np.diag(diag_values)
        # Check for potential unit roots
        has_near_unit_root = not np.all(np.abs(diag_values) > eps1)
        # Check for potentially too close roots
        has_near_repeated_root = (len(diag_values) > 1 and not np.all(np.abs(np.diff(diag_values)) > eps1))
        # If no UR and no repeated diagonal elements, safe to use the closed-form formula for A and B
        diagonal_closed_form_safe = (is_diagonal and not has_near_repeated_root and not has_near_unit_root)
    else:
        diagonal_closed_form_safe = False

    # If the matrix is diagonal but has too close eigenvalues, then change it into Jordan-type
    if is_diagonal and not diagonal_closed_form_safe and len(diag_values) > 1:
        diff_diag = np.abs(np.diff(diag_values))
        super_diag = jordan_type_if_bad_diagonal(diff_diag, eps1=eps1)
        for i, value in enumerate(super_diag):
            K1Q_X[i, i + 1] += value

    # m1 is the one-based position of the last factor belonging to the
    # first connected canonical chain. Equivalently, it determines the
    # position of the unique non-zero element of K0Q_X.
    # If all superdiagonal links are zero, m1 = 1.
    # If the first k factors are successively linked, m1 = k.
    N = K1Q_X.shape[0]
    super_diagonal = np.array([K1Q_X[i, i + 1] for i in range(N - 1)])
    chain = np.concatenate(([1.0], super_diagonal))
    cumulative = np.cumprod(chain)
    nonzero = np.flatnonzero(cumulative != 0)
    m1 = int(nonzero.max() + 1) if len(nonzero) else 1

    return K1Q_X, diagonal_closed_form_safe, m1


def rotation_canonical_to_observable(W: np.ndarray, K1Q_X: np.ndarray, K0Q_X: np.ndarray,
    rho1_X: np.ndarray, rho0_X: float, B_X: np.ndarray, A_X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Maps model parameters from the canonical JSZ basis X to the observable pricing-portfolio basis P.
    Mechanism: canonical yields satisfy y_t = A_X + B_X' X_t while observable pricing portfolios are P_t = W y_t.
    So P_t = W A_X + W B_X' X_t = a + C X_t, a = W A_X, C = W B_X'.
    Using X_t = C^{-1}(P_t - a), the Q-dynamics and short-rate equation can be rewritten in observable base:
        K1Q_P = C K1Q_X C^{-1},
        K0Q_P = C K0Q_X - K1Q_P a,
        rho0_P = rho0_X - rho1_X' C^{-1} a,
        rho1_P = C^{-T} rho1_X.

    Inputs:
    -----------
        :W: PCA pricing-portfolio weights.
        :K1Q_X, K0Q_X: Q drift matrix and intercept vector in the canonical basis.
        :rho1_X, rho0_X: Short-rate factor loadings and intercept in the canonical basis.
        :B_X, A_X: Affine yield slope and intercept loadings in the canonical basis.

    Output
    ----------
        Tuple (K0Q_P, K1Q_P, rho0_P, rho1_P), expressed in the observable pricing-portfolio basis.
    """
    W = np.asarray(W, dtype=float)
    B_X = np.asarray(B_X, dtype=float)
    A_X = np.asarray(A_X, dtype=float).reshape(-1)

    C = W @ B_X.T 
    a = W @ A_X   

    K1Q_P = _right_solve(C @ K1Q_X, C)
    K0Q_P = C @ K0Q_X - K1Q_P @ a
    solved_wA_X = np.linalg.solve(C, a)
    rho0_P = float(rho0_X - rho1_X @ solved_wA_X)
    rho1_P = np.linalg.solve(C.T, rho1_X)

    return K0Q_P, K1Q_P, rho0_P, rho1_P


def compute_loadings_orchestrator(W: np.ndarray, K1Q_X: np.ndarray,
    rho0_P: float, Sigma_P: np.ndarray, maturities: np.ndarray, dt: float,
    Sigma_X: np.ndarray | None = None) -> JSZLoadingsResult:
    """
    Construct affine JSZ yield loadings conditional on candidate Q-dynamics. The canonical yield intercept can be decomposed as
        A_X = alpha0_X * kinf + alpha1_X,
    where alpha0_X = contribution of the canonical Q intercept per unit of kinf and alpha1_X = Gaussian covariance / convexity contribution.
    The function:
        1. adjusts K1Q_X when necessary;
        2. computes yield loadings with K0Q_X normalized to one and Sigma_X = 0, thereby obtaining alpha0_X;
        3. recovers Sigma_X from Sigma_P using the affine mapping between canonical and observable factors;
        4. recomputes the intercept including Gaussian convexity and obtains alpha1_X by difference;
        5. for the supplied rho0_P, solves analytically for the compatible canonical Q-intercept scalar kinf;
        6. constructs final canonical yield loadings;
        7. expresses yield loadings and Q parameters in the observable pricing-portfolio basis.
    In jsz_llk, the function is called with rho0_P = 0 primarily to obtain B, alpha0 and alpha1. 
    The final likelihood-specific kinf is concentrated out using the cross-sectional pricing errors.

    Inputs
    -------------
        :W: PCA pricing-portfolio weights.
        :K1Q_X: candidate Q drift matrix in the canonical basis.
        :rho0_P: observable-basis short-rate intercept used to recover a compatible value of kinf.
        :Sigma_P: factor innovation covariance matrix in the observable basis.
        :maturities: vector of ILS maturities expressed in years.
        :dt: Model period expressed in years, e.g. 1/12 for monthly dynamics.
        :Sigma_X: Optional innovation covariance in the canonical basis.
        If omitted, it is recovered from Sigma_P.

    Output:
    -----------
        JSZLoadingsResult dataclass (refer to the corresponding code)
    """
    # Import and initialisation
    W = np.asarray(W, dtype=float)
    K1Q_X = np.asarray(K1Q_X, dtype=float)
    Sigma_P = np.asarray(Sigma_P, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    J = len(maturities)
    N = K1Q_X.shape[0]

    # In canonical form, rho0_X = 0, rho1_X = 1_N
    rho0d = 0.0
    rho1d = np.ones(N)
    # We convert maturities (year) into model periods (monthly if dt = 1/12)
    mats_periods = np.rint(maturities / dt).astype(int)

    # Adjust K1Q_X if necessary
    # m1 determines the position of the unique non-zero element of K0Q_X
    K1Q_X, diagonal_closed_form_safe, m1 = adjust_diagonal_matrix(K1Q_X)
    # Temporarily normalize the Q intercept to one: K0Q_X = e_m1
    # rather than its final value kinf * e_m1.
    # This lets us compute the sensitivity of yield intercepts to kinf.
    K0Q_X = np.zeros(N)
    K0Q_X[m1 - 1] = 1.0

    # First pricing ignores covariance bc we want to get the deterministic part
    # With: K0Q_X[m1-1] = 1 and Sigma_X = 0,
    # the resulting yield intercept is alpha0_X.
    zero_sigma = np.zeros((N, N))

    # Use the diagonal closed-form formulas for A and B if possible, otherwise uses standard recursion equations
    if diagonal_closed_form_safe:
        B_X, alpha0_X = compute_ab_drift_diagonal(mats_periods, K0Q_X, np.diag(K1Q_X),
                                                  zero_sigma, rho0d * dt, rho1d * dt, timestep=dt)
    else:
        B_X, alpha0_X = compute_ab_drift(mats_periods, K0Q_X, K1Q_X, zero_sigma, rho0d * dt,
                                        rho1d * dt, timestep=dt)

    # Mapping between observable factors and canonical base (see rotation_canonical_to_observable documentation)
    C = W @ B_X.T
    # If Sigma_X has not been supplied, we recover it from Sigma_P:
    #     Sigma_X = C^{-1} Sigma_P C^{-T}.
    if Sigma_X is None:
        temp = np.linalg.solve(C, Sigma_P)
        Sigma_X = _right_solve(temp, B_X @ W.T)

    # Now we recompute the affine yield intercept using the actual Sigma_X.
    # K0Q_X is STILL normalized to one here. Therefore the resulting intercept A_X1 contains:
    #       A_X1 = alpha0_X + covariance contribution.
    if diagonal_closed_form_safe:
        B_X, A_X1 = compute_ab_drift_diagonal(mats_periods, K0Q_X, np.diag(K1Q_X), Sigma_X, rho0d * dt, 
                                              rho1d * dt, timestep=dt)
    else:
        B_X, A_X1 = compute_ab_drift(mats_periods, K0Q_X, K1Q_X, Sigma_X, rho0d * dt, rho1d * dt, timestep=dt)

    # Difference between first and second computations of A_X = covariance contribution
    alpha1_X = A_X1 - alpha0_X
    # B_X is theoretically unchanged between the two calls because the B recursion does not depend on Sigma_X.
    C = W @ B_X.T

    # kinf can actually be recovered using other estimated parameters (i.e no need to incorporate it in the likelihood computations)
    # Indeed rho0_P = rho0_X - rho1_X' C^{-1}*a with rho0_X = 0 and rho1_X = 1.
    # Since A_X = alpha0_X * kinf + alpha1_X, we have:
    #   a = W A_X = W alpha0_X * kinf + W alpha1_X.
    # Define:
    #     a0 = 1' C^{-1} W alpha0_X
    #     a1 = 1' C^{-1} W alpha1_X.
    # Then rho0_P = -(a0 * kinf + a1),
    # so kinf can be solved analytically: kinf = -(rho0_P + a1) / a0.
    W_alpha0 = W @ alpha0_X
    W_alpha1 = W @ alpha1_X
    a0 = float(np.ones(N) @ np.linalg.solve(C, W_alpha0))
    a1 = float(np.ones(N) @ np.linalg.solve(C, W_alpha1))
    kinf = -(float(rho0_P) + a1) / a0
    # We replace the temporary unit normalization by the actual Q intercept K0Q_X = kinf * e_m1.
    # So we can finally construct the canonical yield intercept A_X
    K0Q_X = np.zeros(N)
    K0Q_X[m1 - 1] = kinf
    A_X = alpha0_X * kinf + alpha1_X

    # Now we can express affine yield loadings in observable P basis
    # First change of basis
    right = _right_solve(B_X.T, W @ B_X.T)
    projection = np.eye(J) - right @ W
    # Then we can express deterministic and covariance intercept in the P basis
    alpha0_P = projection @ alpha0_X
    alpha1_P = projection @ alpha1_X
    B_P = np.linalg.solve((W @ B_X.T).T, B_X)
    A_P = A_X @ (np.eye(J) - B_X.T @ np.linalg.solve(W @ B_X.T, W)).T

    # We rotate the Q dynamics and short rate parameters from X to P basis
    C = W @ B_X.T
    a = W @ A_X
    K1Q_P = _right_solve(C @ K1Q_X, C)
    K0Q_P = C @ K0Q_X - K1Q_P @ a
    rho0_P_out = float(-np.ones(N) @ np.linalg.solve(C, a))
    rho1_P = np.linalg.solve(C.T, np.ones(N))

    return JSZLoadingsResult(B_P=B_P, A_P=A_P, K0Q_P=K0Q_P, K1Q_P=K1Q_P,
        rho0_P=rho0_P_out, rho1_P=rho1_P, K0Q_X=K0Q_X, K1Q_X=K1Q_X,
        A_X=A_X, B_X=B_X, Sigma_X=Sigma_X, alpha0_P=alpha0_P,
        alpha1_P=alpha1_P, alpha0_X=alpha0_X, alpha1_X=alpha1_X, m1=m1)


def reduced_rank_free_intercept_regress(Y: np.ndarray, X: np.ndarray, rank: int | None = None,
                                        Omega: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a multivariate regression with an unrestricted intercept and an optional reduced-rank restriction on the slope matrix.
    For rank = N, it's OLS. For rank < N, the slope matrix is constrained to the requested rank,
    using either an unweighted reduced-rank regression or an Omega-weighted transformation when the innovation covariance is supplied.
    
    Inputs:
    -----------
        :Y: matrix of dependent variables.
        :X: matrix of regressors.
        :rank: Rank imposed on the slope matrix. If None, we suppose full rank.
        :Omega: innovation covariance matrix used for the weighted reduced-rank regression.

    Output:
    ----------
        Tuple (beta, alpha, Omega_out, log_llk) with slope matrix, intercept vector, innovation covariance matrix
        and per-observation likelihood
    """
    # Import and basic checks
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, N = Y.shape
    M = X.shape[1]

    # If rank is not provided, we suppose full rank
    if rank is None:
        rank = N
    # OLS
    Ybar = Y.mean(axis=0)
    Xbar = X.mean(axis=0)
    Yc = Y - Ybar
    Xc = X - Xbar
    beta_ols, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)

    # Type of regression
    if rank == N:
        beta = beta_ols
    elif Omega is None:
        M00 = (Yc.T @ Yc) / T
        M0k = (Yc.T @ Xc) / T
        Mkk = (Xc.T @ Xc) / T
        A = M0k.T @ np.linalg.solve(M00, M0k)
        evals, evecs = eig(A, Mkk)
        order = np.argsort(evals.real)[::-1]
        b = evecs[:, order[:rank]].real
        a = M0k @ b @ np.linalg.inv(b.T @ Mkk @ b)
        beta = (a @ b.T).T
    else:
        P = cholesky(Xc.T @ Xc, lower=False)
        L = cholesky(Omega, lower=False)
        transformed = _right_solve(P @ beta_ols, L)
        U, s, Vh = np.linalg.svd(transformed, full_matrices=False)
        s[rank:] = 0.0
        beta = np.linalg.solve(P, (U * s) @ Vh @ L)
    if Omega is None:
        residuals = Yc - Xc @ beta
        Omega_out = residuals.T @ residuals / T
    else:
        Omega_out = np.asarray(Omega, dtype=float)

    residuals = Yc - Xc @ beta
    solved = np.linalg.solve(Omega_out, residuals.T)
    log_llk = N / 2 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(Omega_out)) + np.sum(residuals.T * solved, axis=0)
    alpha = Ybar - Xbar @ beta
    return beta, alpha, Omega_out, log_llk


def jsz_llk(yields_o: np.ndarray, W: np.ndarray, K1Q_X: np.ndarray,
            Sigma_P: np.ndarray, maturities: np.ndarray, dt: float,
            K0P_P: np.ndarray, sigma_e: float | None = None, 
            rank_rp: int | None = None, bias_adjusted: bool = False, apply_killian: bool = True) -> JSZLikelihoodResult:
    """
    Computes the JSZ log-likelihood (negative because we minimize).
    L_tot = L_Q + L_P, 
    L_Q = cross-sectional likelihood (how well the model fits the ILS)
    L_P = time-series likelihood (how well is fitted the estimated dynamics of the pricing portfolio)
    For a candidate set of Q-dynamics, factor covariance and P-intercept, we:
        1. construct the observable pricing portfolios
        2. build the implied affine Q yield loadings 
        3. concentrate out the canonical Q intercept parameter kinf
        4. compute model-implied fitted ILS rates and cross-sectional pricing errors
        5. concentrate out the measurement-error standard deviation sigma_e when it is not supplied
        6. construct the P-dynamics of the pricing portfolios
        7. compute P innovations and their time-series likelihood
        8. return the sum of the cross-sectional and time-series negative log-likelihood contributions.
    For the full-rank specification, K1P is obtained from an unrestricted VAR of the observable pricing portfolios. For reduced-rank
    specifications, it is obtained from reduced_rank_free_intercept_regress.
    Formulas ==> JSZ 2011

    Inputs:
    ----------
        :yields_o: matrix of observed ILS rates.
        :W: matrix of PCA pricing-portfolio weights
        :K1Q_X: candidate Q drift matrix in the canonical basis.
        :Sigma_P: candidate innovation covariance matrix of the observable pricing portfolios.
        :maturities: vector of ILS maturities expressed in years.
        :dt: Model period expressed in years, e.g. 1/12 for monthly dynamics.
        :K0P_P: N-vector of physical-measure factor-dynamics intercepts.
        :sigma_e: Optional standard deviation of cross-sectional pricing errors. If None, it is concentrated out analytically.
        :rank_rp: Rank restriction used for the market-price-of-risk dynamics. If None, the full-rank case rank_rp = N is used.
        :bias_adjusted: if True, estimate full-rank P-dynamics adjusted for the persistence bias
        :apply_killian: if False, doesn't apply the Killian shrinkage. Useful in expanding setting because for some dates (2022)
                                it is actually impossible to get eigenvalues <1.
    
    Output:
    ----------
        JSZLikelihoodResult dataclass with Q-dynamics and short-rate coefficients expressed in the observable pricing-portfolio basis, 
        alongside per-period total, cross-sectional and time-series negative log-likelihood contributions, 
        affine loadings, P- and Q-dynamics, pricing portfolios, kinf, and sigma_e (as encapsulated in JSZLikelihoodResult).
    """
    # Imports 
    yields_o = np.asarray(yields_o, dtype=float)
    W = np.asarray(W, dtype=float)
    Sigma_P = np.asarray(Sigma_P, dtype=float)
    K0P_P = np.asarray(K0P_P, dtype=float).reshape(-1)
    T = yields_o.shape[0] - 1
    J = yields_o.shape[1]
    N = W.shape[0]
    if rank_rp is None:
        rank_rp = N
    P = yields_o @ W.T

    #  1. Q LIKELIHOOD
    # First get the implied Q loadings (in their form with kinf)
    loadings0 = compute_loadings_orchestrator(W=W, K1Q_X=K1Q_X, rho0_P=0.0, Sigma_P=Sigma_P, maturities=maturities, dt=dt)
    # JSZ: since P = Wy are the exact factors used by the model, pricing errors can only be on directions of yields orthogonal to W
    V = null_space(W).T # basis orthogonal to W
    P_orth = V.T @ V    # projection matrix on V = Null(W). Any pricing error is in this space

    # Second we concentrate out kinf
    # Because y = A_P + B_P' P + eps and A_P = alpha0_Pkinf + alpha1_P, we have on average
    # y_bar = alpha0_P*kinf + alpha1_P + B_P' P_bar + eps_bar
    # Then let r = y_bar - alpha1_P - B_P' P = alpha0_P*kinf + eps_bar
    # We only consider the part orthogonal to W: P_orth*r = P_orth*alpha0_P*kinf + P_orth*eps_bar
    # Minimizing pricing errors is thus the same as minimizing P_orth*eps_bar (again JSZ argue that the pricing error can only be orthogonal to W)
    # which is basically an OLS: we search kinf minimising ||P_orth*(r - alpha0_P*kinf)||_2
    # Analytical solution: kinf = [r'P_orth alpha0_P]/[alpha0_P'P_orth alpha0_P]
    mean_yields = yields_o[1:].mean(axis=0)
    mean_P = P[1:].mean(axis=0)
    residual_mean = mean_yields - loadings0.alpha1_P - loadings0.B_P.T @ mean_P
    alpha0 = loadings0.alpha0_P
    numerator = residual_mean @ (P_orth @ alpha0)
    denominator = alpha0 @ (P_orth @ alpha0)
    kinf = float(numerator / denominator)

    # Now that we have the true kinf, we can get the yield loadings (canonical) 
    A_X = loadings0.alpha0_X * kinf + loadings0.alpha1_X
    A_P = loadings0.alpha0_P * kinf + loadings0.alpha1_P
    K0Q_X = np.zeros(N)
    K0Q_X[loadings0.m1 - 1] = kinf      # K0Q_X = kinf e_m1

    # We proceed to change coordinates back to the observable basis for all parameters
    rho0_X = 0.0
    rho1_X = np.ones(N)
    K0Q_P, K1Q_P, rho0_P, rho1_P = rotation_canonical_to_observable(W=W,
        K1Q_X=loadings0.K1Q_X, K0Q_X=K0Q_X, rho1_X=rho1_X, rho0_X=rho0_X,
        B_X=loadings0.B_X, A_X=A_X)

    # Now fitting errors = observed yields - fitted yields
    yields_m = A_P + P @ loadings0.B_P
    yield_errors = yields_o[1:] - yields_m[1:]
    square_errors = yield_errors ** 2

    # Hypothesis: cross-sectional errors in non-priced directions are gaussian,
    # e_orth ~ N(0, sigma^2_e I_{J-N}),
    # Then we can estimate the parameter sigma using standard estimate formula for the variance
    if sigma_e is None:
        denominator_sigma = T * (J - N)
        if denominator_sigma <= 0:
            raise ValueError("Need J > N to estimate sigma_e.")
        sigma_e = float(np.sqrt(square_errors.sum() / denominator_sigma))

    if not np.isfinite(sigma_e) or sigma_e <= 0:
        raise FloatingPointError("sigma_e is non-positive or non-finite.")

    # Final Q-log-likelihood formula
    llkQ = 0.5 * square_errors.sum(axis=1) / sigma_e**2 + (J - N) * 0.5 * np.log(2 * np.pi) + 0.5 * (J - N) * np.log(sigma_e**2)

    # 2. P LIKELIHOOD
    # If full rank, P-dynamics = standard VAR estimation
    if rank_rp == N:
        if bias_adjusted:
            K1PplusI, _, _ = estimate_var_bias_adjusted_kilian(P, Sigma_P, apply_killian)
        else:
            K1PplusI, _, _ = estimate_var(P)
        K1P_P = K1PplusI - np.eye(N)
    # If reduced rank, estimate K1P-K1Q
    else:
        Y1 = np.diff(P, axis=0) - P[:-1] @ K1Q_P.T
        X = P[:-1]
        beta1, alpha1, _, _ = reduced_rank_free_intercept_regress(Y1, X, rank=rank_rp, Omega=Sigma_P)
        K1P_P = beta1.T + K1Q_P
        K0P_P = alpha1

    # eps_P = P_t+1 - K0_P - (I+K1P) P_t
    innovations = P[1:].T - (K0P_P[:, None] + (np.eye(N) + K1P_P) @ P[:-1].T)
    
    sign, logdet = np.linalg.slogdet(Sigma_P) # = log(|Sigma_P|)
    if sign <= 0:
        raise np.linalg.LinAlgError("Sigma_P must be positive definite.")
    solved_innovations = np.linalg.solve(Sigma_P, innovations)
    llkP = 0.5 * N * np.log(2 * np.pi) + 0.5 * logdet + 0.5 * np.sum(innovations * solved_innovations, axis=0)
    
    # 3. TOTAL LIKELIHOOD
    llk = llkQ + llkP

    return JSZLikelihoodResult(llk=llk, A_P=A_P, B_P=loadings0.B_P, A_X=A_X, B_X=loadings0.B_X, kinf=kinf,
        K0P_P=K0P_P, K1P_P=K1P_P, sigma_e=sigma_e, K0Q_P=K0Q_P,
        K1Q_P=K1Q_P, rho0_P=rho0_P, rho1_P=rho1_P, P=P,
        llkP=llkP, llkQ=llkQ, K0Q_X=K0Q_X, K1Q_X=loadings0.K1Q_X,
        rho0_X=rho0_X, rho1_X=rho1_X)
