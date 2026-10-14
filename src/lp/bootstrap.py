################################################
# bootstrap.py
# Two-stage bootstrap for LP based on ATSM-generated outcomes.
# Outer bootstrap: residual bootstrap, we first resample ATSM VAR under P innovations, then generate a new factors path 
# under the estimated P-dynamics according to these resampled errors, then we can resample pricing errors and construct 
# pseudo ILS panel, on which we estimate a new ATSM and refilter.
# Inner bootstrap: CONDITIONAL ON the ATSM bootstrap sample, we draw LP coefficients from the NW asymptotic distribution

################################################

from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
import numpy as np
import pandas as pd

from atsm.apply_decomp import average_decomposition_frames, decompose_model
from atsm.estimation import estimate_atsm
from atsm.model import ATSMModel
from atsm.pricing_portfolios import compute_pricing_portfolios
from lp.local_projection import estimate_lp_irfs


@dataclass(slots=True)
class LPBootstrapSpec:
    """
    Definition of one LP exercise run inside every ATSM bootstrap replication.
    """
    name: str
    curve: str
    shocks: pd.DataFrame
    shock_cols: list[str]
    curve_col: str

@dataclass(slots=True)
class LPBootstrapStorage:
    """
    Outer-bootstrap LP coefficients and covariance matrices for one LP exercise.
    """
    curve_points: list
    shock_names: list[str]
    horizons: np.ndarray
    betas: dict[str, np.ndarray]
    covariances: dict[str, np.ndarray]

def _pricing_error_projection(W: np.ndarray) -> np.ndarray:
    """
    Projection matrix onto the subspace orthogonal to the pricing portfolios.
    """
    W = np.asarray(W, dtype=float)
    gram = W @ W.T
    return np.eye(W.shape[1]) - W.T @ np.linalg.solve(gram, W)

def generate_atsm_bootstrap_sample(yields: np.ndarray, model: ATSMModel, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one pseudo monthly ILS panel using an ATSM residual bootstrap.
    Because we resample innovations as complete vectors, we preserve their contemporaneous covariance structure.
    Also we project pricing errors onto the subspace orthogonal to the pricing portfolios, ensuring the pricing errors
    remain orthogonal to the pricing portfolios as in JSZ spirit

    Inputs:
    ----------
        :yields: original monthly ILS panel.
        :model: estimated ATSM used as the bootstrap data-generating process.
        :rng: random-number generator.

    Output:
    ----------
        Tuple (pseudo_yields, pseudo_factors).
    """
    yields = np.asarray(yields, dtype=float)
    # pricing portfolios = PCA loadings * ILS rates
    W = np.asarray(model.W, dtype=float)
    P = yields @ W.T
    if yields.ndim != 2 or yields.shape[1] != W.shape[1]:
        raise ValueError("Monthly yields and model W have incompatible dimensions.")
    N = W.shape[0]
    T = len(yields)
    # See code atsm: we distinguish between K1 the mean reverting matrix and I + K1 the transition one
    transition = np.eye(N) + model.K1P_P

    # Under P, P_t = K0P + (I + K1P) P_{t-1} + u_t so we get u_t easily
    innovations = P[1:] - (model.K0P_P + P[:-1] @ transition.T)
    innovations = innovations - innovations.mean(axis=0, keepdims=True)
    # Sampling WITH replacement of the innovations
    innovation_draws = innovations[rng.integers(0, len(innovations), size=T - 1)]

    P_bootstrap = np.empty_like(P)
    P_bootstrap[0] = P[0]
    # New P factors generated via bootstrap
    for t in range(1, T):
        P_bootstrap[t] = model.K0P_P + transition @ P_bootstrap[t - 1] + innovation_draws[t - 1]

    # y_t = A + B'P_t + e_t. Draw whole pricing-error vectors so their
    # cross-sectional dependence is retained.
    fitted_original = model.AP + P @ model.BP
    pricing_errors = yields - fitted_original
    projection = _pricing_error_projection(W)
    pricing_errors = pricing_errors @ projection
    pricing_errors = pricing_errors - pricing_errors.mean(axis=0, keepdims=True)

    # Same logic: bootstraped yields = ones generated from the fitted values plus resampled pricing errors
    error_draws = pricing_errors[rng.integers(0, len(pricing_errors), size=T)]
    fitted_bootstrap = model.AP + P_bootstrap @ model.BP
    yields_bootstrap = fitted_bootstrap + error_draws

    # We check that the pseudo yields reproduce the generated factors as they should
    implied_factors = yields_bootstrap @ W.T
    if not np.allclose(implied_factors, P_bootstrap, atol=1e-10, rtol=1e-8):
        raise FloatingPointError("Bootstrap pseudo yields do not reproduce the generated pricing portfolios.")
    if not np.isfinite(yields_bootstrap).all():
        raise FloatingPointError("Bootstrap pseudo yields contain non-finite values.")
    return yields_bootstrap, P_bootstrap

def _align_pca_weights(W: np.ndarray, reference_W: np.ndarray) -> np.ndarray:
    """
    This is only used to align PCA factor ordering and signs when PCA weights are explicitly re-estimated in the
    bootstrap. The baseline bootstrap keeps W fixed, as in the ATSM estimation
    and filtering basis used for the main decomposition.
    """
    W = np.asarray(W, dtype=float)
    reference_W = np.asarray(reference_W, dtype=float)
    if W.shape != reference_W.shape:
        raise ValueError("W and reference_W must have the same shape.")
    n_factors = W.shape[0]
    best_permutation = None
    best_score = -np.inf
    for permutation in permutations(range(n_factors)):
        score = sum(abs(float(W[permutation[i]] @ reference_W[i])) for i in range(n_factors))
        if score > best_score:
            best_score = score
            best_permutation = permutation
    aligned = np.vstack([W[best_permutation[i]] for i in range(n_factors)])
    for i in range(n_factors):
        if float(aligned[i] @ reference_W[i]) < 0:
            aligned[i] *= -1.0
    return aligned

def _bootstrap_atsm_config(atsm_config: dict, n_random_seeds: int | None = None, n_repeats: int | None = None, maxiter: int | None = None) -> dict:
    """
    Copy the ATSM configuration
    """
    config = deepcopy(atsm_config)
    estimation = config["estimation"]
    if n_random_seeds is not None:
        estimation["n_random_seeds"] = int(n_random_seeds)
    if n_repeats is not None:
        estimation["n_repeats"] = int(n_repeats)
    if maxiter is not None:
        estimation["maxiter"] = int(maxiter)
    return config

def estimate_bootstrap_models(pseudo_yields: np.ndarray, maturities: np.ndarray, reference_W: np.ndarray, atsm_config: dict,
                             random_seed: int, decomposition_model: str = "average", reestimate_pca: bool = False, verbose: bool = False) -> tuple[ATSMModel | None, ATSMModel | None]:
    """
    Re-estimate the standard and/or bias-adjusted ATSM on a boostrap-obtained ILS sample.

    Inputs
    -----------
        :pseudo_yields: : pseudo ILS yields for the bootstrap sample.
        :maturities: maturities corresponding to the pseudo-obtained yields.
        :reference_W: reference PCA weights to align the bootstrap sample factors. Either serves as the W we'll keep fixed or as the one we need to align to (order/signs)
        :atsm_config: ATSM configuration dictionary.
        :random_seed: random seed
        :decomposition_model: Decomposition model to use ("standard", "bias_adjusted", or "average").
        :reestimate_pca: Whether to re-estimate the PCA factors for the bootstrap sample. Baseline = False
        :verbose: Whether to print logs during estimation.

    Output
    -----------
        Tuple containing the re-estimated standard and bias-adjusted ATSM models. 
    """
    if decomposition_model not in {"standard", "bias_adjusted", "average"}:
        raise ValueError("decomposition_model must be standard, bias_adjusted or average.")
    # If we want to reestimate PCA on the boostrap sample, we need to recompute pricing portfolios and align the resulting PCA weights with the reference W
    if reestimate_pca:
        factors = compute_pricing_portfolios(yields=pseudo_yields, n_factors=int(atsm_config["n_factors"]), factor_signs=None)
        W = _align_pca_weights(factors.W, reference_W)
    else:
        W = np.asarray(reference_W, dtype=float).copy()

    model_standard = None
    model_bias = None
    if decomposition_model in {"standard", "average"}:
        model_standard = estimate_atsm(yields=pseudo_yields, W=W, maturities=maturities, dt=float(atsm_config["dt"]),
                                       config=atsm_config, random_seed=random_seed, verbose=verbose, bias_adjusted=False)
    if decomposition_model in {"bias_adjusted", "average"}:
        model_bias = estimate_atsm(yields=pseudo_yields, W=W, maturities=maturities, dt=float(atsm_config["dt"]), config=atsm_config,
                                   random_seed=random_seed + 1, verbose=verbose, bias_adjusted=True)
    return model_standard, model_bias


def decompose_daily_bootstrap(model_standard: ATSMModel | None, model_bias: ATSMModel | None, daily_yields: np.ndarray, daily_dates: pd.DatetimeIndex,
                              maturities: np.ndarray, decomposition_config: dict, decomposition_model: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Apply bootstrap ATSM parameters to the original daily ILS observations. Because the daily rates are the same across samples,
    we just propagate uncertainty in the decomposition itself, not in the daily yields

    Inputs
    ---------
        :model_standard: Standard ATSM model estimated from the bootstrap sample.
        :model_bias: Bias-adjusted ATSM model estimated from the bootstrap sample.
        :daily_yields: daily ILS yields.
        :daily_dates: Dates corresponding to the daily yields.
        :maturities: Maturities for the yields.
        :decomposition_config: Configuration dictionary for the decomposition.
        :decomposition_model: Specifies which decomposition to use: "standard", "bias_adjusted", or "average".

    Output
    -------
        Tuple containing the spot and forward decompositions. 
    """
    # If we ask for a specific model decomposition 
    if decomposition_model == "standard":
        return decompose_model(model=model_standard, yields=daily_yields, dates=daily_dates, maturities=maturities, dec_cfg=decomposition_config)
    if decomposition_model == "bias_adjusted":
        return decompose_model(model=model_bias, yields=daily_yields, dates=daily_dates, maturities=maturities, dec_cfg=decomposition_config)
    # Otherwise we use both and also compute the 50/50
    spot_standard, forward_standard = decompose_model(model=model_standard, yields=daily_yields, dates=daily_dates, maturities=maturities, dec_cfg=decomposition_config)
    spot_bias, forward_bias = decompose_model(model=model_bias, yields=daily_yields, dates=daily_dates, maturities=maturities, dec_cfg=decomposition_config)
    spot_average = average_decomposition_frames(spot_standard, spot_bias, keys=["date", "maturity"])
    if forward_standard is not None and forward_bias is not None:
        forward_average = average_decomposition_frames(forward_standard, forward_bias, keys=["date", "forward"])
    else:
        forward_average = None
    return spot_average, forward_average


def _initialize_storage(point_results: dict[str, pd.DataFrame], shock_names: list[str], n_bootstrap: int) -> LPBootstrapStorage:
    """
    Helper to locate compact arrays for outer-bootstrap LP estimates.
    """
    components = ["IC", "EC", "IRP"]
    reference = point_results["IC"]
    curve_points = reference.index.tolist()
    horizons = np.array(sorted(reference.columns.get_level_values("horizon").unique()), dtype=int)
    n_curve = len(curve_points)
    n_horizons = len(horizons)
    n_shocks = len(shock_names)
    betas = {component: np.full((n_bootstrap, n_curve, n_horizons, n_shocks), np.nan, dtype=float) 
                for component in components}
    covariances = {component: np.full((n_bootstrap, n_curve, n_horizons, n_shocks, n_shocks), np.nan, dtype=float)
                  for component in components}

    return LPBootstrapStorage(curve_points=curve_points, shock_names=list(shock_names), horizons=horizons, betas=betas, covariances=covariances)


def _store_lp_replication(storage: LPBootstrapStorage, bootstrap_index: int, results: dict[str, pd.DataFrame], covariance_matrices: dict[tuple[str, float, int], np.ndarray]) -> None:
    """
    Helper that stores one outer-bootstrap LP replication sample in preallocated arrays
    """
    desired_columns = pd.MultiIndex.from_product([storage.shock_names, storage.horizons], names=["shock", "horizon"])
    for component in ["IC", "EC", "IRP"]:
        beta = results[component].xs("beta", axis=1, level="statistic")
        beta = beta.reindex(index=storage.curve_points, columns=desired_columns)
        values = beta.to_numpy(dtype=float).reshape(len(storage.curve_points), len(storage.shock_names), len(storage.horizons))
        storage.betas[component][bootstrap_index] = values.transpose(0, 2, 1)
        for curve_idx, curve_point in enumerate(storage.curve_points):
            for horizon_idx, horizon in enumerate(storage.horizons):
                storage.covariances[component][bootstrap_index, curve_idx, horizon_idx] = covariance_matrices[(component, curve_point, int(horizon))]

def run_atsm_lp_bootstrap(monthly_yields: np.ndarray, daily_yields: np.ndarray, daily_dates: pd.DatetimeIndex, maturities: np.ndarray,
                          generator_model: ATSMModel, atsm_config: dict, decomposition_config: dict, point_results: dict[str, dict[str, pd.DataFrame]],
                          lp_specs: list[LPBootstrapSpec], n_bootstrap: int = 100, max_horizon: int = 25, ic_lags: int = 5, shock_lags: int = 0,
                          confidence_level: float = 0.90, nw_lags: int | str = "horizon", decomposition_model: str = "average", reestimate_pca: bool = False,
                          require_optimizer_success: bool = True, max_attempts: int | None = None, random_seed: int = 12345, atsm_n_random_seeds: int | None = None,
                          atsm_n_repeats: int | None = None, atsm_maxiter: int | None = None, verbose: bool = True) -> tuple[dict[str, LPBootstrapStorage], pd.DataFrame]:
    """
    Run the outer ATSM bootstrap and re-estimate all LP for all shocks that are provided. Basically we have 
    a signle ATSM (outer) bootstrap that is shared by all LP specifications which are themselves bootstraped (inner)

    Inputs
    ----------
        :monthly_yields: Matrix of monthly yields used for the outer ATSM bootstrap
        :daily_yields: Matrix of daily yields used for the filtering
        :daily_dates: Dates corresponding to the daily yields.
        :maturities: Maturities corresponding to the monthly yields.
        :generator_model: ATSM model used to generate the bootstrap samples.
        :atsm_config: Configuration dictionary for the ATSM estimation.
        :decomposition_config: Configuration dictionary for the decomposition of daily yields.
        :point_results: Dictionary containing the point estimates of the LPs for each specification.
        :lp_specs: List of LPBootstrapSpec objects defining the LP exercises.
        :n_bootstrap: Number of bootstrap replications.
        :max_horizon: Maximum horizon for the LPs.
        :ic_lags: Number of lags for the information criterion in the LPs.
        :shock_lags: Number of lags for the shocks in the LPs.
        :confidence_level: Confidence level for the bootstrap intervals.
        :nw_lags: Number of Newey-West lags to use.
        :decomposition_model: Decomposition model to use ("average" or other).
        :reestimate_pca: Whether to re-estimate the PCA in the decomposition.
        :require_optimizer_success: Whether to require optimizer success in the ATSM estimation.
        :max_attempts: Maximum number of attempts for the ATSM bootstrap.
        :random_seed: Random seed for reproducibility.
        :atsm_n_random_seeds: Number of random seeds for the ATSM estimation.
        :atsm_n_repeats: Number of repeats for the ATSM estimation.
        :atsm_maxiter: Maximum number of iterations for the ATSM estimation.
        :verbose: Whether to print progress messages.

    Output
    ----------
        Tuple containing the bootstrap results for each LP specification.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be strictly positive.")
    if max_attempts is None:
        max_attempts = max(n_bootstrap * 2, n_bootstrap + 10)
    if max_attempts < n_bootstrap:
        raise ValueError("max_attempts must be at least n_bootstrap.")

    # Extract info and create objects/storage
    bootstrap_config = _bootstrap_atsm_config(atsm_config=atsm_config, n_random_seeds=atsm_n_random_seeds, n_repeats=atsm_n_repeats, maxiter=atsm_maxiter)
    storage = {spec.name: _initialize_storage(point_results=point_results[spec.name], shock_names=spec.shock_cols, n_bootstrap=n_bootstrap)
                                              for spec in lp_specs}
    rng = np.random.default_rng(random_seed)
    diagnostics = []
    successful = 0
    attempt = 0

    # We distinguish between the number of successful bootstrap, that we want in the end,
    # and the number of attempts, which may be larger due to potential failures (ex: Killian procedure fails)
    # As long as max_attempts is not reached, we keep trying to get successful bootstrap samples
    while successful < n_bootstrap and attempt < max_attempts:
        attempt += 1
        draw_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        draw_rng = np.random.default_rng(draw_seed)
        if verbose:
            print("\n" + "=" * 70)
            print(f"ATSM BOOTSTRAP {successful + 1} / {n_bootstrap} - attempt {attempt}")
            print("=" * 70)
        try:
            # Step 1: generate bootstrap sample of pseudo-yields
            pseudo_yields, _ = generate_atsm_bootstrap_sample(yields=monthly_yields, model=generator_model, rng=draw_rng)
            
            # Step 2: estimate ATSM based on the pseudo-yields
            model_standard, model_bias = estimate_bootstrap_models(pseudo_yields=pseudo_yields, maturities=maturities,
                                            reference_W=generator_model.W, atsm_config=bootstrap_config, random_seed=draw_seed,
                                            decomposition_model=decomposition_model, reestimate_pca=reestimate_pca, verbose=False)
            models_to_check = [model for model in [model_standard, model_bias] if model is not None]
            if require_optimizer_success and not all(model.optimizer_success for model in models_to_check):
                raise RuntimeError("At least one bootstrap ATSM optimizer did not report success.")

            # Step 3: use the boostrap ATSM to decompose daily yields and forwards into expectations and IRP
            spot_decomposition, forward_decomposition = decompose_daily_bootstrap(model_standard=model_standard, model_bias=model_bias, daily_yields=daily_yields,
                                                            daily_dates=daily_dates, maturities=maturities, decomposition_config=decomposition_config, 
                                                            decomposition_model=decomposition_model)
            # Step 4: estimate LP IRFs based on the bootstrap decomposed yields and forwards for each shock specification
            for spec in lp_specs:
                if spec.curve == "spot":
                    decomposition = spot_decomposition
                elif spec.curve == "forward":
                    decomposition = forward_decomposition
                else:
                    raise ValueError("LPBootstrapSpec.curve must be spot or forward.")
                if decomposition is None:
                    raise ValueError(f"No decomposition available for LP specification {spec.name}.")
                lp_results, covariance_matrices = estimate_lp_irfs(decomposition=decomposition, shocks=spec.shocks, max_horizon=max_horizon,
                                                                    ic_lags=ic_lags, shock_lags=shock_lags, confidence_level=confidence_level,
                                                                    curve_col=spec.curve_col, shock_cols=spec.shock_cols, nw_lags=nw_lags,
                                                                    return_covariances=True)
                _store_lp_replication(storage=storage[spec.name], bootstrap_index=successful, results=lp_results, covariance_matrices=covariance_matrices)
            diagnostics.append({"attempt": attempt,
                        "bootstrap_replication": successful + 1,
                        "seed": draw_seed,
                        "success": True,
                        "error": None,
                        "standard_optimizer_success": None if model_standard is None else model_standard.optimizer_success,
                        "bias_optimizer_success": None if model_bias is None else model_bias.optimizer_success,
                        "standard_objective": np.nan if model_standard is None else model_standard.objective_value,
                        "bias_objective": np.nan if model_bias is None else model_bias.objective_value})
            successful += 1

        except Exception as exc:
            diagnostics.append({"attempt": attempt,
                        "bootstrap_replication": np.nan,
                        "seed": draw_seed,
                        "success": False,
                        "error": str(exc),
                        "standard_optimizer_success": np.nan,
                        "bias_optimizer_success": np.nan,
                        "standard_objective": np.nan,
                        "bias_objective": np.nan})
            if verbose:
                print(f"Bootstrap attempt failed: {exc}")

    if successful < n_bootstrap:
        raise RuntimeError(f"Only {successful} successful ATSM bootstrap replications were obtained after {attempt} attempts.")
    
    return storage, pd.DataFrame(diagnostics)


def _center_interval(lower: float, upper: float, point_estimate: float) -> tuple[float, float]:
    """
    Center bootstrap percentile interval so its midpoint equals the original point estimate
    """
    midpoint = 0.5 * (lower + upper)
    shift = point_estimate - midpoint
    return lower + shift, upper + shift

def combine_bootstrap_uncertainty(point_results: dict[str, pd.DataFrame], storage: LPBootstrapStorage, n_inner_draws: int = 100, 
                                  confidence_level: float = 0.90, random_seed: int = 12345, recenter: bool = True) -> dict[str, pd.DataFrame]:
    """
    Combination of the outer and inner bootstraps
    For each outer ATSM decomposition, we take Gaussian draws across the shock coefficients using the 
    corresponding NW covariance matrix
    Point estimates remain those from the original sample.

    Inputs
    ---------
        :point_results: Point estimates from the original sample.
        :storage: Contains outer bootstrap results.
        :n_inner_draws: Number of inner Gaussian draws per outer replication.
        :confidence_level: Confidence level
        :random_seed: Random seed for reproducibility.
        :recenter: Whether to recenter the intervals around the original point estimates.

    Returns
    ---------
        Combined point estimates and confidence intervals for each component.
    """
    if n_inner_draws <= 0:
        raise ValueError("n_inner_draws must be strictly positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    # Initialisation
    rng = np.random.default_rng(random_seed)
    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    output = {}
    n_bootstrap = storage.betas["IC"].shape[0]
    n_curve = len(storage.curve_points)
    n_horizons = len(storage.horizons)
    n_shocks = len(storage.shock_names)
    desired_beta_columns = pd.MultiIndex.from_product([storage.shock_names, storage.horizons], names=["shock", "horizon"])

    for component in ["IC", "EC", "IRP"]:
        # First we take the initial point estimate for the current component
        original = point_results[component].copy()
        beta_original = original.xs("beta", axis=1, level="statistic")
        beta_original = beta_original.reindex(index=storage.curve_points, columns=desired_beta_columns)
        beta_original_array = beta_original.to_numpy(dtype=float).reshape(n_curve, n_shocks, n_horizons).transpose(0, 2, 1)

        bootstrap_std = np.full((n_curve, n_horizons, n_shocks), np.nan)
        bootstrap_lower = np.full((n_curve, n_horizons, n_shocks), np.nan)
        bootstrap_upper = np.full((n_curve, n_horizons, n_shocks), np.nan)

        # loop on each maturity
        for curve_idx in range(n_curve):
            means = storage.betas[component][:, curve_idx, :, :]   # beta hat
            covariances = storage.covariances[component][:, curve_idx, :, :, :] # NW covariance matrices
            if not np.isfinite(means).all() or not np.isfinite(covariances).all():
                raise RuntimeError(f"Non-finite bootstrap estimates for {component}, curve point {storage.curve_points[curve_idx]}.")

            # We force the covariance matrices to be symmetric to avoid numerical imprecisions
            covariances = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))

            # How to get Gaussian draws with the desired covariance structure:
            # We draw from a standard Normal first, then transform it using the square root of the covariance matrix to obtain the desired correlation structure
            # For that we perform an eigen decomposition 
            # Sqrt(covariance) = eigenvectors * sqrt(diagonal matrix of eigenvalues) * eigenvectors'
            eigenvalues, eigenvectors = np.linalg.eigh(covariances)
            eigenvalues = np.clip(eigenvalues, 0.0, None)
            sqrt_covariances = np.einsum("...ij,...j,...kj->...ik", eigenvectors, np.sqrt(eigenvalues), eigenvectors)
            # Then our Gaussian draws are obtained by multiplying the independent N(0,I) draws by the square root of the covariance matrix
            # eps = sqrt_covariances @ z,     z ~ N(0, I) so V(eps) = sqrt_covariances @ I @ sqrt_covariances' = covariances

            standard_normals = rng.standard_normal(size=(n_bootstrap, n_inner_draws, n_horizons, n_shocks))
            noise = np.einsum("brhj,bhij->brhi", standard_normals, sqrt_covariances)

            # We add the noise to the outer bootstrap coefs and compute CI
            draws = means[:, None, :, :] + noise
            pooled = draws.reshape(n_bootstrap * n_inner_draws, n_horizons, n_shocks)
            lower = np.quantile(pooled, lower_q, axis=0)
            upper = np.quantile(pooled, upper_q, axis=0)

            if recenter:
                midpoint = 0.5 * (lower + upper)
                shift = beta_original_array[curve_idx] - midpoint
                lower = lower + shift
                upper = upper + shift

            bootstrap_std[curve_idx] = np.std(pooled, axis=0, ddof=1)
            bootstrap_lower[curve_idx] = lower
            bootstrap_upper[curve_idx] = upper

        # Save results
        new_frames = []
        for statistic, values in [("bootstrap_std_error", bootstrap_std), ("bootstrap_ci_lower", bootstrap_lower), ("bootstrap_ci_upper", bootstrap_upper)]:
            # Convert M x H x S into M x (shock, horizon).
            matrix = values.transpose(0, 2, 1).reshape(n_curve, n_shocks * n_horizons)
            columns = pd.MultiIndex.from_tuples([(shock, int(horizon), statistic) for shock in storage.shock_names
                                    for horizon in storage.horizons], names=["shock", "horizon", "statistic"])
            new_frames.append(pd.DataFrame(matrix, index=original.index, columns=columns))

        original_statistics = list(dict.fromkeys(point_results[component].columns.get_level_values("statistic")))
        final_statistics = original_statistics + ["bootstrap_std_error", "bootstrap_ci_lower", "bootstrap_ci_upper"]
        desired_columns = pd.MultiIndex.from_tuples([(shock, int(horizon), statistic) for shock in storage.shock_names
                                    for horizon in storage.horizons
                                    for statistic in final_statistics], names=["shock", "horizon", "statistic"])
        original = pd.concat([original] + new_frames, axis=1)
        original = original.reindex(columns=desired_columns)
        output[component] = original
    return output
