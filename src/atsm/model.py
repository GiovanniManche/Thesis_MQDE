from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(slots=True)
class ILSFactors:
    """
    PCA weights and observable JSZ pricing portfolios (W*ILS).
    """
    W: np.ndarray    
    P: np.ndarray
    explained_variance_ratio: np.ndarray
    pca_mean: np.ndarray


@dataclass(slots=True)
class ATSMModel:
    """
    Estimated ATSM results.
    """
    W: np.ndarray
    maturities: np.ndarray
    dt: float
    # Coefficient matrices (observable base)
    K0P_P: np.ndarray
    K1P_P: np.ndarray
    K0Q_P: np.ndarray
    K1Q_P: np.ndarray
    # Covariance matrix
    Sigma_P: np.ndarray
    # Short term rate dynamics
    rho0_P: float
    rho1_P: np.ndarray
    # Recursive pricing equations coefficients
    AP: np.ndarray
    BP: np.ndarray
    AX: np.ndarray
    BX: np.ndarray
    # Coefficient matrices (canonical / normalized base)
    kinfQ: float
    K0Q_X: np.ndarray
    K1Q_X: np.ndarray
    rho0_X: float
    rho1_X: np.ndarray
    # Measurement-error standard deviation
    sigma_e: float
    # Total and decomposed likelihoods after optimization 
    llk_series: np.ndarray
    llkP: np.ndarray
    llkQ: np.ndarray
    # JSZ pricing portfolios
    P_monthly: np.ndarray
    # Optimizer info
    optimizer_success: bool
    optimizer_message: str
    objective_value: float
    optimizer_parameters: np.ndarray

    metadata: dict[str, Any]


@dataclass(slots=True)
class ATSMDecomposition:
    """
    Spot and forward ILS decomposition.
    """
    observed: np.ndarray
    fitted: np.ndarray
    expected: np.ndarray
    premium: np.ndarray
    A_Q: np.ndarray
    B_Q: np.ndarray
    A_P: np.ndarray
    B_P: np.ndarray
    forward_observed: np.ndarray | None = None
    forward_fitted: np.ndarray | None = None
    forward_expected: np.ndarray | None = None
    forward_premium: np.ndarray | None = None
    forward_labels: list[str] | None = None
