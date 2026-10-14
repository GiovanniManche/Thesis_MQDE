################################################
# pricing_portfolios.py
# Constructs the observable JSZ pricing portfolios from the ILS curve using principal component loadings.
#
# - compute_pricing_portfolios: estimates PCA loading weights W from ILS rates and constructs the observable pricing portfolios P_t = W y_t.
################################################

import numpy as np
from sklearn.decomposition import PCA
from .model import ILSFactors


def compute_pricing_portfolios(yields: np.ndarray, n_factors: int = 3, factor_signs: list[float] | np.ndarray | None = None) -> ILSFactors:
    """
    Estimate PCA weights from ILS rates, then construct observable pricing portfolios as in JSZ: P = W* y_t

    Inputs:
    -----------
        :yields: ILS rates
        :n_factors: number of PCA factors to keep
        :factor_signs: principal components are defined up to a sign, so we can precise the signs we want for the factors     
            
    Output:
    ----------
        ILSFactors dataclass containing the PCA loadings, the pricing portfolios, the explained variance ratios and PCA sample mean.
    """
    # Import and standard checks
    yields = np.asarray(yields, dtype=float)
    if yields.ndim != 2:
        raise ValueError("yields must be a T x J matrix.")
    if n_factors > min(yields.shape):
        raise ValueError("n_factors exceeds the rank supported by the data.")

    # Run PCA
    pca = PCA(n_components=n_factors)
    pca.fit(yields)

    W = pca.components_.copy()  
    # Sign restrictions
    if factor_signs is not None:
        signs = np.asarray(factor_signs, dtype=float)
        if signs.shape != (n_factors,):
            raise ValueError("factor_signs must have one entry per factor.")
        if np.any(np.abs(signs) != 1):
            raise ValueError("factor_signs entries must be +1 or -1.")
        W = signs[:, None] * W

    P = yields @ W.T

    return ILSFactors(W=W, P=P, explained_variance_ratio=pca.explained_variance_ratio_.copy(), pca_mean=pca.mean_.copy())


# def align_weights_to_reference(
#     W: np.ndarray,
#     reference_W: np.ndarray,
# ) -> np.ndarray:
#     """
#     Align PCA signs to a reference loading matrix.

#     This is useful only for MATLAB/Python validation.  PCA directions are
#     otherwise identified only up to sign.
#     """
#     W = np.asarray(W, dtype=float).copy()
#     reference_W = np.asarray(reference_W, dtype=float)

#     if W.shape != reference_W.shape:
#         raise ValueError("W and reference_W must have the same shape.")

#     for i in range(W.shape[0]):
#         if float(W[i] @ reference_W[i]) < 0:
#             W[i] *= -1.0

#     return W
