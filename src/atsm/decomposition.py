################################################
# decomposition.py
# Applies an estimated ATSM to observed ILS curves and decomposes inflation compensation into expected inflation and inflation risk premia.
#
# - decompose_ils: computes fitted spot ILS rates under Q, expected inflation under P, and the inflation risk premium as their difference; 
#                  optionally computes the corresponding forward-rate decompositions.
# - spot_decomposition_frame: converts the spot decomposition into a long-format DataFrame.
# - forward_decomposition_frame: converts the forward decomposition into a long-format DataFrame.
################################################

import numpy as np
import pandas as pd

from .recursion import compute_ab
from .model import ATSMDecomposition, ATSMModel


def decompose_ils(model: ATSMModel, yields: np.ndarray, compute_forwards: bool = True, compute_5y5y: bool = True) -> ATSMDecomposition:
    """
    Applies an estimated ATSM to an ILS panel with the same maturities, provides the fitted and decomposition for spots and 
    if wanted forward rates.
    Forward rates are computed using the continuous-compounding forward-rate convention:
        f_{n,m} = [(n+m) z_{n+m} - n z_n] / m
    where f_{n,m} is the m-year forward rate starting in n years and z_n is the n-year zero-coupon ILS rate.
    For example: f_{5,5} = (10 z_10 - 5 z_5) / 5 = 2 z_10 - z_5
    The observable pricing portfolios are always P_t = y_t @ W.T using the fixed monthly PCA weights stored in the model.

    Inputs:
    -----------
        :model: ATSMModel Dataclass containing the estimated parameters of the ATSM.
        :yields: ILS rates
        :compute_forwards: if True, compute forwards 1Y1Y to 1Y9Y (and their associated decomposition)
        :compute_5y5y: if True AND compute_forwards is also True, compute the forward 5Y5Y (and its associated decomposition)     
                
    Output:
    ----------
        ATSMDecomposition dataclass containing the observed, fitted, expected ILS rates, IRP, yield loadings under P and Q.
        If compute_forwards is True, contains the same elements but for forward rates as well.
    """
    # Import + standard checks
    yields = np.asarray(yields, dtype=float) 
    maturities = np.asarray(model.maturities, dtype=float)
    if yields.ndim != 2 or yields.shape[1] != len(maturities):
        raise ValueError("yields must have the same maturity dimension as the estimated model.")
    N = model.W.shape[0]
    P = yields @ model.W.T   # pricing portfolios

    period_maturities = np.rint(maturities * 12).astype(int)

    # We get fitted ILS rates with Q-yield loadings
    # We get expected average future inflation with P-"yield loadings" (covariance is set to 0 as we consider expectations and 
    # not a genuine risk-adjusted bond-pricing object)
    # IRP = difference between both.
    A_Q, B_Q = compute_ab(maturities=period_maturities,
        rho0_P=model.rho0_P * (1.0 / 12.0), rho1_P=model.rho1_P * (1.0 / 12.0),
        muQ=model.K0Q_P, PhiQ= np.eye(N) + model.K1Q_P, Sigma_P=model.Sigma_P)
    A_Q = A_Q * 12.0
    B_Q = B_Q * 12.0

    fitted = A_Q + P @ B_Q

    A_P, B_P = compute_ab(maturities=period_maturities,
        rho0_P=model.rho0_P * (1.0 / 12.0), rho1_P=model.rho1_P * (1.0 / 12.0),
        muQ=model.K0P_P, PhiQ=np.eye(N) + model.K1P_P, Sigma_P=np.zeros((N, N)))
    A_P = A_P * 12.0
    B_P = B_P * 12.0

    expected = A_P + P @ B_P
    premium = fitted - expected

    # Initialisation for the forwards
    forward_observed = None
    forward_fitted = None
    forward_expected = None
    forward_premium = None
    labels = None

    if compute_forwards:
        # To compute forwards, we apply the formula above
        maturity_to_index = {int(m): i for i, m in enumerate(maturities)}
        consecutive = all(i in maturity_to_index for i in range(1, 11))
        if consecutive:
            labels = [f"1y{i}y" for i in range(1, 10)]
            forward_observed = np.full((len(yields), 9), np.nan)
            forward_fitted = np.full((len(yields), 9), np.nan)
            forward_expected = np.full((len(yields), 9), np.nan)

            for loop in range(1, 10):
                i0 = maturity_to_index[loop]
                i1 = maturity_to_index[loop + 1]
                forward_observed[:, loop - 1] = yields[:, i1] * (loop + 1) - yields[:, i0] * loop
                forward_fitted[:, loop - 1] = fitted[:, i1] * (loop + 1) - fitted[:, i0] * loop
                forward_expected[:, loop - 1] = expected[:, i1] * (loop + 1) - expected[:, i0] * loop

            if compute_5y5y and 5 in maturity_to_index and 10 in maturity_to_index:
                i5 = maturity_to_index[5]
                i10 = maturity_to_index[10]
                data_5y5y = 2.0 * yields[:, i10] - yields[:, i5]
                fitted_5y5y = 2.0 * fitted[:, i10] - fitted[:, i5]
                expected_5y5y = 2.0 * expected[:, i10] - expected[:, i5]

                forward_observed = np.column_stack([forward_observed, data_5y5y])
                forward_fitted = np.column_stack([forward_fitted, fitted_5y5y])
                forward_expected = np.column_stack([forward_expected, expected_5y5y])
                labels.append("5y5y")
            forward_premium = forward_fitted - forward_expected

    return ATSMDecomposition(observed=yields, fitted=fitted, expected=expected, premium=premium,
        A_Q=A_Q, B_Q=B_Q, A_P=A_P, B_P=B_P,
        forward_observed=forward_observed, forward_fitted=forward_fitted, forward_expected=forward_expected,
        forward_premium=forward_premium, forward_labels=labels)


def spot_decomposition_frame(dates: pd.DatetimeIndex, maturities: np.ndarray, decomposition: ATSMDecomposition) -> pd.DataFrame:
    """
    Helper function returning a date x maturity decomposition dataframe for spot ILS.
    """
    records = []
    for j, maturity in enumerate(maturities):
        block = pd.DataFrame(
            {
                "date": dates,
                "maturity": int(maturity),
                "observed": decomposition.observed[:, j],
                "fitted": decomposition.fitted[:, j],
                "expected": decomposition.expected[:, j],
                "irp": decomposition.premium[:, j],
            }
        )
        records.append(block)

    return pd.concat(records, ignore_index=True)


def forward_decomposition_frame(dates: pd.DatetimeIndex, decomposition: ATSMDecomposition) -> pd.DataFrame | None:
    """
    Helper function returning a date x maturity decomposition dataframe for forward ILS.
    """
    if decomposition.forward_labels is None:
        return None

    records = []
    for j, label in enumerate(decomposition.forward_labels):
        block = pd.DataFrame(
            {
                "date": dates,
                "forward": label,
                "observed": decomposition.forward_observed[:, j],
                "fitted": decomposition.forward_fitted[:, j],
                "expected": decomposition.forward_expected[:, j],
                "irp": decomposition.forward_premium[:, j],
            }
        )
        records.append(block)

    return pd.concat(records, ignore_index=True)
