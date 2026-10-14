#################################
# mps.py
# Computes monetary policy surprise factors as in Altavilla et al. (2019) and Gürkaynak et al. (2005)
#   - load_altavilla_mps_relevant: extract inputs for Altavilla factors
#   - load_gss_mps_relevant: extract inputs for GSS Target and Path factors
#   - compute_PCA: perform PCA on high-frequency interest-rate changes
#   - rotate_altavilla: identify Altavilla factors through constrained rotation
#   - compute_factor_loadings: estimate diagnostic yield-curve loadings
#   - compute_mps_altavilla_orchestrator: compute Altavilla factors
#   - compute_gss_mps: compute GSS Target and Path factors
#   - export_mps: export MPS results to Excel
#################################

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class MPSResult:
    loadings: pd.DataFrame
    factors: pd.DataFrame
    explained_variance: pd.Series


def load_altavilla_mps_relevant(file: str, release: str, conference: str):
    """
    Load the Altavilla database relevant inputs.

    Inputs:
    -----------
        :file: Path to the Altavilla database file.
        :release: Release Window sheet.
        :conference: Press Conference Window sheet.

    Output:
    ----------
        Dataframes containing the relevant Altavilla data.
    """
    # Imports and check date coherence
    release_data = pd.read_excel(file, sheet_name=release, index_col="date")
    conference_data = pd.read_excel(file, sheet_name=conference, index_col="date")
    release_data.index = pd.to_datetime(release_data.index)
    conference_data.index = pd.to_datetime(conference_data.index)
    if len(release_data) != len(conference_data):
        raise ValueError("Release and conference windows do not have the same number of observations.")
    conference_data.index = release_data.index.copy()

    # As in Altavilla 2019, we replace OIS 2/5/10Y by corresponding German yields when OIS missing
    for data in [release_data, conference_data]:
        data["OIS_2Y"] = data["OIS_2Y"].fillna(data["DE2Y"])
        data["OIS_5Y"] = data["OIS_5Y"].fillna(data["DE5Y"])
        data["OIS_10Y"] = data["OIS_10Y"].fillna(data["DE10Y"])

    # We take values from 2002 onwards
    release_data = release_data.loc[release_data.index >= pd.Timestamp("2001-12-31")].copy()
    conference_data = conference_data.loc[conference_data.index >= pd.Timestamp("2001-12-31")].copy()

    excluded_dates = pd.to_datetime(["2001-09-17", "2008-10-08", "2008-11-06"])
    release_data = release_data.loc[~release_data.index.isin(excluded_dates)].copy()
    conference_data = conference_data.loc[~conference_data.index.isin(excluded_dates)].copy()
    relevant_columns = ["OIS_1M", "OIS_3M", "OIS_6M", "OIS_1Y", "OIS_2Y", "OIS_5Y", "OIS_10Y"]

    # We remove completely empty event windows
    release_data = release_data.dropna(subset=relevant_columns, how="all").copy()
    conference_data = conference_data.dropna(subset=relevant_columns, how="all").copy()
    # Check that no partial missing observations remain
    for name, data in [("release", release_data), ("conference", conference_data)]:
        missing = data[relevant_columns].isna().any(axis=1)
        if missing.any():
            raise ValueError(f"Remaining missing values in {name} window at indices: {data.index[missing].tolist()}")
    return release_data[relevant_columns], conference_data[relevant_columns]

def load_gss_mps_relevant(file: str, release: str):
    """
    Load the relevant press-release-window inputs for the GSS Target and Path factor estimation.

    Inputs:
    -----------
        :file: Path to the EA-MPD database file.
        :release: Press Release Window sheet.

    Output:
    ----------
        DataFrame containing OIS_1M, OIS_3M, OIS_6M and OIS_1Y.
    """

    release_data = pd.read_excel(file, sheet_name=release, index_col="date")
    release_data.index = pd.to_datetime(release_data.index)
    relevant_columns = ["OIS_1M", "OIS_3M", "OIS_6M", "OIS_1Y"]
    for data in [release_data]:
        data["OIS_2Y"] = data["OIS_2Y"].fillna(data["DE2Y"])
        data["OIS_5Y"] = data["OIS_5Y"].fillna(data["DE5Y"])
        data["OIS_10Y"] = data["OIS_10Y"].fillna(data["DE10Y"])

    
    # We take values from 2002 onwards
    release_data = release_data.loc[release_data.index >= pd.Timestamp("2001-12-31")].copy()

    excluded_dates = pd.to_datetime(["2001-09-17", "2008-10-08", "2008-11-06"])
    release_data = release_data.loc[~release_data.index.isin(excluded_dates)].copy()

    # We remove completely empty event windows
    release_data = release_data.dropna(subset=relevant_columns, how="all").copy()
    missing = release_data[relevant_columns].isna().any(axis=1)

    if missing.any():
        raise ValueError(f"Remaining missing values in GSS inputs at indices: {release_data.index[missing].tolist()}")
    return release_data[relevant_columns]

def compute_PCA(X: pd.DataFrame, n_factors: int):
    """
    Does PCA

    Inputs:
    -----------
        :X: High-frequency OIS changes.
        :n_factors: Number of principal components.

    Output:
    ----------
        PCA loadings, PCA factors and explained variance shares.
    """
    pca = PCA(n_components=n_factors, svd_solver="full")
    factors = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=[f"PC{i+1}" for i in range(n_factors)])
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=factors.columns)
    explained_variance = pd.Series(pca.explained_variance_ratio_, index=factors.columns)
    return loadings, factors, explained_variance

def rotate_altavilla(data: pd.DataFrame, n_factors: int, window: str, crisis_date: str = "2008-09-04"):
    """
    Rotate the PCs according to the Altavilla et al. identification.
    How does it work:
        1. Perform PCA on the HF OIS changes.
        2. Normalize the factors to have unit variance.
        3. Find the rotation matrix that minimizes the pre-crisis variance of the third factor,
           imposes orthogonality, and ensures that factors 2 and 3 do not load on the 1-month OIS
        4. Scale the rotated factors such that Target has unit loading on the 1M, Timing on the 6M, FG on the 2Y, and QE on the 10Y

    Inputs:
    -----------
        :data: High-frequency OIS changes and dates.
        :n_factors: Number of PCs.
        :window: Either "release" or "conference".
        :crisis_date: Beginning of the crisis period used to identify the QE factor.

    Output:
    ----------
        Rotated and normalized MPS factors.
    """
    if n_factors != 3:
        raise ValueError("The Altavilla rotation requires exactly three principal components.")
    rate_columns = ["OIS_1M", "OIS_3M", "OIS_6M", "OIS_1Y", "OIS_2Y", "OIS_5Y", "OIS_10Y"]
    pca_loadings, pca_factors, explained_variance = compute_PCA(data[rate_columns], n_factors)

    # 1. Unit variance
    factor_scale = pca_factors.std(axis=0, ddof=1)
    factors = pca_factors.divide(factor_scale, axis=1).to_numpy()
    loadings = pca_loadings.multiply(factor_scale, axis=1).to_numpy()

    # 2. We need to find the rotation matrix that:
    #    - Minimize the pre-crisis variance of the third rotated factor (QE factor)
    #    - Impose orthogonality of the rotation matrix
    #    - Ensure that factors 2 and 3 do not load on the 1-month OIS
    pre_crisis = data.index < pd.Timestamp(crisis_date)
    factors_pre = factors[pre_crisis]

    def unpack_rotation(x):
        """
        Transform a flat array x into a 3x3 matrix
        """
        return np.asarray(x).reshape(3, 3, order="F")

    def objective(x):
        """
        Objective function to minimize the pre-crisis variance of the third rotated factor.
        """
        U = unpack_rotation(x)
        # Minimize the pre-crisis variance of the third rotated factor
        candidate_qe = factors_pre @ U[:, 2]
        return 0.5 * np.mean(candidate_qe ** 2)

    def restrictions(x):
        U = unpack_rotation(x)
        # Orthogonality restrictions
        UU = U @ U.T
        restrictions_orthogonal = np.array([
            UU[0, 0] - 1.0,
            UU[1, 1] - 1.0,
            UU[2, 2] - 1.0,
            UU[0, 1],
            UU[0, 2],
            UU[1, 2]])
        # Factors 2 and 3 must NOT load on the 1-month OIS
        restrictions_1m = np.array([
            loadings[0, :] @ U[:, 1],
            loadings[0, :] @ U[:, 2]])
        return np.concatenate([restrictions_orthogonal, restrictions_1m])

    # As a first candidate for our optimized rotation matrix we use the identity matrix
    x0 = np.eye(3).reshape(-1, order="F")
    result = minimize(objective, x0, method="SLSQP", constraints={"type": "eq", "fun": restrictions}, options={"ftol": 1e-12, "maxiter": 5000})
    if not result.success:
        raise RuntimeError(f"Altavilla rotation failed for {window}: {result.message}")
    # Optimized under constraint rotation matrix U
    U = unpack_rotation(result.x)
    # Factors * rotated matrix = rotated factors
    rotated = factors @ U

    if window == "release":
        rotated_factors = pd.DataFrame({"Target": rotated[:, 0]}, index=data.index)
        reference_rates = {"Target": "OIS_1M"}
    elif window == "conference":
        rotated_factors = pd.DataFrame({"Timing": rotated[:, 0], "FG": rotated[:, 1], "QE": rotated[:, 2]}, index=data.index)
        reference_rates = {"Timing": "OIS_6M", "FG": "OIS_2Y", "QE": "OIS_10Y"}
    else:
        raise ValueError("window must be either 'release' or 'conference'.")

    # Economic scaling:
    # Target -> unit loading on 1M
    # Timing -> unit loading on 6M
    # FG -> unit loading on 2Y
    # QE -> unit loading on 10Y
    for factor, rate in reference_rates.items():
        x = rotated_factors[factor].to_numpy()
        y = data[rate].to_numpy()
        X_ols = np.column_stack([np.ones(len(x)), x])
        scale = np.linalg.lstsq(X_ols, y, rcond=None)[0][1]
        rotated_factors[factor] = rotated_factors[factor] * scale

    return rotated_factors, explained_variance


def compute_factor_loadings(data: pd.DataFrame, factors: pd.DataFrame):
    """
    Estimate the yield-curve loadings of the final MPS factors. For each factor and OIS maturity, 
    it estimates a separate OLS regression of the OIS change on the factor.
    It's different from PCA loadings as it directly measures the impact of each factor on the observed OIS changes, 
    rather than capturing the directions of maximum variance.

    Inputs:
    -----------
        :data: High-frequency OIS changes.
        :factors: Final normalized monetary policy surprise factors.

    Output:
    ----------
        Dataframe of OLS coefficients, with OIS maturities in rows
        and monetary policy factors in columns.
    """

    rate_columns = data.columns.tolist()
    factor_columns = factors.columns.tolist()
    merged = data.join(factors, how="inner")

    loadings = pd.DataFrame(
        index=rate_columns,
        columns=factor_columns,
        dtype=float
    )

    for factor in factor_columns:
        x = merged[factor].to_numpy()
        X_ols = np.column_stack([np.ones(len(x)), x])

        for rate in rate_columns:
            y = merged[rate].to_numpy()

            loadings.loc[rate, factor] = np.linalg.lstsq(
                X_ols,
                y,
                rcond=None
            )[0][1]

    return loadings


def compute_mps_altavilla_orchestrator(file: str, release: str, conference: str, n_factors: int = 3):
    """
    Compute the Altavilla et al. monetary policy surprise factors.

    Inputs:
    -----------
        :file: Path to the Altavilla database file.
        :release: Release Window sheet.
        :conference: Press Conference Window sheet.
        :n_factors: Number of principal components used in the rotation.

    Output:
    ----------
        MPSResults dataclass containing MPS factors, yield-curve loadings and PCA explained variance.
    """

    release_data, conference_data = load_altavilla_mps_relevant(file, release, conference)

    release_factors, release_explained_variance = rotate_altavilla(release_data, n_factors, window="release")
    conference_factors, conference_explained_variance = rotate_altavilla(conference_data, n_factors, window="conference")
    release_loadings = compute_factor_loadings(release_data, release_factors)
    conference_loadings = compute_factor_loadings(conference_data, conference_factors)
    factors = release_factors.join(conference_factors, how="left")

    # Scale each factor by its event-date standard deviation for the LP analysis.
    # A one-unit shock then corresponds to a one-standard-deviation surprise.
    for factor in ["Target", "Timing", "FG", "QE"]:
        factors[f"{factor}_std"] = (factors[factor] / factors[factor].std(ddof=1))
    loadings = pd.concat([release_loadings, conference_loadings], axis=1)
    explained_variance = pd.concat([release_explained_variance.rename(lambda x: f"release_{x}"),
                                    conference_explained_variance.rename(lambda x: f"conference_{x}")])
    return MPSResult(loadings=loadings, factors=factors, explained_variance=explained_variance)


def compute_mps_gss_orchestrator(file: str, release: str):
    """
    Compute Target and Path factors as in GSS . Adaptation to EA follows Herbert et al. (2026)
    The procedure:
        1. Loads press-release-window changes in the 1M, 3M, 6M and 1Y OIS.
        2. Standardizes the four high-frequency interest-rate changes.
        3. Extracts the first two principal components.
        4. Rotates the PCs so that Target and Path remain orthogonal and
           Path has zero loading on the 1M OIS surprise.
        5. Scales Target to have unit loading on the 1M OIS.
        6. Scales Path so that Target and Path have equal loadings on the
           1Y OIS.
        7. Defines the aggregate monetary policy surprise as Target + Path.

    Inputs:
    -----------
        :file: Path to the EA-MPD database file.
        :release: Press Release Window sheet.

    Output:
    ----------
        MPSResult containing Target, Path, aggregate MPS, factor loadings
        and PCA explained variance shares.
    """

    X_raw = load_gss_mps_relevant(file, release)
    X_norm = (X_raw - X_raw.mean()) / X_raw.std(ddof=1)

    # We extract the first two PCs
    pca_loadings, pca_factors, explained_variance = compute_PCA(X_norm, n_factors=2)

    # Normalisation to unit variance
    pca_factors = (pca_factors - pca_factors.mean()) / pca_factors.std(ddof=1)
    F1 = pca_factors["PC1"].to_numpy()
    F2 = pca_factors["PC2"].to_numpy()

    # GSS rotation
    # Rotation is such that F2 does not affect current policy rate
    ois_1m = X_raw["OIS_1M"].to_numpy()
    X_1m = np.column_stack([np.ones(len(F1)),F1,F2])
    # OIS1M = alpha1*F1 + alpha2*F2 + error
    alpha = np.linalg.lstsq(X_1m, ois_1m, rcond=None)[0]
    alpha1 = alpha[1]
    alpha2 = alpha[2]
    if abs(alpha2) < 1e-12:
        raise ValueError("The loading of PC2 on the 1M OIS is too close to zero to construct the GSS rotation.")

    # Construction of the rotation matrix U
    # We search u1, u2 such that Target = u1F1 + u2F2 with u in the same direction as alpha but with norm 1
    # Hence we want ||u||= 1 (unit norm) and u1/u2 = alpha1/alpha2 (same direction)
    # so u1 = alpha1/alpha2 u2 
    # and ||u|| = sqrt(u1^2 + u2^2) = sqrt((alpha1/alpha2 * u2)^2 + u2^2) = sqrt((alpha1^2/alpha2^2 + 1) * u2^2) = sqrt(alpha1^2/alpha2^2 + 1) * u2 = 1
    # so u2 = 1 / sqrt(alpha1^2/alpha2^2 + 1)
    ratio = alpha1 / alpha2
    u2 = np.sqrt(1.0 / (ratio ** 2 + 1.0))
    u1 = ratio * u2

    if abs(u1) < 1e-12:
        raise ValueError("The Target rotation coefficient u1 is too close to zero.")

    # So now we want beta such that it has unit norm and is orthogonal to u
    # We thus need ||beta|| = 1 and u1beta1 + u2beta2 = 0 (orthogonal to u)
    # From the orthogonality condition we get beta1 = - (u2/u1) * beta2
    # From unit norm condition we get ||beta|| = sqrt(beta1^2 + beta2^2) = sqrt((-(u2/u1) * beta2)^2 + beta2^2) 
    # = sqrt((u2^2/u1^2 + 1) * beta2^2) = sqrt(u2^2/u1^2 + 1) * beta2 = 1
    # so beta2 = 1 / sqrt(u2^2/u1^2 + 1)
    beta2 = np.sqrt(1.0 / ((u2 / u1) ** 2 + 1.0))
    beta1 = -(u2 * beta2) / u1

    # Orthogonal rotation matrix U constructed from u and beta
    U = np.array([[u1, beta1],
                [u2, beta2]])
    # Rotated factors = F1 and F2 rotated such that F1 loads on the Target direction (1M) and F2 on the Path direction (orthogonal to Path)
    rotated = np.column_stack([F1,F2]) @ U

    target_temp = rotated[:, 0]
    path_temp = rotated[:, 1]

    # Scaling
    # GSS rescale the factors so that:
    #   - F1 moves OIS1M one for one -> for that we regress the 1M OIS on the rotated F1 and scale F1 by the first coef of that regression
    #       because if D_OIS1M = bF1 + error, then if Target = b * F1 then D_OIS1M = 1*Target + error
    #   - F1 and F2 have the same magnitude effect on the 1Y OIS -> for that we regress the 1Y OIS on the rotated F1 and F2 
    #       and scale F2 relative to F1 based on the regression coefficients
    #       because if D_OIS1Y = bTarget + cF2 + error, then if Path = (c/b) * F2, we have c*F2 = b*Path
    # and thus D_OIS1Y = b*Target + b*Path + error
    scale_target = np.linalg.lstsq(target_temp[:, None], ois_1m, rcond=None)[0][0]
    target = target_temp * scale_target
    ois_1y = X_raw["OIS_1Y"].to_numpy()
    X_1y = np.column_stack([np.ones(len(target)), target, path_temp]) 
    coefficients_1y = np.linalg.lstsq(X_1y, ois_1y, rcond=None)[0]
    beta_target_1y = coefficients_1y[1]
    beta_path_1y = coefficients_1y[2]

    if abs(beta_target_1y) < 1e-10:
        raise ValueError("Target has a near-zero loading on the 1Y OIS, so Path cannot be normalized relative to Target.")
    scale_path = beta_path_1y / beta_target_1y
    path = path_temp * scale_path

    # Final GSS factors
    factors = pd.DataFrame(
        {
            "Target": target,
            "Path": path
        },
        index=X_raw.index)
    
    # Diagnostic loadings of the final Target and Path factors
    loadings = compute_factor_loadings(X_raw, factors)
    # Aggregate MPS used in the reversal-shock framework
    factors["MPS"] = factors["Target"] + factors["Path"]

    # Scale each factor by its event-date standard deviation for the LP analysis.
    # A one-unit shock then corresponds to a one-standard-deviation surprise.
    for factor in ["Target", "Path", "MPS"]:
        factors[f"{factor}_std"] = factors[factor] / factors[factor].std(ddof=1)

    return MPSResult(
        loadings=loadings,
        factors=factors,
        explained_variance=explained_variance
    )


def export_mps(result: MPSResult, output: str):
    """
    Export the monetary policy surprise estimation results to an Excel file.

    Inputs:
    -----------
        :result: MPSResult object returned by compute_mps.
        :output: Path to the output Excel file.
    """

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    explained_variance = result.explained_variance.rename("explained_variance_ratio").to_frame()
    with pd.ExcelWriter(output) as writer:
        result.factors.to_excel(writer,sheet_name="factors",index=True,index_label="date")
        result.loadings.to_excel(writer,sheet_name="loadings",index=True,index_label="maturity")
        explained_variance.to_excel(writer,sheet_name="explained_variance",index=True,index_label="factor")