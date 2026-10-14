################################################
# 03_expanding_estimation.py
#
# Expanding-window estimation of the JSZ inflation ATSM.
#
# Starting from a specified vintage, the model is re-estimated
# every month using all observations available up to that date.
#
# The PCA pricing-portfolio weights W are held fixed across
# vintages so that parameter estimates remain expressed in the
# same observable factor basis.
#
# The script stores only parameter-stability diagnostics:
#   - P and Q dynamics;
#   - short-rate loadings;
#   - covariance parameters;
#   - transition-matrix eigenvalues;
#   - spectral radii;
#   - optimizer diagnostics.
#
# No fitted or decomposed inflation series are stored.
################################################

import argparse
from itertools import permutations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from atsm.estimation import estimate_atsm
from atsm.serialization import load_model_json
from atsm.pricing_portfolios import compute_pricing_portfolios
from utils.config import load_config
from utils.raw_ils import load_ils_panel
from utils.paths import project_path
from utils.helpers import save_csv, flatten_matrix, flatten_vector

DEFAULT_CONFIG_PATH = "config/backasted.yaml"

def expanding_estimation(yields, dates, maturities, atsm_cfg, first_vintage, random_seed, 
                         bias_adjusted=False, apply_killian: bool = True):
    """
    Function to estimate the ATSM recursively on an expanding sample.
    Each time a month is added, the pricing portfolios are reestimated, then the whole ATSM

    Inputs:
    -----------
        :yields: Monthly ILS rates used for the expanding estimation.
        :dates: Monthly observation dates 
        :maturities: ILS maturities in years.
        :atsm_cfg: ATSM estimation parameters from the configuration yaml.
        :first_vintage: First month for which an expanding-window ATSM is estimated.
        :random_seed: Random seed used in the ATSM optimization.
        :bias_adjusted: If true, estimate the peristence bias adjusted version
        :apply_killian: if False, doesn't apply the Killian shrinkage. Useful in expanding setting because for some dates (2022)
                                it is actually impossible to get eigenvalues <1.
    Output:
    ----------
        Dataframe containing the estimated model coefficients and persistence diagnostics for each expanding-window vintage.
    """
    first_period = pd.Period(first_vintage, freq="M")
    periods = dates.to_period("M")
    matching_indices = np.where(periods == first_period)[0]
    if len(matching_indices) == 0:
        raise ValueError(f"No monthly observation found for first expanding vintage {first_period}.")
    first_end_idx = matching_indices[-1]
    rows = []
    previous_eig_P = None
    previous_eig_Q = None
    n_estimations = len(dates) - first_end_idx
    previous_W = None

    # Expanding window loop
    for count, end_idx in enumerate(range(first_end_idx, len(dates)), start=1):
        yields_vintage = yields[:end_idx + 1]
        estimation_date = dates[end_idx]
        # User log
        print("\n" + "=" * 70)
        print(f"EXPANDING ESTIMATION {count} / {n_estimations} - {estimation_date.date()}")
        print("=" * 70)
        print(f"Sample: {dates[0].date()} -> {estimation_date.date()} ({len(yields_vintage)} observations)")

        # We compute W each time. However to ensure we don't have significantly different values because of sign changes,
        # we ensure a stable sign convention
        factors = compute_pricing_portfolios(yields=yields_vintage, n_factors=int(atsm_cfg["n_factors"]), factor_signs=None)
        W_vintage = factors.W
        
        if previous_W is not None:
            for i in range(W_vintage.shape[0]):
                if np.dot(W_vintage[i], previous_W[i]) < 0:
                    W_vintage[i] *= -1
        previous_W = W_vintage.copy()

        print("W rank:", np.linalg.matrix_rank(W_vintage))
        print("W shape:", W_vintage.shape)
        print("PCA EVR:", factors.explained_variance_ratio)

        model = estimate_atsm(yields=yields_vintage, W=W_vintage, maturities=maturities,
                              dt=float(atsm_cfg["dt"]), config=atsm_cfg, random_seed=random_seed,
                              verbose=False, bias_adjusted=bias_adjusted, apply_killian=apply_killian)

        N = model.K1P_P.shape[0]
        # We save the max eigenvalues
        eig_P = np.linalg.eigvals(np.eye(N) + model.K1P_P)
        eig_Q = np.linalg.eigvals(np.eye(N) + model.K1Q_P)
        idx_P = np.argmax(np.abs(eig_P))
        idx_Q = np.argmax(np.abs(eig_Q))
        maxeig_P = float(np.abs(eig_P[idx_P]))
        maxeig_Q = float(np.abs(eig_Q[idx_Q]))
        dominant_eig_P = eig_P[idx_P]
        dominant_eig_Q = eig_Q[idx_Q]


        row = {"date": estimation_date,
                "n_observations": len(yields_vintage),
                "spectral_radius_P": maxeig_P,
                "spectral_radius_Q": maxeig_Q,
                "dominant_eigenvalue_P_real": float(dominant_eig_P.real),
                "dominant_eigenvalue_P_imag": float(dominant_eig_P.imag),
                "dominant_eigenvalue_Q_real": float(dominant_eig_Q.real),
                "dominant_eigenvalue_Q_imag": float(dominant_eig_Q.imag),
                "long_run_inflation": model.metadata["long_run_inflation"],
            }

        row.update(flatten_vector("K0P", model.K0P_P))
        row.update(flatten_matrix("K1P",model.K1P_P))
        row.update(flatten_vector("K0Q",model.K0Q_P))
        row.update(flatten_matrix("K1Q",model.K1Q_P))
        row.update(flatten_vector("rho1",model.rho1_P))
        row.update(flatten_matrix("Sigma",model.Sigma_P))

        for i in range(N):
            row[f"eig_P_{i + 1}_real"] = float(eig_P[i].real)
            row[f"eig_P_{i + 1}_imag"] = float(eig_P[i].imag)
            row[f"eig_P_{i + 1}_abs"] = float(abs(eig_P[i]))
            row[f"eig_Q_{i + 1}_real"] = float(eig_Q[i].real)
            row[f"eig_Q_{i + 1}_imag"] = float(eig_Q[i].imag)
            row[f"eig_Q_{i + 1}_abs"] = float(abs(eig_Q[i]))

        rows.append(row)

    return pd.DataFrame(rows)

# MAIN EXPANDING SETTING PIPELINE
def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanding-window JSZ ATSM estimations.")
    # Configuration
    parser.add_argument("--config",default=DEFAULT_CONFIG_PATH,help="Path to YAML configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    sample_cfg = config["sample"]
    atsm_cfg = config["atsm"]
    output_cfg = config["output"]
    expanding_cfg = atsm_cfg["expanding"]

    dates, yields, maturities, _ = load_ils_panel(path=data_cfg["monthly_input"],date_column=data_cfg["date_column"],
                                                  maturity_columns=data_cfg["maturity_columns"], scale=data_cfg.get("scale", 1.0),
                                                  sheet_name=data_cfg.get("sheet_name"), end_date=sample_cfg["monthly_end"])

    baseline_model = load_model_json(project_path(output_cfg["model_file"]))
    W_fixed = baseline_model.W.copy()

    if not np.array_equal(maturities, baseline_model.maturities):
        raise ValueError("Monthly input maturities do not match the baseline ATSM.")

    random_seed = int(config["project"].get("random_seed",12345))
    first_vintage = expanding_cfg.get("first_vintage","2018-01-31")

    # Standard ATSM
    print("\n" + "%" * 70)
    print("STANDARD ATSM - EXPANDING")
    print("%" * 70)
    expanding_standard = expanding_estimation(yields=yields, dates=dates, maturities=maturities,
                                              atsm_cfg=atsm_cfg, first_vintage=first_vintage,
                                              random_seed=random_seed, bias_adjusted=False, apply_killian=expanding_cfg["apply_killian"])

    save_csv(expanding_standard, output_cfg["expanding_standard_file"])

    # Bias-adjusted ATSM
    if expanding_cfg.get("estimate_bias_adjusted", True):
        print("\n" + "%" * 70)
        print("PERSISTENCE-BIAS-CORRECTION ATSM - EXPANDING")
        print("%" * 70)
        expanding_bias = expanding_estimation(yields=yields, dates=dates, maturities=maturities,
                                              atsm_cfg=atsm_cfg, first_vintage=first_vintage,
                                              random_seed=random_seed, bias_adjusted=True, apply_killian=expanding_cfg["apply_killian"])
        save_csv(expanding_bias, output_cfg["expanding_bias_file"])

if __name__ == "__main__":
    main()