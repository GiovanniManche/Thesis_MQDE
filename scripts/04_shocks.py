################################################
# 04_shocks.py
# Script for computing and exporting monetary policy and geopolitical risk shocks
################################################

from datetime import timedelta
from pathlib import Path
import sys
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.raw_ils import load_ils_trading_dates
from shocks.mps import compute_mps_altavilla_orchestrator, compute_mps_gss_orchestrator, export_mps
from shocks.gpr_shocks import compute_gpr_shocks_orchestrator, export_gpr_shocks

DEFAULT_CONFIG_PATH = "config/shocks.yaml"
ILS_DAILY_FILE = PROJECT_ROOT / "output/atsm/decomposition/daily/backasted/daily_spot_standard.csv"

def standardize_mps_on_ils_sample(result, ils_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Function to standardize MPS factors based on the usable ILS sample (so that they have unit standard deviation over the relevant sample)
    """
    # First extract MPS values which are within the ILS sample dates
    factors = result.factors.copy()
    factors.index = pd.to_datetime(factors.index)
    ils_dates = pd.DatetimeIndex(ils_dates).normalize()
    usable = factors.index.normalize().isin(ils_dates)
    raw_columns = [column for column in factors.select_dtypes(include="number").columns if not column.endswith("_std")]
    # Then we standardize each factor based on the usable ILS sample
    for column in raw_columns:
        standardization_sample = factors.loc[usable, column].dropna()
        if standardization_sample.empty:
            raise ValueError(f"No usable ILS-sample observations are available to standardize {column}.")
        standard_deviation = standardization_sample.std(ddof=1)
        if not pd.notna(standard_deviation) or standard_deviation <= 0:
            raise ValueError(f"Invalid standard deviation for MPS factor {column}.")
        factors[f"{column}_std"] = factors[column] / standard_deviation
    result.factors = factors
    return result

def standardize_gpr_innovations(result: pd.Series, ils_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Standardize GPR ARMA innovations using their standard deviation over the ILS-compatible estimation sample.
    NOTE: while we use 2002-2005 innovations for the rolling volatility, we do not use them for shocks identification
    and futhermore do not use them to standardise our final shocks.    
    """
    innovations = result.innovations.copy()
    shocks = result.shocks.copy()
    innovations.index = pd.to_datetime(innovations.index)
    ils_dates = pd.DatetimeIndex(pd.to_datetime(ils_dates)).normalize()
    usable = innovations.index.normalize().isin(ils_dates)
    standardization_sample = innovations.loc[usable, "innovation"].dropna()
    if standardization_sample.empty:
        raise ValueError("No usable ILS-sample observations are available to standardize GPR innovations.")
    standard_deviation = standardization_sample.std(ddof=1)
    if not pd.notna(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Invalid standard deviation for GPR innovations.")
    # Standardized innovation on all dates
    innovations["innovation_std"] = innovations["innovation"] / standard_deviation
    # Zero on non-shock dates
    innovations["gpr_shock_std"] = innovations["gpr_shock"] / standard_deviation
    # Same variables in the event-only table
    shocks["innovation_std"] = shocks["innovation"] / standard_deviation
    shocks["gpr_shock_std"] = shocks["gpr_shock"] / standard_deviation
    result.innovations = innovations
    result.shocks = shocks
    return result

def run_mps_shocks(mps_config: dict, ils_dates: pd.DatetimeIndex):
    """
    Estimate daily MPS innovations and export large positive surprises.
    Standardize wrt the ILS-compatible sample
    Also shows some user logs
    """
    input_file = Path(mps_config["input"])
    result_altavilla = compute_mps_altavilla_orchestrator(file=input_file, release=mps_config["release_sheet"], conference=mps_config["conference_sheet"])
    result_altavilla = standardize_mps_on_ils_sample(result_altavilla, ils_dates)
    export_mps(result=result_altavilla, output=mps_config["output_altavilla"])
    print(f"Altavilla shocks estimation completed: {mps_config['output_altavilla']}")

    result_gss = compute_mps_gss_orchestrator(file=input_file, release=mps_config["release_sheet"])
    result_gss = standardize_mps_on_ils_sample(result_gss, ils_dates)
    export_mps(result=result_gss, output=mps_config["output_gss"]) 
    print(f"GSS MPS estimation completed: {mps_config['output_gss']}")

    return result_altavilla, result_gss

def run_gpr_shocks(gpr_config: dict, ils_dates: pd.DatetimeIndex):
    """
    Estimate daily GPR innovations and export large positive surprises.
    Also shows some user logs
    """
    result = compute_gpr_shocks_orchestrator(gpr_config=gpr_config, trading_dates=ils_dates)
    result = standardize_gpr_innovations(result, ils_dates)
    export_gpr_shocks(result=result, output=gpr_config["output"])

    difference_order = int(result.summary.loc["difference_order_d", "value"])
    transformation = "GPR levels" if difference_order == 0 else "first-differenced GPR"
    # User logs
    print(f"GPR shocks exported to {gpr_config['output']}")
    print(f"Selected ARMA({int(result.summary.loc['ar_order_p', 'value'])}, "
          f"{int(result.summary.loc['ma_order_q', 'value'])}) on {transformation}")
    print(f"ADF test on model GPR: stat={float(result.summary.loc['adf_statistic', 'value']):.3f}, "
          f"p={float(result.summary.loc['adf_p_value', 'value']):.4g}")
    print(f"Brent classification: {int(result.summary.loc['n_energy_shocks', 'value'])} energy / "
          f"{int(result.summary.loc['n_non_energy_shocks', 'value'])} non-energy")
    print(f"Selected shocks: {len(result.shocks)}")
    print(f"Rolling-vol-adjusted innovation threshold: {float(result.summary.loc['threshold_value', 'value']):.3f} SD")
    return result

def main():
    """Compute and export monetary policy and geopolitical risk shocks."""
    with open(DEFAULT_CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    ils_dates = load_ils_trading_dates(ILS_DAILY_FILE)
    print("\n" + "=" * 70)
    print("Running MPS shocks estimation")
    run_mps_shocks(config["mps"], ils_dates)
    print("=" * 70)
    print("Running GPR shocks estimation")
    run_gpr_shocks(config["gpr"], ils_dates)
    print("=" * 70)
    print("DONE")

if __name__ == "__main__":
    main()
