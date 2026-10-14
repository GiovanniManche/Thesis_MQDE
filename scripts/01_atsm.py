################################################
# 01_atsm.py
# Main script for monthly estimation of the JSZ inflation ATSM. It:
#   1. loads the monthly ILS panel;
#   2. constructs observable PCA pricing portfolios;
#   3. estimates the standard ATSM;
#   4. estimates the persistence-bias-adjusted ATSM;
#   5. decomposes observed ILS rates into expected inflation and
#      inflation risk premia for both specifications;
#   6. averages the two decompositions 50/50;
#   7. computes and saves model diagnostics;
#   8. serializes the complete ATSMModel objects to JSON.
################################################
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from atsm.decomposition import decompose_ils, forward_decomposition_frame, spot_decomposition_frame
from atsm.diagnostics import model_diagnostics
from atsm.estimation import estimate_atsm
from atsm.pricing_portfolios import compute_pricing_portfolios
from atsm.serialization import save_model_json
from atsm.apply_decomp import decompose_model, average_decomposition_frames
from utils.config import load_config
from utils.raw_ils import load_ils_panel
from utils.paths import project_path
from utils.helpers import save_csv


DEFAULT_CONFIG_PATH = "config/backasted.yaml"

# MAIN ESTIMATION PIPELINE
def main() -> None:
    """
    Run the complete monthly ATSM estimation and decomposition pipeline.
    """

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Estimate the standard and bias-adjusted JSZ ILS ATSMs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML configuration file.")
    parser.add_argument("--quiet", action="store_true", help="Reduce optimizer output.")
    args = parser.parse_args()

    # Configuration
    config = load_config(args.config)
    data_cfg = config["data"]
    sample_cfg = config["sample"]
    atsm_cfg = config["atsm"]
    output_cfg = config["output"]

    # Load monthly ILS for estimation
    dates_m, yields_m, maturities, _ = load_ils_panel(path=data_cfg["monthly_input"], date_column=data_cfg["date_column"],
                                                      maturity_columns=data_cfg["maturity_columns"], scale=data_cfg.get("scale", 1.0),
                                                      sheet_name=data_cfg.get("sheet_name"), end_date=sample_cfg["monthly_end"])

    # Check: if maturities in the ILS file do not correspond to the ones in the configuration yaml
    configured_maturities = np.array(sorted(int(m) for m in data_cfg["maturity_columns"].keys()))
    if not np.array_equal(maturities, configured_maturities):
        raise ValueError("Unexpected maturity ordering in the monthly input.")

    # Observable pricing portfolios - PCA
    pca_cfg = atsm_cfg.get("pca", {})
    factors = compute_pricing_portfolios(yields=yields_m, n_factors=int(atsm_cfg["n_factors"]), factor_signs=pca_cfg.get("factor_signs"))
    print("Explained variance ratios:", factors.explained_variance_ratio)

    # Maximum likelihood estimation
    # First standard ATSM, then bias-corrected
    random_seed = int(config["project"].get("random_seed", 12345))
    print("\n" + "=" * 70)
    print("ESTIMATING STANDARD ATSM")
    print("=" * 70)

    model_standard = estimate_atsm(yields=yields_m, W=factors.W, maturities=maturities,
                                   dt=float(atsm_cfg["dt"]), config=atsm_cfg, random_seed=random_seed,
                                   verbose=not args.quiet, bias_adjusted=False)

    print("\n" + "=" * 70)
    print("ESTIMATING BIAS-ADJUSTED ATSM")
    print("=" * 70)

    model_bias = estimate_atsm(yields=yields_m, W=factors.W, maturities=maturities,
                               dt=float(atsm_cfg["dt"]), config=atsm_cfg, random_seed=random_seed,
                               verbose=not args.quiet, bias_adjusted=True)

    model_standard.metadata["explained_variance_ratio"] = factors.explained_variance_ratio.copy()
    model_bias.metadata["explained_variance_ratio"] = factors.explained_variance_ratio.copy()

    # Monthly decomposition
    dec_cfg = atsm_cfg.get("decomposition", {})
    monthly_spot_standard, monthly_forward_standard = decompose_model(model=model_standard, yields=yields_m, dates=dates_m, 
                                                                       maturities=maturities, dec_cfg=dec_cfg)
    monthly_spot_bias, monthly_forward_bias = decompose_model(model=model_bias, yields=yields_m, dates=dates_m,
                                                               maturities=maturities, dec_cfg=dec_cfg)
    monthly_spot_average = average_decomposition_frames(monthly_spot_standard, monthly_spot_bias, keys=["date", "maturity"])
    if (monthly_forward_standard is not None and monthly_forward_bias is not None):
        monthly_forward_average = average_decomposition_frames(monthly_forward_standard, monthly_forward_bias, keys=["date", "forward"])
    else:
        monthly_forward_average = None

    # Diagnostics
    diagnostics_standard = model_diagnostics(model_standard)
    diagnostics_standard["explained_variance_ratio"] = factors.explained_variance_ratio.copy()
    diagnostics_standard["model_specification"] = "standard"
    diagnostics_bias = model_diagnostics(model_bias)
    diagnostics_bias["explained_variance_ratio"] = factors.explained_variance_ratio.copy()
    diagnostics_bias["model_specification"] = "bias_adjusted"
    model_standard.metadata.update(diagnostics_standard)
    model_bias.metadata.update(diagnostics_bias)

    # We save complete model parameters and diagnostics in JSON files
    save_model_json(model_standard, project_path(output_cfg["model_file"]))
    save_model_json(model_bias, project_path(output_cfg["model_file_bias"]))

    # We save model decompositions in CSV files
    save_csv(monthly_spot_standard, output_cfg["monthly_spot_file"])
    save_csv(monthly_forward_standard, output_cfg["monthly_forward_file"])
    save_csv(monthly_spot_bias, output_cfg["monthly_spot_file_bias"])
    save_csv(monthly_forward_bias, output_cfg["monthly_forward_file_bias"])
    save_csv(monthly_spot_average, output_cfg["monthly_spot_file_average"])
    save_csv(monthly_forward_average, output_cfg["monthly_forward_file_average"])

    # Final summary printed in the console
    print("\nATSM estimations complete.")

    print("\n" + "=" * 70)
    print("\nSTANDARD MODEL")
    print(f"Optimizer success: {model_standard.optimizer_success}")
    print(f"Objective: {model_standard.objective_value:.10g}")
    print(f"Max |eig(P)|: {diagnostics_standard['max_abs_eigenvalue_P']:.6f}")
    print(f"Long-run inflation: {diagnostics_standard['long_run_inflation_percent']:.4f}%")
    print("=" * 70)

    print("\nBIAS-ADJUSTED MODEL")
    print(f"Optimizer success: {model_bias.optimizer_success}")
    print(f"Objective: {model_bias.objective_value:.10g}")
    print(f"Max |eig(P)|: {diagnostics_bias['max_abs_eigenvalue_P']:.6f}")
    print(f"Long-run inflation: {diagnostics_bias['long_run_inflation_percent']:.4f}%")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()