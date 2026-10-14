################################################
# 02_daily_filtering.py
# Daily filtering of the monthly-estimated ILS ATSM.
# The script loads adjusted daily ILS and decompose them based on 
# the standard and bias-adjusted parameters. 50/50 decomposition is also done.
# Importantly, we do not reestimate the ATSM at daily frequency, according to literature standards.
################################################

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from atsm.serialization import load_model_json
from atsm.apply_decomp import decompose_model, average_decomposition_frames
from utils.config import load_config
from utils.raw_ils import load_ils_panel
from utils.paths import project_path
from utils.helpers import save_csv

DEFAULT_CONFIG_PATH = "config/backasted.yaml"

# MAIN DAILY FILTERING PIPELINE
def main() -> None:
    """
    Apply the monthly-estimated ATSM to the adjusted daily ILS panel.
    """
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Apply monthly JSZ ATSMs to daily adjusted ILS data.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML configuration file.")
    args = parser.parse_args()

    # Configuration
    config = load_config(args.config)
    data_cfg = config["data"]
    sample_cfg = config["sample"]
    atsm_cfg = config["atsm"]
    output_cfg = config["output"]

    # Load daily adjusted ILS for filtering
    dates_d, yields_d, maturities, _ = load_ils_panel(path=data_cfg["daily_input"], date_column=data_cfg["date_column"],
                                                       maturity_columns=data_cfg["maturity_columns"], scale=data_cfg.get("scale", 1.0),
                                                       sheet_name=data_cfg.get("sheet_name"), end_date=sample_cfg["daily_end"])

    print("\n" + "=" * 70)
    print("DAILY ILS PANEL")
    print("=" * 70)
    print(f"Start: {dates_d.min().date()}")
    print(f"End:   {dates_d.max().date()}")
    print(f"N days: {len(dates_d)}")

    # Load ATSM model parameters
    model_standard = load_model_json(project_path(output_cfg["model_file"]))
    model_bias = load_model_json(project_path(output_cfg["model_file_bias"]))

    # Checks for consistency
    if not np.allclose(model_standard.W, model_bias.W):
        raise ValueError("Standard and bias-adjusted models do not use the same pricing-portfolio weights W.")
    if not np.array_equal(model_standard.maturities, model_bias.maturities):
        raise ValueError("Standard and bias-adjusted models use different maturity grids.")
    if not np.array_equal(maturities, model_standard.maturities):
        raise ValueError("Daily ILS maturities do not match the maturities used to estimate the ATSM.")

    # We compute daily pricing portfolios USING THE W ESTIMATED MONTHLY 
    P_daily = yields_d @ model_standard.W.T
    state_columns = [f"factor_{i + 1}" for i in range(P_daily.shape[1])]
    daily_states = pd.DataFrame(P_daily, columns=state_columns)
    daily_states.insert(0, "date", dates_d)

    # Daily decompositions
    # First standard, then bias adjusted, then 50/50
    dec_cfg = atsm_cfg.get("decomposition", {})
    daily_spot_standard, daily_forward_standard = decompose_model(model=model_standard,
                                                    yields=yields_d, dates=dates_d, 
                                                    maturities=maturities, dec_cfg=dec_cfg)
    
    daily_spot_bias, daily_forward_bias = decompose_model(model=model_bias,
                                            yields=yields_d, dates=dates_d,
                                            maturities=maturities, dec_cfg=dec_cfg)

    daily_spot_average = average_decomposition_frames(daily_spot_standard, daily_spot_bias, keys=["date", "maturity"])
    if daily_forward_standard is not None and daily_forward_bias is not None:
        daily_forward_average = average_decomposition_frames(daily_forward_standard, daily_forward_bias, keys=["date", "forward"])
    else:
        daily_forward_average = None

    # Save
    save_csv(daily_states, output_cfg["daily_states_file"])
    save_csv(daily_spot_standard, output_cfg["daily_spot_file"])
    save_csv(daily_forward_standard, output_cfg["daily_forward_file"])
    save_csv(daily_spot_bias, output_cfg["daily_spot_file_bias"])
    save_csv(daily_forward_bias, output_cfg["daily_forward_file_bias"])
    save_csv(daily_spot_average, output_cfg["daily_spot_file_average"])
    save_csv(daily_forward_average, output_cfg["daily_forward_file_average"])

    # Summary and logs
    print("\n" + "=" * 70)
    print("DAILY FILTERING COMPLETED")
    print("=" * 70)
    print(f"N daily observations: {len(dates_d)}")
    print(f"N pricing portfolios: {P_daily.shape[1]}")
    print("\nDaily states:")
    print(daily_states.head())
    print("\nAverage daily spot decomposition:")
    print(daily_spot_average.head())
    if daily_forward_average is not None:
        print("\nAverage daily forward decomposition:")
        print(daily_forward_average.head())

if __name__ == "__main__":
    main()