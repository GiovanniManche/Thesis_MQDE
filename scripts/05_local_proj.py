################################################
# 05_local_proj.py
# Main script for daily local projections and two-stage inference. It:
#   1. loads the monthly and daily ILS panels and baseline ATSM models;
#   2. reconstructs the daily spot and forward decomposition used in the LPs;
#   3. loads Altavilla monetary-policy shocks and narrative GPR shocks;
#   4. estimates baseline LPs with 5 lags of daily changes in fitted IC;
#   5. checks the IC = EC + IRP coefficient additivity identity;
#   6. runs an Eser-style residual bootstrap of the ATSM;
#   7. re-estimates all LPs in every ATSM bootstrap replication;
#   8. adds Loria-style Newey-West Gaussian draws at the LP stage;
#   9. exports point estimates, bootstrap confidence bands and diagnostics.
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
from lp.bootstrap import LPBootstrapSpec, combine_bootstrap_uncertainty, decompose_daily_bootstrap, run_atsm_lp_bootstrap
from lp.local_projection import estimate_lp_irfs, lp_results_to_long, max_additivity_error
from utils.config import load_config
from utils.paths import project_path
from utils.raw_ils import load_ils_panel

DEFAULT_CONFIG_PATH = "config/local_proj.yaml"

def load_mps_shocks(config: dict) -> pd.DataFrame:
    """
    Load standardized Altavilla factors used jointly in the monetary policy LPs.
    """
    path = project_path(config["input"])
    data = pd.read_excel(path, sheet_name=config.get("sheet", "factors"))
    columns = ["date"] + list(config["columns"])
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing MPS columns: {missing}")

    data = data[columns].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    return data.sort_values("date")

def load_gpr_shocks(config: dict) -> pd.DataFrame:
    """
    Load retained GPR events and split them into energy/non-energy shocks.
    """
    path = project_path(config["input"])
    # Load
    data = pd.read_excel(path, sheet_name=config.get("sheet", "shocks_narrative"))
    shock_column = config.get("shock_column", "gpr_shock_std")
    energy_column = config.get("energy_column", "is_energy")
    keep_column = config.get("keep_column", "keep")
    energy_name = config.get("energy_name", "GPR_energy_std")
    nonenergy_name = config.get("nonenergy_name", "GPR_nonenergy_std")
    required = ["date", shock_column, energy_column]
    # Clean
    if keep_column is not None:
        required.append(keep_column)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing GPR columns: {missing}")
    if keep_column is not None:
        data = data.loc[data[keep_column].fillna(False).astype(bool)].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data[shock_column] = pd.to_numeric(data[shock_column], errors="coerce")
    data = data.dropna(subset=[shock_column, energy_column])
    # We split into energy and non-energy shocks
    data[energy_name] = np.where(data[energy_column].astype(bool), data[shock_column], 0.0)
    data[nonenergy_name] = np.where(data[energy_column].astype(bool), 0.0, data[shock_column])
    return data[["date", energy_name, nonenergy_name]].sort_values("date")

def load_kerssenfischer_shocks(config: dict, ils_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load pure monetary policy and information shocks from Kerssenfischer.
    """

    data = pd.read_excel(project_path(config["input"]), sheet_name=config.get("sheet", "shocks"),
                         usecols=["Time", "purepolicy", "info"])

    data = data.rename(columns={"Time": "date", "purepolicy": "PurePolicy_std", "info": "Information_std"})
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    shock_columns = ["PurePolicy_std", "Information_std"]
    for col in shock_columns:
        data[col] = pd.to_numeric(data[col], errors="raise")

    if data["date"].isna().any() or data["date"].duplicated().any() or data[shock_columns].isna().any().any():
        raise ValueError("Invalid Kerssenfischer shocks.")

    # Standardize on ECB announcement dates overlapping the daily ILS sample,
    # consistently with the Altavilla monetary policy surprises.
    ils_dates = pd.DatetimeIndex(pd.to_datetime(ils_dates)).normalize()
    usable = data["date"].isin(ils_dates)

    for col in shock_columns:
        standard_deviation = data.loc[usable, col].std(ddof=1)
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError(f"Invalid standard deviation for {col}.")
        data[col] = data[col] / standard_deviation

    return data.sort_values("date").reset_index(drop=True)


def save_lp_results(results: dict[str, pd.DataFrame], path: Path) -> None:
    """
    Save one complete LP exercise in tidy long format
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = lp_results_to_long(results)
    frame.to_csv(path, index=False)

def build_lp_specs(mps: pd.DataFrame, gpr: pd.DataFrame, kers: pd.DataFrame, mps_columns: list[str], 
                   gpr_columns: list[str], kers_columns: list[str]) -> list[LPBootstrapSpec]:
    """
    We define here the LP exercises we want, so here MPS/GPR x spot/forward.
    """
    return [
        LPBootstrapSpec(name="spot_mps", curve="spot", shocks=mps, shock_cols=mps_columns, curve_col="maturity"),
        LPBootstrapSpec(name="forward_mps", curve="forward", shocks=mps, shock_cols=mps_columns, curve_col="forward"),
        LPBootstrapSpec(name="spot_gpr", curve="spot", shocks=gpr, shock_cols=gpr_columns, curve_col="maturity"),
        LPBootstrapSpec(name="forward_gpr", curve="forward", shocks=gpr, shock_cols=gpr_columns, curve_col="forward"),
        LPBootstrapSpec(name="spot_kerssenfischer", curve="spot", shocks=kers, shock_cols=kers_columns, curve_col="maturity"),
        LPBootstrapSpec(name="forward_kerssenfischer", curve="forward", shocks=kers, shock_cols=kers_columns, curve_col="forward")
    ]


def main() -> None:
    """
    Estimate baseline LPs and, unless disabled, two-stage bootstrap confidence bands.
    """
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Estimate ILS decomposition local projections and bootstrap inference.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to local-projection YAML configuration file.")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Estimate point LPs only.")
    parser.add_argument("--outer", type=int, default=None, help="Override the number of outer ATSM bootstrap replications.")
    parser.add_argument("--inner", type=int, default=None, help="Override the number of inner LP Gaussian draws.")
    parser.add_argument("--quiet", action="store_true", help="Reduce bootstrap progress output.")
    args = parser.parse_args()

    # Configuration
    config = load_config(args.config)
    atsm_config = load_config(config.get("atsm_config", "config/backasted.yaml"))
    lp_cfg = config["lp"]
    bootstrap_cfg = config["bootstrap"]
    shocks_cfg = config["shocks"]
    output_dir = project_path(config["output"]["directory"])
    data_cfg = atsm_config["data"]
    sample_cfg = atsm_config["sample"]
    model_cfg = atsm_config["atsm"]
    model_output_cfg = atsm_config["output"]

    # We load monthly and daily ILS data
    monthly_dates, monthly_yields, maturities, _ = load_ils_panel(path=data_cfg["monthly_input"],
                                                    date_column=data_cfg["date_column"], maturity_columns=data_cfg["maturity_columns"],
                                                    scale=data_cfg.get("scale", 1.0), sheet_name=data_cfg.get("sheet_name"),
                                                    end_date=sample_cfg["monthly_end"])
    daily_dates, daily_yields, daily_maturities, _ = load_ils_panel(path=data_cfg["daily_input"], date_column=data_cfg["date_column"],
                                                    maturity_columns=data_cfg["maturity_columns"], scale=data_cfg.get("scale", 1.0),
                                                    sheet_name=data_cfg.get("sheet_name"), end_date=sample_cfg["daily_end"])
    if not np.array_equal(maturities, daily_maturities):
        raise ValueError("Monthly and daily ILS panels use different maturity grids.")

    # We get the ORIGINAL estimated ATSMs
    model_standard = load_model_json(project_path(model_output_cfg["model_file"]))
    model_bias = load_model_json(project_path(model_output_cfg["model_file_bias"]))
    if not np.allclose(model_standard.W, model_bias.W):
        raise ValueError("Standard and bias-adjusted ATSMs use different pricing-portfolio weights W.")
    decomposition_model = lp_cfg.get("decomposition_model", "average")
    dec_cfg = model_cfg.get("decomposition", {})

    # Point estimate = the LP with baseline decomposition (50/50)
    # Side note: we could very well import the ones we exported in Excel in script 2, but for coherence we recompute them here
    spot_decomposition, forward_decomposition = decompose_daily_bootstrap(model_standard=model_standard,
                                                model_bias=model_bias, daily_yields=daily_yields,
                                                daily_dates=daily_dates, maturities=maturities,
                                                decomposition_config=dec_cfg, decomposition_model=decomposition_model)
    if forward_decomposition is None:
        raise ValueError("Forward decomposition is required by the baseline LP specification.")

    # Shocks
    mps = load_mps_shocks(shocks_cfg["mps"])
    gpr = load_gpr_shocks(shocks_cfg["gpr"])
    kers = load_kerssenfischer_shocks(shocks_cfg["kerssenfischer"], daily_dates)
    kers_columns = ["PurePolicy_std", "Information_std"]
    mps_columns = list(shocks_cfg["mps"]["columns"])
    gpr_columns = [shocks_cfg["gpr"].get("energy_name", "GPR_energy_std"),
                   shocks_cfg["gpr"].get("nonenergy_name", "GPR_nonenergy_std")]
    
    # As of now Kerssenfischer shocks ends in July 2026
    last_kers_date = kers["date"].max()
    missing_dates = daily_dates[daily_dates > last_kers_date]
    if len(missing_dates):
        missing_tail = pd.DataFrame({"date": missing_dates,
                                    "PurePolicy_std": np.nan, "Information_std": np.nan})
        kers = pd.concat([kers, missing_tail], ignore_index=True)
    
    lp_specs = build_lp_specs(mps=mps, gpr=gpr, kers=kers, mps_columns=mps_columns, gpr_columns=gpr_columns, 
                              kers_columns=kers_columns)
    # LP parameters
    max_horizon = int(lp_cfg.get("max_horizon", 25))
    ic_lags = int(lp_cfg.get("ic_lags", 5))
    shock_lags = int(lp_cfg.get("shock_lags", 0))
    confidence_level = float(lp_cfg.get("confidence_level", 0.90))
    nw_lags = lp_cfg.get("nw_lags", "horizon")

    # Point LPs
    print("\n" + "=" * 70)
    print("BASELINE LOCAL PROJECTIONS")
    print("=" * 70)
    point_results = {}
    sanity_rows = []

    for spec in lp_specs:
        decomposition = spot_decomposition if spec.curve == "spot" else forward_decomposition
        results = estimate_lp_irfs(decomposition=decomposition, shocks=spec.shocks,
                                   max_horizon=max_horizon, ic_lags=ic_lags, 
                                   shock_lags=shock_lags, confidence_level=confidence_level,
                                   curve_col=spec.curve_col, shock_cols=spec.shock_cols, nw_lags=nw_lags)
        point_results[spec.name] = results
        additivity_error = max_additivity_error(results)
        sanity_rows.append({"lp_specification": spec.name, "max_abs_beta_additivity_error": additivity_error})
        save_lp_results(results=results, path=output_dir / "point" / f"{spec.name}.csv")
        print(f"{spec.name}: max |beta_IC - beta_EC - beta_IRP| = {additivity_error:.3e}")

    # Report sanity checks: IC LPs coefs should = EC LPs coefs + IRP LPs coefs
    sanity = pd.DataFrame(sanity_rows)
    sanity_path = output_dir / "point" / "sanity_checks.csv"
    sanity_path.parent.mkdir(parents=True, exist_ok=True)
    sanity.to_csv(sanity_path, index=False)

    bootstrap_enabled = bool(bootstrap_cfg.get("enabled", True)) and not args.skip_bootstrap
    if not bootstrap_enabled:
        print("\nBootstrap skipped. Point LP estimates exported.")
        return

    # Bootstrap settings
    n_bootstrap = int(args.outer if args.outer is not None else bootstrap_cfg.get("outer_replications", 100))
    n_inner_draws = int(args.inner if args.inner is not None else bootstrap_cfg.get("inner_draws", 100))
    random_seed = int(bootstrap_cfg.get("random_seed", atsm_config["project"].get("random_seed", 12345)))
    generator_name = bootstrap_cfg.get("generator_model", "standard")
    if generator_name == "standard":
        generator_model = model_standard
    elif generator_name == "bias_adjusted":
        generator_model = model_bias
    else:
        raise ValueError("bootstrap.generator_model must be standard or bias_adjusted.")

    # Bootstrap for the standard errors
    print("\n" + "=" * 70)
    print("TWO-STAGE BOOTSTRAP")
    print("=" * 70)
    print(f"Outer ATSM replications: {n_bootstrap}")
    print(f"Inner LP draws:          {n_inner_draws}")
    print(f"Decomposition:           {decomposition_model}")
    print(f"ATSM bootstrap DGP:      {generator_name}")

    outer_storage, bootstrap_diagnostics = run_atsm_lp_bootstrap(monthly_yields=monthly_yields, daily_yields=daily_yields,
                                                                daily_dates=daily_dates, maturities=maturities, generator_model=generator_model,
                                                                atsm_config=model_cfg, decomposition_config=dec_cfg, point_results=point_results,
                                                                lp_specs=lp_specs, n_bootstrap=n_bootstrap, max_horizon=max_horizon, ic_lags=ic_lags,
                                                                shock_lags=shock_lags, confidence_level=confidence_level, nw_lags=nw_lags,
                                                                decomposition_model=decomposition_model, reestimate_pca=bool(bootstrap_cfg.get("reestimate_pca", False)),
                                                                require_optimizer_success=bool(bootstrap_cfg.get("require_optimizer_success", True)),
                                                                max_attempts=bootstrap_cfg.get("max_attempts"), random_seed=random_seed,
                                                                atsm_n_random_seeds=bootstrap_cfg.get("atsm_n_random_seeds"),
                                                                atsm_n_repeats=bootstrap_cfg.get("atsm_n_repeats"), 
                                                                atsm_maxiter=bootstrap_cfg.get("atsm_maxiter"), verbose=not args.quiet)

    diagnostics_path = output_dir / "bootstrap" / "atsm_bootstrap_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_diagnostics.to_csv(diagnostics_path, index=False)

    # Loria-style inner draws and final confidence bands
    for job_index, spec in enumerate(lp_specs):
        final_results = combine_bootstrap_uncertainty(point_results=point_results[spec.name], storage=outer_storage[spec.name],
                                                      n_inner_draws=n_inner_draws, confidence_level=confidence_level,
                                                      random_seed=random_seed + 10000 + job_index,
                                                      recenter=bool(bootstrap_cfg.get("recenter_intervals", True)))
        save_lp_results(results=final_results, path=output_dir / "bootstrap" / f"{spec.name}.csv")
        
    print("\n" + "=" * 70)
    print("LOCAL PROJECTIONS AND BOOTSTRAP COMPLETED")
    print("=" * 70)
    print(f"Outputs: {output_dir}")





if __name__ == "__main__":
    main()
