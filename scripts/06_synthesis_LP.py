################################################
# 06_synthesis_LP.py
# Post-estimation synthesis of the bootstrap local projections. It:
#   1. loads the final LP outputs produced by 05_local_proj.py;
#   2. exports bootstrap IRF grids for every shock;
#   3. optionally exports selected individual bootstrap IRFs;
#   4. computes cross-shock synthetic statistics using forward-rate IRFs only;
#   5. exports the synthetic measures to a single Excel workbook;
#   6. produces the propagation, persistence, contribution, maturity-location
#      and EC-vs-IRP synthesis figures.
################################################

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lp.measures_and_charts import (
    DEFAULT_FORWARD_ORDER,
    compute_synthetic_measures,
    export_synthetic_measures_excel,
    load_bootstrap_lp_results,
    plot_bootstrap_irf,
    plot_bootstrap_irf_grid,
    plot_ec_irp_scatter,
    plot_irp_contribution_shares,
    plot_maturity_centres,
    plot_propagation_persistence,
)

from utils.config import load_config
from utils.paths import project_path


DEFAULT_CONFIG_PATH = "config/local_proj.yaml"
DEFAULT_LP_FILES = ["spot_mps", "forward_mps", "spot_gpr", "forward_gpr"]
DEFAULT_FORWARD_FILES = ["forward_mps", "forward_gpr"]


def _save_figure(fig, path: Path, dpi: int = 300) -> None:
    """
    Save and close a matplotlib figure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _load_lp_outputs(lp_output_dir: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load named bootstrap LP outputs from the standard output directory.
    """
    outputs = {}
    for name in names:
        path = lp_output_dir / "bootstrap" / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing LP bootstrap output: {path}")
        outputs[name] = load_bootstrap_lp_results(path)
    return outputs


def main() -> None:
    """
    Produce bootstrap IRF figures and cross-shock synthesis outputs.
    """
    # Parser
    parser = argparse.ArgumentParser(
        description="Plot bootstrap ILS local projections and compute cross-shock synthesis measures."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to local-projection YAML configuration file.",
    )
    parser.add_argument(
        "--skip-irf-grids",
        action="store_true",
        help="Skip the full curve-point x component IRF grids.",
    )
    parser.add_argument(
        "--skip-synthesis-figures",
        action="store_true",
        help="Compute/export measures but skip the cross-shock synthesis figures.",
    )
    args = parser.parse_args()
    # Imports
    config = load_config(args.config)
    lp_output_dir = project_path(config["output"]["directory"])
    synthesis_cfg = config.get("synthesis", {})


    output_dir = project_path(synthesis_cfg.get("output_directory", str(lp_output_dir / "synthesis")))
    figure_dir = output_dir / "figures"
    rate_scale = float(synthesis_cfg.get("rate_scale", 10_000.0))
    confidence_level = float(synthesis_cfg.get("confidence_level", config["lp"].get("confidence_level", 0.90)))
    figure_format = str(synthesis_cfg.get("figure_format", "png")).lstrip(".")
    figure_dpi = int(synthesis_cfg.get("figure_dpi", 300))
    shock_labels = synthesis_cfg.get("shock_labels", {})
    lp_file_names = list(synthesis_cfg.get("irf_inputs", DEFAULT_LP_FILES))
    lp_outputs = _load_lp_outputs(lp_output_dir=lp_output_dir, names=lp_file_names)

    
    # Selected monetary-policy IRFs: shocks in rows, components in columns.
    selected_rows = [
        ("Not energy-related", "GPR_nonenergy_std", "1y1y"),
        ("Not energy-related", "GPR_nonenergy_std", "1y4y"),
        ("Not energy-related", "GPR_nonenergy_std", "5y5y"),
    ]

    components = ["IC", "EC", "IRP"]
    results = lp_outputs["forward_gpr"]

    fig, axes = plt.subplots(len(selected_rows), 3, figsize=(10, 2.5 * len(selected_rows)), sharex=True, squeeze=False)

    for i, (label, shock, curve_point) in enumerate(selected_rows):
        for j, component in enumerate(components):
            ax = axes[i, j]

            plot_bootstrap_irf(
                results=results,
                shock=shock,
                curve_point=curve_point,
                component=component,
                rate_scale=rate_scale,
                confidence_level=confidence_level,
                ax=ax,
                title=""
            )

            ax.get_legend().remove()
            ax.set_title(component if i == 0 else "")
            ax.set_ylabel(f"{label} ({curve_point})\nBasis points" if j == 0 else "")
            ax.set_xlabel("Horizon (business days)" if i == len(selected_rows) - 1 else "")

        # Common y-axis scale across IC, EC and IRP within each shock.
        lower = min(ax.get_ylim()[0] for ax in axes[i])
        upper = max(ax.get_ylim()[1] for ax in axes[i])
        for ax in axes[i]:
            ax.set_ylim(lower, upper)

    fig.suptitle("Non-energy-based geopolitical surprises: decomposition of inflation compensation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    _save_figure(fig, figure_dir / "selected_irfs" / "not_energy_gpr_IRFs.png", dpi=300)

    
    # Kerssenfischer IRFs: shocks in rows, components in columns.
    selected_rows = [
        ("Pure Monetary Policy", "PurePolicy_std", "1y1y"),
        ("Central Bank Information", "Information_std", "1y1y"),
        ("Central Bank Information", "Information_std", "1y4y"),
    ]


    components = ["IC", "EC", "IRP"]
    results = lp_outputs["forward_kerssenfischer"]

    fig, axes = plt.subplots(len(selected_rows), 3, figsize=(12, 2.8 * len(selected_rows)), sharex=True, squeeze=False)

    for i, (label, shock, curve_point) in enumerate(selected_rows):
        for j, component in enumerate(components):
            ax = axes[i, j]

            plot_bootstrap_irf(
                results=results,
                shock=shock,
                curve_point=curve_point,
                component=component,
                rate_scale=rate_scale,
                confidence_level=confidence_level,
                ax=ax,
                title=""
            )

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

            ax.set_title(component if i == 0 else "")
            ax.set_ylabel(f"{label} ({curve_point})\nBasis points" if j == 0 else "")
            ax.set_xlabel("Horizon (business days)" if i == len(selected_rows) - 1 else "")

        # Common y-axis scale across components for each shock.
        lower = min(ax.get_ylim()[0] for ax in axes[i])
        upper = max(ax.get_ylim()[1] for ax in axes[i])

        for ax in axes[i]:
            ax.set_ylim(lower, upper)

    fig.suptitle("Pure monetary policy vs. central bank information", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _save_figure(fig, figure_dir / "selected_irfs" / "Kerssenfischer_combined_IRFs.png", dpi=300)


    # Bootstrap IRF grids: rows are maturities/forwards, columns are IC/EC/IRP.
    if not args.skip_irf_grids and bool(synthesis_cfg.get("plot_irf_grids", True)):
        print("\n" + "=" * 70)
        print("BOOTSTRAP IRF GRIDS")
        print("=" * 70)

        for name, frame in lp_outputs.items():
            curve_type = "forward" if name.startswith("forward_") else "spot"
            curve_order = DEFAULT_FORWARD_ORDER if curve_type == "forward" else None
            for shock in frame["shock"].drop_duplicates():
                display_shock = shock_labels.get(shock, shock)
                fig, _ = plot_bootstrap_irf_grid(results=frame, shock=shock, curve_type=curve_type, 
                                                 curve_order=curve_order, rate_scale=rate_scale,
                                                 confidence_level=confidence_level,
                                                 title=f"{display_shock} - {curve_type} rates")
                output_path = (figure_dir / "irf_grids" / f"{name}_{shock}.{figure_format}")
                _save_figure(fig, output_path, dpi=figure_dpi)
                print(output_path)

    # Hand-picked individual IRFs configured in local_proj.yaml
    selected_irfs = synthesis_cfg.get("selected_irfs", [])
    for selection in selected_irfs:
        source = selection["source"]
        if source not in lp_outputs:
            raise ValueError(f"Selected IRF source {source!r} is not in synthesis.irf_inputs.")
        shock = selection["shock"]
        curve_point = selection["curve_point"]
        component = selection["component"]
        display_shock = shock_labels.get(shock, shock)

        fig, _ = plot_bootstrap_irf(results=lp_outputs[source], shock=shock, curve_point=curve_point,
                                    component=component, rate_scale=rate_scale, confidence_level=confidence_level,
                                    title=f"{display_shock} - {curve_point} - {component}")
        output_path = (figure_dir / "selected_irfs" / f"{source}_{shock}_{curve_point}_{component}.{figure_format}")
        _save_figure(fig, output_path, dpi=figure_dpi)

    # Synthetic measures are computed ON FORWARDS ONLY to avoid 
    # double-counting the same term-structure information through spots + forwards.
    forward_names = list(synthesis_cfg.get("forward_inputs", DEFAULT_FORWARD_FILES))
    missing_forward_sources = [name for name in forward_names if name not in lp_outputs]
    if missing_forward_sources:
        extra_outputs = _load_lp_outputs(lp_output_dir=lp_output_dir, names=missing_forward_sources)
        lp_outputs.update(extra_outputs)
    forward_frames = []
    for name in forward_names:
        frame = lp_outputs[name].copy()
        frame.insert(0, "lp_source", name)
        forward_frames.append(frame)
    forward_results = pd.concat(forward_frames, ignore_index=True)

    # Shock names must be unique across source files because the statistics are
    # grouped by shock. 
    source_counts = (forward_results[["lp_source", "shock"]].drop_duplicates()
                    .groupby("shock")["lp_source"].nunique())
    duplicated_across_sources = source_counts[source_counts > 1]
    if not duplicated_across_sources.empty:
        raise ValueError("The same shock name appears in multiple forward input files. "
                         "Rename shocks or concatenate the intended source before synthesis: "
                        f"{duplicated_across_sources.index.tolist()}")

    measures = compute_synthetic_measures(results=forward_results, curve_type="forward", 
                                          far_forward_points=synthesis_cfg.get("far_forward_points"),
        far_forward_start=float(synthesis_cfg.get("far_forward_start", 5.0)),
        long_end_point=synthesis_cfg.get("long_end_point", "5y5y"),
        persistence_horizon=int(synthesis_cfg.get("persistence_horizon", 20)),
        rate_scale=rate_scale,
        shock_labels=shock_labels)

    workbook_path = output_dir / synthesis_cfg.get("workbook_name", "lp_synthetic_measures.xlsx")
    export_synthetic_measures_excel(measures=measures, path=workbook_path)

    print("\n" + "=" * 70)
    print("SYNTHETIC MEASURES")
    print("=" * 70)
    print(measures["summary"].to_string(index=False))
    print(f"\nExcel output: {workbook_path}")

    # Cross-shock synthesis figures
    if not args.skip_synthesis_figures and bool(synthesis_cfg.get("plot_synthesis_figures", True)):
        summary = measures["summary"]
        # Persistence and propagation plots
        fig, _ = plot_propagation_persistence(summary=summary, component="EC")
        _save_figure(fig, figure_dir / f"figure_25_ec_propagation_persistence.{figure_format}", dpi=figure_dpi)
        fig, _ = plot_propagation_persistence(summary=summary, component="IRP")
        _save_figure(fig, figure_dir / f"figure_26_irp_propagation_persistence.{figure_format}", dpi=figure_dpi)

        # Contribution shares
        fig, _ = plot_irp_contribution_shares(summary=summary)
        _save_figure(fig, figure_dir / f"figure_27_irp_contribution.{figure_format}", dpi=figure_dpi)
        # Gravity centres
        fig, _ = plot_maturity_centres(summary=summary)
        _save_figure(fig, figure_dir / f"figure_28_maturity_centres.{figure_format}", dpi=figure_dpi)

        # EC vs IRP scatter plot by shocks
        fig, _ = plot_ec_irp_scatter(results=forward_results, rate_scale=rate_scale, shock_labels=shock_labels)
        _save_figure(fig, figure_dir / f"figure_29_ec_irp_scatter.{figure_format}", dpi=figure_dpi)

    print("\n" + "=" * 70)
    print("LP SYNTHESIS COMPLETED")
    print("=" * 70)
    print(f"Outputs: {output_dir}")

if __name__ == "__main__":
    main()
