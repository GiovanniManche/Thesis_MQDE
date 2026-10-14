################################################
# measures_and_charts.py
# Plotting and cross-shock synthesis functions for LPs.
#
# Synthetic statistics are computed on one representation of the term
# structure at a time. The baseline application uses forwards only.
################################################

from __future__ import annotations
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
REQUIRED_LP_COLUMNS = {"curve_point", "component", "shock", "horizon", "beta", "bootstrap_ci_lower", "bootstrap_ci_upper"}
DEFAULT_COMPONENT_ORDER = ["IC", "EC", "IRP"]
DEFAULT_FORWARD_ORDER = ["1y1y", "1y2y", "1y3y", "1y4y", "1y5y", "1y6y", "1y7y", "1y8y", "1y9y", "5y5y"]
# Main thesis palette
COMPONENT_COLORS = {
    "IC": "#4D4D4D",
    "EC": "#4C72B0",
    "IRP": "#C44E52",
}
SHOCK_COLORS = {
    "Target": "#4D4D4D",
    "Timing": "#4C72B0",
    "Forward Guidance": "#C44E52",
    "QE": "#5B8266",
    "GPR - energy": "#DDA15E",
    "GPR - non-energy": "#457B9D",
    "Pure monetary policy": "#8172B2",
    "Central bank information": "#CCB974",
}

SHOCK_MARKERS = {
    "Target": "o",
    "Timing": "s",
    "Forward Guidance": "^",
    "QE": "D",
    "GPR - energy": "v",
    "GPR - non-energy": "P",
    "Pure monetary policy": "X",
    "Central bank information": "*",
}


# Loading and curve-point helpers
def load_bootstrap_lp_results(path: str | Path) -> pd.DataFrame:
    """
    Load one tidy LP bootstrap output exported by 05_local_proj.py.
    """
    data = pd.read_csv(path)
    missing = REQUIRED_LP_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f'Missing LP result columns in {path}: {sorted(missing)}')
    data = data.copy()
    data['component'] = data['component'].astype(str).str.upper()
    data['shock'] = data['shock'].astype(str)
    data['horizon'] = pd.to_numeric(data['horizon'], errors='raise').astype(int)
    numeric_columns = ['beta', 'bootstrap_ci_lower', 'bootstrap_ci_upper']
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    if data[numeric_columns].isna().any().any():
        raise ValueError(f'LP result file {path} contains missing bootstrap IRF values.')
    duplicates = data.duplicated(subset=['curve_point', 'component', 'shock', 'horizon'])
    if duplicates.any():
        raise ValueError(f'LP result file {path} contains duplicate IRF observations.')
    return data

def _normalise_curve_point(value) -> str:
    """
    Return a lower-case string representation used for curve-point matching.
    """
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    return str(value).strip().lower()

def _resolve_curve_point(data: pd.DataFrame, curve_point):
    """
    Match a requested curve point while preserving the value stored in the data.
    """
    target = _normalise_curve_point(curve_point)
    available = data['curve_point'].drop_duplicates().tolist()
    matches = [value for value in available if _normalise_curve_point(value) == target]
    if len(matches) != 1:
        raise ValueError(f'Could not uniquely match curve point {curve_point!r}. Available curve points: {available}')
    return matches[0]

def forward_start_year(forward_label: str) -> float:
    """
    Extract the forward starting date, in years, from labels such as 1y5y or 5y5y.

    NOTE: the convention is xYnY with x the maturity of the forward and n the forward start year
    """
    label = _normalise_curve_point(forward_label)
    match = re.fullmatch('([0-9]+(?:\\.[0-9]+)?)y([0-9]+(?:\\.[0-9]+)?)y', label)
    if match is None:
        raise ValueError(f'Could not parse forward label {forward_label!r}.')
    return float(match.group(2))

def curve_coordinate(curve_point, curve_type: str='forward') -> float:
    """
    Return the maturity coordinate used in centre-of-gravity calculations.

    For spot rates this is the spot maturity. For forwards it is the forward

    start year rather than the terminal maturity.
    """
    curve_type = curve_type.lower()
    if curve_type == 'forward':
        return forward_start_year(str(curve_point))
    if curve_type == 'spot':
        return float(curve_point)
    raise ValueError("curve_type must be either 'spot' or 'forward'.")

def order_curve_points(data: pd.DataFrame, curve_type: str, curve_order: list | None=None) -> list:
    """
    Return curve points in an economically meaningful plotting order.
    """
    available = data['curve_point'].drop_duplicates().tolist()
    if curve_order is not None:
        ordered = []
        for requested in curve_order:
            target = _normalise_curve_point(requested)
            for value in available:
                if _normalise_curve_point(value) == target and value not in ordered:
                    ordered.append(value)
        remaining = [value for value in available if value not in ordered]
    else:
        ordered = []
        remaining = available
    if curve_type.lower() == 'spot':
        remaining = sorted(remaining, key=float)
    elif curve_type.lower() == 'forward':
        if curve_order is None:
            default_rank = {label: index for index, label in enumerate(DEFAULT_FORWARD_ORDER)}
            remaining = sorted(
                remaining,
                key=lambda value: (
                    default_rank.get(_normalise_curve_point(value), 10_000),
                    curve_coordinate(value, "forward"),
                    _normalise_curve_point(value),
                ),
            )
    else:
        raise ValueError("curve_type must be either 'spot' or 'forward'.")
    return ordered + remaining



# IRF plots
def plot_bootstrap_irf(
    results: pd.DataFrame, shock: str, curve_point, component: str, rate_scale: float = 10_000.0,
    confidence_level: float = 0.90, ax=None, title: str | None = None, show_asymptotic: bool = False):
    """
    Plot one bootstrap local-projection IRF.

    Inputs:
    ----------
        :results: tidy LP bootstrap output.
        :shock: shock coefficient to plot.
        :curve_point: spot maturity or forward label.
        :component: IC, EC or IRP.
        :rate_scale: scale applied to rates. 10,000 converts decimal rates to bp.
        :confidence_level: confidence level displayed in the legend.
        :ax: optional matplotlib axis.
        :title: optional plot title.
        :show_asymptotic: if True, also plot the original Newey-West interval.

    Output:
    ----------
        Matplotlib Figure and Axis.
    """
    component = component.upper()
    matched_curve_point = _resolve_curve_point(results, curve_point)
    subset = results.loc[
        (results["shock"] == shock)
        & (results["curve_point"] == matched_curve_point)
        & (results["component"] == component)
    ].sort_values("horizon")
    if subset.empty:
        raise ValueError(f'No IRF found for shock={shock}, curve_point={curve_point}, component={component}.')
    color = COMPONENT_COLORS[component]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
    else:
        fig = ax.figure
    horizon = subset['horizon'].to_numpy()
    beta = rate_scale * subset['beta'].to_numpy()
    lower = rate_scale * subset['bootstrap_ci_lower'].to_numpy()
    upper = rate_scale * subset['bootstrap_ci_upper'].to_numpy()
    ax.axhline(0.0, linewidth=0.8)
    ax.fill_between(horizon, lower, upper, color=color, alpha=0.2, label=f'Bootstrap {int(round(100 * confidence_level))}% CI')
    ax.plot(horizon, beta, linewidth=1.5, label='Point estimate', color=color)
    if show_asymptotic:
        required = {'ci_lower', 'ci_upper'}
        if not required.issubset(results.columns):
            raise ValueError('Asymptotic confidence interval columns are missing.')
        ax.plot(horizon, rate_scale * subset['ci_lower'].to_numpy(), linestyle='--', linewidth=0.8, label='NW interval', color=color)
        ax.plot(horizon, rate_scale * subset['ci_upper'].to_numpy(), linestyle='--', linewidth=0.8, color=color)
    ax.set_xlabel('Horizon (business days)')
    ax.set_ylabel('Basis points')
    if title is None:
        title = f'{shock} - {matched_curve_point} - {component}'
    ax.set_title(title)
    ax.legend(frameon=False)
    return (fig, ax)

def plot_bootstrap_irf_grid(
    results: pd.DataFrame, shock: str, curve_type: str, curve_order: list | None = None,
    component_order: list[str] | None = None, rate_scale: float = 10_000.0,
    confidence_level: float = 0.90, same_y_within_row: bool = True, title: str | None = None,
):
    """
    Plot one shock with curve points in rows and IC/EC/IRP in columns.
    The same y-axis range is used across IC, EC and IRP within each row by
    default, making the decomposition directly comparable at a given maturity.
    """
    subset = results.loc[results['shock'] == shock].copy()
    if subset.empty:
        raise ValueError(f'Shock {shock!r} is not present in the LP results.')
    components = component_order or DEFAULT_COMPONENT_ORDER
    components = [component.upper() for component in components]
    curve_points = order_curve_points(data=subset, curve_type=curve_type, curve_order=curve_order)
    nrows = len(curve_points)
    ncols = len(components)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 2.55 * nrows), squeeze=False, sharex=True)
    # Each row is one maturity/forward; columns are IC, EC and IRP.
    for row, curve_point in enumerate(curve_points):
        row_min = np.inf
        row_max = -np.inf
        for col, component in enumerate(components):
            ax = axes[row, col]
            color = COMPONENT_COLORS[component]
            cell = subset.loc[(subset['curve_point'] == curve_point) & (subset['component'] == component)].sort_values('horizon')
            if cell.empty:
                ax.set_visible(False)
                continue
            horizon = cell['horizon'].to_numpy()
            beta = rate_scale * cell['beta'].to_numpy()
            lower = rate_scale * cell['bootstrap_ci_lower'].to_numpy()
            upper = rate_scale * cell['bootstrap_ci_upper'].to_numpy()
            ax.axhline(0.0, linewidth=0.7, color='#4D4D4D', alpha=0.7)
            ax.fill_between(horizon, lower, upper, alpha=0.2, color=color, linewidth=0.0)
            ax.plot(horizon, beta, linewidth=1.3, color=color)
            row_min = min(row_min, float(np.nanmin(lower)))
            row_max = max(row_max, float(np.nanmax(upper)))
            if row == 0:
                ax.set_title(component)
            if col == 0:
                ax.set_ylabel(f'{curve_point}\nBasis points')
            if row == nrows - 1:
                ax.set_xlabel('Horizon (business days)')
        # Use one y-scale across IC/EC/IRP for a given curve point.
        if same_y_within_row and np.isfinite(row_min) and np.isfinite(row_max):
            padding = 0.08 * max(row_max - row_min, 1e-08)
            lower_limit = min(row_min - padding, 0.0)
            upper_limit = max(row_max + padding, 0.0)
            for col in range(ncols):
                if axes[row, col].get_visible():
                    axes[row, col].set_ylim(lower_limit, upper_limit)
    if title is None:
        title = f'Bootstrap local projections - {shock}'
    fig.suptitle(f'{title} ({int(round(100 * confidence_level))}% confidence bands)', y=1.002)
    fig.tight_layout()
    return (fig, axes)


# Synthetic measures
def _safe_ratio(numerator: float, denominator: float) -> float:
    """
    Return a ratio or NaN when the denominator is numerically zero.
    """
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return np.nan
    return float(numerator / denominator)

def _bounded_ratio(numerator: float, denominator: float, tolerance: float=1e-08) -> float:
    """
    Compute a theoretically [0, 1] ratio and guard against material violations.
    """
    value = _safe_ratio(numerator, denominator)
    if not np.isfinite(value):
        return value
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(f'Theoretically bounded ratio lies outside [0, 1]: {value}')
    return float(np.clip(value, 0.0, 1.0))

def _resolve_far_forward_points(data: pd.DataFrame, curve_type: str, far_forward_points: list | None, far_forward_start: float) -> list:
    """
    Resolve the set F used in long-end propagation and persistence statistics.
    """
    available = data['curve_point'].drop_duplicates().tolist()
    if far_forward_points is not None:
        resolved = []
        for curve_point in far_forward_points:
            resolved.append(_resolve_curve_point(data, curve_point))
        return list(dict.fromkeys(resolved))
    return [curve_point for curve_point in available if curve_coordinate(curve_point, curve_type) >= far_forward_start]

def compute_synthetic_measures(
    results: pd.DataFrame, curve_type: str = "forward", far_forward_points: list | None = None,
    far_forward_start: float = 5.0, long_end_point = "5y5y", persistence_horizon: int = 20,
    rate_scale: float = 10_000.0, shock_labels: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute the cross-shock synthetic statistics from point-estimate IRFs. All magnitude statistics 
    are computed after conversion to basis points.

    Inputs:
    ----------
        :results: tidy LP results for one term-structure representation.
        :curve_type: "forward" for the baseline synthesis or "spot" for robustness.
        :far_forward_points: explicit set F. If None, points starting/maturing at
                             or beyond far_forward_start are used.
        :far_forward_start: threshold in years used when F is not supplied.
        :long_end_point: representative long-end point, baseline 5y5y.
        :persistence_horizon: h* entering the persistence statistic.
        :rate_scale: rate conversion factor, baseline 10,000 for basis points.
        :shock_labels: optional mapping from variable names to display names.

    Output:
    ----------
        Dictionary containing summary, by_curve_point, long_end_peaks and metadata
        DataFrames.
    """
    required = {'curve_point', 'component', 'shock', 'horizon', 'beta'}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f'Missing LP columns: {sorted(missing)}')
    data = results.copy()
    data['component'] = data['component'].astype(str).str.upper()
    data['shock'] = data['shock'].astype(str)
    data['horizon'] = pd.to_numeric(data['horizon'], errors='raise').astype(int)
    data['beta'] = pd.to_numeric(data['beta'], errors='coerce')
    if data['beta'].isna().any():
        raise ValueError('Point-estimate IRFs contain missing beta values.')
    required_components = set(DEFAULT_COMPONENT_ORDER)
    missing_components = required_components.difference(data['component'].unique())
    if missing_components:
        raise ValueError(f'Missing LP components: {sorted(missing_components)}')
    duplicate_keys = data.duplicated(subset=['shock', 'curve_point', 'component', 'horizon'])
    if duplicate_keys.any():
        raise ValueError('Synthetic-measure input contains duplicate IRF observations.')
    max_horizon = int(data['horizon'].max())
    if persistence_horizon < 0 or persistence_horizon > max_horizon:
        raise ValueError(f'persistence_horizon must lie between 0 and {max_horizon}.')
    long_end_value = _resolve_curve_point(data, long_end_point)
    far_points = _resolve_far_forward_points(data=data, curve_type=curve_type, 
                                            far_forward_points=far_forward_points, far_forward_start=far_forward_start)
    if not far_points:
        raise ValueError('The far-forward set is empty.')
    # Convert the point-estimate IRFs to basis points before computing magnitudes.
    data['response_bp'] = rate_scale * data['beta']
    data['abs_response_bp'] = data['response_bp'].abs()
    data['curve_coordinate_years'] = data['curve_point'].map(lambda value: curve_coordinate(value, curve_type))
    data['is_far_forward'] = data['curve_point'].isin(far_points)
    # A_{s,n}^c: cumulative absolute response over LP horizons at each curve point.
    by_curve = (
        data.groupby(
            ["shock", "component", "curve_point", "curve_coordinate_years", "is_far_forward"],
            as_index=False, sort=False,
        )["abs_response_bp"]
        .sum()
        .rename(columns={"abs_response_bp": "A_bp_days"})
    )
    if shock_labels is not None:
        by_curve.insert(1, 'shock_label', by_curve['shock'].map(shock_labels).fillna(by_curve['shock']))
    # Peak response at the representative long-end point (5y5y by default).
    long_end = data.loc[data['curve_point'] == long_end_value].copy()
    peak_rows = []
    for (shock, component), group in long_end.groupby(['shock', 'component'], sort=False):
        peak_index = group['response_bp'].abs().idxmax()
        peak = group.loc[peak_index]
        peak_rows.append({'shock': shock, 'shock_label': shock_labels.get(shock, shock) if shock_labels is not None else 
                          shock, 'component': component, 'long_end_point': long_end_value, 'M_long_bp': float(abs(peak['response_bp'])), 
                          'signed_peak_bp': float(peak['response_bp']), 'peak_horizon': int(peak['horizon'])})
    long_end_peaks = pd.DataFrame(peak_rows)
    # Put IC, EC and IRP side by side to check additivity and measure cancellation/reinforcement.
    response_wide = data.pivot(index=['shock', 'curve_point', 'horizon', 'is_far_forward'], columns='component', values='response_bp').reset_index()
    max_additivity_error = float(np.nanmax(np.abs(response_wide['IC'] - response_wide['EC'] - response_wide['IRP'])))
    summary_rows = []
    for shock, shock_data in data.groupby('shock', sort=False):
        row = {'shock': shock, 'shock_label': shock_labels.get(shock, shock) if shock_labels is not None else shock}
        component_total = {}
        component_long = {}
        for component in DEFAULT_COMPONENT_ORDER:
            component_data = shock_data.loc[shock_data['component'] == component]
            total_area = float(component_data['abs_response_bp'].sum())
            long_area = float(component_data.loc[component_data['curve_point'].isin(far_points), 'abs_response_bp'].sum())
            late_long_area = float(
                component_data.loc[
                    component_data["curve_point"].isin(far_points)
                    & (component_data["horizon"] >= persistence_horizon),
                    "abs_response_bp",
                ].sum()
            )
            point_area = by_curve.loc[(by_curve['shock'] == shock) & (by_curve['component'] == component)].copy()
            centre_numerator = float((point_area['curve_coordinate_years'] * point_area['A_bp_days']).sum())
            component_total[component] = total_area
            component_long[component] = long_area
            row[f'A_{component}_bp_days'] = total_area
            row[f'L_{component}'] = _bounded_ratio(long_area, total_area)
            row[f'CG_{component}_years'] = _safe_ratio(centre_numerator, total_area)
            row[f'P_{component}'] = _bounded_ratio(late_long_area, long_area)
            peak = long_end_peaks.loc[(long_end_peaks['shock'] == shock) & (long_end_peaks['component'] == component)]
            if peak.empty:
                row[f'M_{component}_long_bp'] = np.nan
                row[f'M_{component}_long_horizon'] = np.nan
                row[f'M_{component}_long_signed_bp'] = np.nan
            else:
                peak = peak.iloc[0]
                row[f'M_{component}_long_bp'] = float(peak['M_long_bp'])
                row[f'M_{component}_long_horizon'] = int(peak['peak_horizon'])
                row[f'M_{component}_long_signed_bp'] = float(peak['signed_peak_bp'])
        row['deanchoring_index'] = row['L_EC'] * row['P_EC']
        row['risk_repricing_index'] = row['L_IRP'] * row['P_IRP']
        contribution_denom = component_total['EC'] + component_total['IRP']
        row['share_IRP'] = _bounded_ratio(component_total['IRP'], contribution_denom)
        row['share_EC'] = 1.0 - row['share_IRP'] if np.isfinite(row['share_IRP']) else np.nan
        long_contribution_denom = component_long['EC'] + component_long['IRP']
        row['share_IRP_long'] = _bounded_ratio(component_long['IRP'], long_contribution_denom)
        row['share_EC_long'] = 1.0 - row['share_IRP_long'] if np.isfinite(row['share_IRP_long']) else np.nan
        shock_wide = response_wide.loc[response_wide['shock'] == shock]
        reinforcement_denom = float((shock_wide['EC'].abs() + shock_wide['IRP'].abs()).sum())
        reinforcement_numer = float(shock_wide['IC'].abs().sum())
        row['reinforcement_index'] = _bounded_ratio(reinforcement_numer, reinforcement_denom)
        row['cancellation_index'] = 1.0 - row['reinforcement_index'] if np.isfinite(row['reinforcement_index']) else np.nan
        shock_long_wide = shock_wide.loc[shock_wide['is_far_forward']]
        reinforcement_long_denom = float((shock_long_wide['EC'].abs() + shock_long_wide['IRP'].abs()).sum())
        reinforcement_long_numer = float(shock_long_wide['IC'].abs().sum())
        row['reinforcement_index_long'] = _bounded_ratio(reinforcement_long_numer, reinforcement_long_denom)
        row['cancellation_index_long'] = 1.0 - row['reinforcement_index_long'] if np.isfinite(row['reinforcement_index_long']) else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    preferred_columns = ['shock', 'shock_label', 'A_EC_bp_days', 'A_IRP_bp_days', 'M_EC_long_bp', 
                        'M_EC_long_signed_bp', 'M_EC_long_horizon', 'M_IRP_long_bp', 'M_IRP_long_signed_bp', 
                        'M_IRP_long_horizon', 'L_EC', 'L_IRP', 'CG_EC_years', 'CG_IRP_years', 'P_EC', 'P_IRP', 
                        'deanchoring_index', 'risk_repricing_index', 'share_EC', 'share_IRP', 'share_EC_long', 'share_IRP_long', 
                        'reinforcement_index', 'cancellation_index', 'reinforcement_index_long', 'cancellation_index_long']
    trailing = [column for column in summary.columns if column not in preferred_columns]
    summary = summary[[column for column in preferred_columns if column in summary.columns] + trailing]
    metadata = pd.DataFrame({'setting': ['curve_type', 'max_horizon', 'persistence_horizon', 'long_end_point', 
                                         'far_forward_points', 'rate_scale', 'max_abs_ic_additivity_error_bp'], 
                                         'value': [curve_type, max_horizon, persistence_horizon, str(long_end_value), ', '.join((
                                             str(value) for value in far_points)), rate_scale, max_additivity_error]})
    return {'summary': summary, 'by_curve_point': by_curve, 'long_end_peaks': long_end_peaks, 'metadata': metadata}


# Cross-shock synthesis charts
def plot_propagation_persistence(summary: pd.DataFrame, component: str, bubble_scale: float=1600.0, min_bubble_size: float=45.0, title: str | None=None):
    """
    Plot propagation against persistence, with long-end magnitude as bubble size.
    """
    component = component.upper()
    if component not in {'EC', 'IRP'}:
        raise ValueError('component must be EC or IRP.')
    x_column = f'L_{component}'
    y_column = f'P_{component}'
    magnitude_column = f'M_{component}_long_bp'
    required = {x_column, y_column, magnitude_column}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f'Missing synthesis columns: {sorted(missing)}')
    data = summary.dropna(subset=[x_column, y_column, magnitude_column]).copy()
    max_magnitude = float(data[magnitude_column].max()) if not data.empty else 0.0
    if max_magnitude > 0:
        sizes = min_bubble_size + bubble_scale * data[magnitude_column] / max_magnitude
    else:
        sizes = np.full(len(data), min_bubble_size)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(data[x_column], data[y_column], s=sizes, color=COMPONENT_COLORS['IRP'], alpha=0.65, edgecolor='none')
    label_column = 'shock_label' if 'shock_label' in data.columns else 'shock'
    for _, row in data.iterrows():
        ax.annotate(str(row[label_column]), (row[x_column], row[y_column]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f'Long-end propagation share, $L^{{{component}}}$')
    ax.set_ylabel(f'Persistence share, $P^{{{component}}}$')
    if title is None:
        title = 'Expected-inflation propagation and persistence' if component == 'EC' else 'Inflation-risk-premium propagation and persistence'
    ax.set_title(title)
    fig.tight_layout()
    return (fig, ax)

def plot_irp_contribution_shares(summary: pd.DataFrame, title: str='Expected-inflation versus inflation-risk-premium contribution'):
    """
    Compare total and long-end IRP contribution shares across shocks.
    """
    required = {'share_IRP', 'share_IRP_long'}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f'Missing synthesis columns: {sorted(missing)}')
    data = summary.copy()
    labels = data['shock_label'].astype(str).tolist() if 'shock_label' in data.columns else data['shock'].astype(str).tolist()
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(data)), 4.8))
    ax.bar(x - width / 2, data['share_IRP'], width=width, label='All forwards', color=COMPONENT_COLORS['EC'])
    ax.bar(x + width / 2, data['share_IRP_long'], width=width, label='Far forwards', color=COMPONENT_COLORS['IRP'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('IRP contribution share')
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return (fig, ax)

def plot_maturity_centres(summary: pd.DataFrame, title: str='Maturity centre of gravity of shock transmission'):
    """
    Compare the EC and IRP maturity centres of gravity across shocks.
    """
    required = {'CG_EC_years', 'CG_IRP_years'}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f'Missing synthesis columns: {sorted(missing)}')
    data = summary.copy()
    labels = data['shock_label'].astype(str).tolist() if 'shock_label' in data.columns else data['shock'].astype(str).tolist()
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(data)), 4.8))
    ax.bar(x - width / 2, data['CG_EC_years'], width=width, label='EC', color=COMPONENT_COLORS['EC'])
    ax.bar(x + width / 2, data['CG_IRP_years'], width=width, label='IRP', color=COMPONENT_COLORS['IRP'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('Forward start / maturity (years)')
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return (fig, ax)

def plot_ec_irp_scatter(
    results: pd.DataFrame, rate_scale: float = 10_000.0, shock_labels: dict[str, str] | None = None,
    title: str = "Expectation revision versus inflation-risk repricing",
):
    """
    Plot EC against IRP responses for every shock, curve point and horizon. Northeast and southwest points 
    correspond to reinforcement between EC and IRP. Southeast and northwest points correspond to partial or complete
    cancellation in inflation compensation.
    Shocks are distinguished by both color and marker.
    """
    data = results.copy()
    data['component'] = data['component'].astype(str).str.upper()
    data['response_bp'] = rate_scale * pd.to_numeric(data['beta'], errors='coerce')
    wide = (data.loc[data["component"].isin(["EC", "IRP"])]
        .pivot(index=["shock", "curve_point", "horizon"], columns="component", values="response_bp")
        .reset_index())
    wide = wide.dropna(subset=['EC', 'IRP'])
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for shock, shock_data in wide.groupby('shock', sort=False):
        label = shock_labels.get(shock, shock) if shock_labels is not None else shock
        color = SHOCK_COLORS.get(label, COMPONENT_COLORS['IC'])
        marker = SHOCK_MARKERS.get(label, 'o')
        ax.scatter(shock_data['EC'], shock_data['IRP'], s=24, alpha=0.55, color=color, 
                   marker=marker, label=label, edgecolor='none')
    ax.axhline(0.0, linewidth=0.8, color=COMPONENT_COLORS['IC'], alpha=0.7)
    ax.axvline(0.0, linewidth=0.8, color=COMPONENT_COLORS['IC'], alpha=0.7)
    max_abs = float(np.nanmax(np.abs(wide[['EC', 'IRP']].to_numpy()))) if not wide.empty else 1.0
    if max_abs > 0:
        limit = 1.05 * max_abs
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
    ax.set_xlabel('Expected-inflation response (bp)')
    ax.set_ylabel('Inflation-risk-premium response (bp)')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return (fig, ax)


def export_synthetic_measures_excel(measures: dict[str, pd.DataFrame], path: str | Path) -> Path:
    """
    Export synthetic measures to a formatted Excel workbook.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, frame in measures.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = 'A2'
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else '' for cell in column_cells]
                width = min(max(max((len(value) for value in values)) + 2, 10), 32)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = '0.0000'
    return path
