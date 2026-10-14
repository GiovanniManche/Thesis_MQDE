################################################
# local_projection.py
# Local-projection estimation for the daily ILS decomposition.
#
# The baseline specification is estimated separately for every curve point,
# component and horizon:
#
#   y_{t+h} - y_{t-1} = alpha_h + beta_h' shock_t
#                       + sum_j rho_{h,j} Delta IC_{t-j} + u_{t,h}
#
# where IC, expected inflation and the inflation risk premium are always
# estimated on the same regression sample. This preserves the exact OLS
# additivity beta_IC = beta_EC + beta_IRP whenever IC = EC + IRP.
################################################

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _prepare_shocks(
    shocks: pd.Series | pd.DataFrame,
    shock_cols: list[str] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Put shock series in a numeric DataFrame indexed by normalized dates.

    Dates explicitly present in the input are kept as event dates even when
    one of the shock values is missing. This allows the LP routine to preserve
    explicit event-date NaNs while filling genuinely non-event dates with zero.
    """
    if isinstance(shocks, pd.Series):
        data = shocks.to_frame(name=shocks.name if shocks.name is not None else "shock")
    elif isinstance(shocks, pd.DataFrame):
        data = shocks.copy()
    else:
        raise TypeError("shocks must be a pandas Series or DataFrame.")

    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col]).dt.normalize()
        data = data.set_index(date_col)
    elif isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index).normalize()
    else:
        raise ValueError("shocks must have either a date column or a DatetimeIndex.")

    if data.index.has_duplicates:
        raise ValueError("Shock data contain duplicate dates.")

    if shock_cols is not None:
        missing = [column for column in shock_cols if column not in data.columns]
        if missing:
            raise ValueError(f"Missing shock columns: {missing}")
        data = data[shock_cols].copy()
    else:
        numeric_columns = data.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            raise ValueError("No numeric shock columns found.")
        data = data[numeric_columns].copy()

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.columns.duplicated().any():
        raise ValueError("Shock column names must be unique.")

    return data.sort_index()


def _prepare_controls(
    controls: pd.DataFrame | None,
    date_col: str = "date",
) -> pd.DataFrame | None:
    """
    Put optional external controls in a numeric DataFrame indexed by dates.

    The routine does not lag external controls automatically. They should be
    supplied exactly as they are meant to enter the local projection.
    """
    if controls is None:
        return None

    data = controls.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col]).dt.normalize()
        data = data.set_index(date_col)
    elif isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index).normalize()
    else:
        raise ValueError("controls must have either a date column or a DatetimeIndex.")

    if data.index.has_duplicates:
        raise ValueError("Control matrix contains duplicate dates.")

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.sort_index()


def estimate_lp_irfs(
    decomposition: pd.DataFrame,
    shocks: pd.Series | pd.DataFrame,
    max_horizon: int = 25,
    ic_lags: int = 5,
    shock_lags: int = 0,
    controls: pd.DataFrame | None = None,
    confidence_level: float = 0.90,
    date_col: str = "date",
    curve_col: str = "maturity",
    ic_col: str = "fitted",
    ec_col: str = "expected",
    irp_col: str = "irp",
    shock_cols: list[str] | None = None,
    nw_lags: int | str = "horizon",
    fill_non_event_dates_with_zero: bool = True,
    return_covariances: bool = False,
):
    """
    Estimate daily local projections for IC, expected inflation and the IRP.

    Inputs:
    ----------
        :decomposition: long-format daily spot or forward decomposition.
        :shocks: event-time shocks as a Series or DataFrame.
        :max_horizon: maximum response horizon in trading days.
        :ic_lags: number of lags of daily changes in fitted IC included as controls.
        :shock_lags: optional number of lags of the shock variables. Baseline is zero.
        :controls: optional additional controls already transformed/lagged as desired.
        :confidence_level: confidence level for the Newey-West intervals.
        :date_col: date-column name.
        :curve_col: curve-point column, e.g. maturity for spots or forward for forwards.
        :ic_col: inflation-compensation column used both as outcome and lag control.
        :ec_col: expected-inflation column.
        :irp_col: inflation-risk-premium column.
        :shock_cols: shock columns estimated jointly.
        :nw_lags: either "horizon" or a fixed non-negative Newey-West lag order.
        :fill_non_event_dates_with_zero: fill dates absent from the shock dataset with zero.
        :return_covariances: if True, also return covariance matrices for the contemporaneous shock coefficients.

    Output:
    ----------
        Dictionary with one wide DataFrame for IC, EC and IRP. Columns are a
        MultiIndex (shock, horizon, statistic). If return_covariances=True, a
        second dictionary keyed by (component, curve_point, horizon) contains
        the Newey-West covariance matrix of the jointly estimated shock coefficients.
    """
    if max_horizon < 0:
        raise ValueError("max_horizon must be non-negative.")
    if ic_lags < 0 or shock_lags < 0:
        raise ValueError("ic_lags and shock_lags must be non-negative.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    data = decomposition.copy()
    required_columns = {date_col, curve_col, ic_col, ec_col, irp_col}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing decomposition columns: {sorted(missing_columns)}")

    data[date_col] = pd.to_datetime(data[date_col]).dt.normalize()
    if data.duplicated(subset=[date_col, curve_col]).any():
        raise ValueError("Decomposition contains duplicate date-curve-point observations.")
    data = data.sort_values([curve_col, date_col])

    shocks_data = _prepare_shocks(shocks=shocks, shock_cols=shock_cols, date_col=date_col)
    controls_data = _prepare_controls(controls=controls, date_col=date_col)
    shock_names = shocks_data.columns.tolist()

    if controls_data is not None:
        overlap_names = set(shock_names).intersection(controls_data.columns)
        if overlap_names:
            raise ValueError(f"Shock and control columns share names: {sorted(overlap_names)}")

    component_columns = {"IC": ic_col, "EC": ec_col, "IRP": irp_col}
    statistics = [
        "beta",
        "std_error",
        "ci_lower",
        "ci_upper",
        "p_value",
        "nobs",
        "n_events",
        "r_squared",
        "nw_lags",
    ]
    records = {component: [] for component in component_columns}
    covariance_matrices = {}
    original_shock_dates = shocks_data.index

    for curve_point, curve_data in data.groupby(curve_col, sort=False):
        curve_data = curve_data.set_index(date_col).sort_index()
        ils_dates = curve_data.index

        aligned_shocks = shocks_data.reindex(ils_dates)
        if fill_non_event_dates_with_zero:
            non_event_dates = ~ils_dates.isin(original_shock_dates)
            aligned_shocks.loc[non_event_dates, shock_names] = 0.0

        regressors = aligned_shocks.copy()

        if shock_lags > 0:
            for lag in range(1, shock_lags + 1):
                for shock_name in shock_names:
                    regressors[f"{shock_name}_lag{lag}"] = aligned_shocks[shock_name].shift(lag)

        component_series = {
            component: pd.to_numeric(curve_data[column], errors="coerce")
            for component, column in component_columns.items()
        }

        # Predetermined controls based on the same fitted IC series for all
        # three outcomes preserve exact coefficient additivity.
        delta_ic = component_series["IC"].diff()
        for lag in range(1, ic_lags + 1):
            regressors[f"delta_ic_lag{lag}"] = delta_ic.shift(lag)

        if controls_data is not None:
            aligned_controls = controls_data.reindex(ils_dates)
            regressors = regressors.join(aligned_controls, how="left")

        regressor_columns = regressors.columns.tolist()

        for horizon in range(max_horizon + 1):
            dependent_variables = {
                component: series.shift(-horizon) - series.shift(1)
                for component, series in component_series.items()
            }

            regression_data = regressors.copy()
            for component in component_columns:
                regression_data[f"__y_{component}"] = dependent_variables[component]
            regression_data = regression_data.dropna()

            if regression_data.empty:
                raise ValueError(f"No usable observations for {curve_col}={curve_point}, horizon={horizon}.")

            X = sm.add_constant(regression_data[regressor_columns], has_constant="add")

            if isinstance(nw_lags, str):
                if nw_lags != "horizon":
                    raise ValueError("nw_lags must be either 'horizon' or a non-negative integer.")
                current_nw_lags = horizon
            else:
                current_nw_lags = int(nw_lags)
                if current_nw_lags < 0:
                    raise ValueError("nw_lags must be non-negative.")

            event_matrix = regression_data[shock_names].fillna(0.0).to_numpy(dtype=float)
            n_events = int(np.any(np.abs(event_matrix) > 0.0, axis=1).sum())

            for component in component_columns:
                y = regression_data[f"__y_{component}"]
                model = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": current_nw_lags, "use_correction": True},
                )

                confidence_intervals = model.conf_int(alpha=(1 - confidence_level))

                if return_covariances:
                    covariance = model.cov_params().loc[shock_names, shock_names].to_numpy(dtype=float)
                    covariance_matrices[(component, curve_point, horizon)] = covariance

                for shock_name in shock_names:
                    interval = confidence_intervals.loc[shock_name]
                    records[component].append(
                        {
                            "curve_point": curve_point,
                            "shock": shock_name,
                            "horizon": horizon,
                            "beta": float(model.params[shock_name]),
                            "std_error": float(model.bse[shock_name]),
                            "ci_lower": float(interval.iloc[0]),
                            "ci_upper": float(interval.iloc[1]),
                            "p_value": float(model.pvalues[shock_name]),
                            "nobs": int(model.nobs),
                            "n_events": n_events,
                            "r_squared": float(model.rsquared),
                            "nw_lags": current_nw_lags,
                        }
                    )

    desired_columns = pd.MultiIndex.from_tuples(
        [
            (shock_name, horizon, statistic)
            for shock_name in shock_names
            for horizon in range(max_horizon + 1)
            for statistic in statistics
        ],
        names=["shock", "horizon", "statistic"],
    )

    results = {}
    for component, component_records in records.items():
        long_results = pd.DataFrame(component_records)
        wide_results = (
            long_results
            .set_index(["curve_point", "shock", "horizon"])[statistics]
            .unstack(["shock", "horizon"])
        )
        wide_results = wide_results.reorder_levels([1, 2, 0], axis=1)
        wide_results.columns.names = ["shock", "horizon", "statistic"]
        wide_results = wide_results.reindex(columns=desired_columns)
        wide_results.index.name = curve_col
        results[component] = wide_results.sort_index()

    if return_covariances:
        return results, covariance_matrices
    return results


def lp_results_to_long(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert the wide LP result dictionary into one tidy DataFrame.
    """
    frames = []
    for component, table in results.items():
        long = table.stack(["shock", "horizon"], future_stack=True).reset_index()
        long.insert(1, "component", component)
        curve_name = table.index.name if table.index.name is not None else "curve_point"
        long = long.rename(columns={curve_name: "curve_point"})
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def max_additivity_error(results: dict[str, pd.DataFrame]) -> float:
    """
    Return max |beta_IC - beta_EC - beta_IRP| over all shocks, horizons and curve points.
    """
    beta_ic = results["IC"].xs("beta", axis=1, level="statistic")
    beta_ec = results["EC"].xs("beta", axis=1, level="statistic")
    beta_irp = results["IRP"].xs("beta", axis=1, level="statistic")
    error = beta_ic - beta_ec - beta_irp
    return float(np.nanmax(np.abs(error.to_numpy(dtype=float))))
