############################### 
# gpr_shocks.py
# We construct our geopolitical news shocks fromn daily GPR on which we fit an ARMA,
# take the 99th percentile of the residuals, and classify them using Brent futures returns.
################################  


from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

@dataclass
class GPRShockResult:
    """
    Dataclass to store the result of the GPR shock estimation, that is the shocks themselves but also model diagnostics
    """
    innovations: pd.DataFrame
    shocks: pd.DataFrame
    regression: pd.DataFrame
    model_selection: pd.DataFrame
    stationarity: pd.DataFrame
    autocorrelation: pd.DataFrame
    summary: pd.DataFrame


def load_gpr_data(file: str, sheet: str = "DailyEAGPR", date_column: str = "date", 
                  gpr_column: str = "Daily EA GPR", start_date: str | None = None, end_date: str | None = None) -> pd.Series:
    """
    Load and clean the daily GPR series.

    Inputs
    ------
        :file: path to the Excel file containing the GPR data.
        :sheet: name of the sheet in the Excel file.
        :date_column: name of the column containing the dates.
        :gpr_column: name of the column containing the GPR values.
        :start_date: start date for filtering the data.
        :end_date: end date for filtering the data.

    Output
    -------
        Cleaned and filtered GPR series indexed by date.    
    """
    # Imports and checks
    data = pd.read_excel(file, sheet_name=sheet, usecols=[date_column, gpr_column])
    if is_numeric_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column], unit="D", origin="1899-12-30")
    else:
        data[date_column] = pd.to_datetime(data[date_column])
    data[gpr_column] = pd.to_numeric(data[gpr_column], errors="coerce")
    data = data.dropna(subset=[date_column, gpr_column]).copy()
    data = data.sort_values(date_column).drop_duplicates(subset=date_column, keep="last")

    # Filtering on the dates (notably to match ILS sample)
    if start_date is not None:
        data = data.loc[data[date_column] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        data = data.loc[data[date_column] <= pd.Timestamp(end_date)].copy()
    if data.empty:
        raise ValueError("No GPR observations remain after applying the requested sample filters.")
    data = data.set_index(date_column)
    if data.index.has_duplicates:
        raise ValueError("The GPR dataset contains duplicate dates after cleaning.")

    # We get the frequency of the dataset (should be daily here)
    inferred_frequency = pd.infer_freq(data.index)
    if inferred_frequency is not None:
        data.index = pd.DatetimeIndex(data.index, freq=inferred_frequency)
    return data[gpr_column].rename("gpr")

def load_brent_data(file: str, sheet: str = "Sheet1", date_column: str = "date", price_column: str = "CO1 Generic 1st Brent Future") -> pd.Series:
    """
    Load and clean the Brent data.

    Inputs
    ------
        :file: path to the Excel file containing the Brent data.
        :sheet: name of the sheet in the Excel file.
        :date_column: name of the column containing the dates.
        :price_column: name of the column containing the Brent prices.

    Output
    -------
        Cleaned Brent price series indexed by date.
    """

    data = pd.read_excel(file, sheet_name=sheet, usecols=[date_column, price_column])
    if is_numeric_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column], unit="D", origin="1899-12-30")
    else:
        data[date_column] = pd.to_datetime(data[date_column])
    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")
    data = data.dropna(subset=[date_column, price_column]).copy()
    data = data.loc[data[price_column] > 0].copy()
    data = data.sort_values(date_column).drop_duplicates(subset=date_column, keep="last")
    if data.empty:
        raise ValueError("No valid Brent observations are available.")
    
    return data.set_index(date_column)[price_column].rename("brent")



def restrict_to_trading_dates(series: pd.Series, trading_dates: Iterable[pd.Timestamp]) -> pd.Series:
    """
    Used to restrict a daily series (GPR, Brent) to the trading dates used by the ILS sample.

    Inputs
    ------
        :series: daily series to be restricted.
        :trading_dates: trading dates to keep.

    Output
    -------
        Series restricted to the specified trading dates.
    """
    trading_dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize()
    series_dates = series.index.normalize()
    return series.loc[series_dates.isin(trading_dates)].copy()

def transform_gpr(gpr: pd.Series, difference_order: int = 1) -> pd.Series:
    """
    Helper function to automatise the differencing or not of the GPR before ARMA estimation.

    Inputs
    ------
        :gpr: GPR series to be transformed.
        :difference_order: order of differencing (0 for no differencing, 1 for first difference).

    Output
    -------
        Transformed GPR series.
    """

    if difference_order == 0:
        return gpr.copy().rename("model_gpr")
    if difference_order == 1:
        return gpr.diff().dropna().rename("model_gpr")
    raise ValueError("difference_order must be either 0 or 1.")

def build_calendar_fixed_effects(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Helper function to create day-of-week fixed effects. Reference date = Monday
    """
    weekday = pd.Series(index.dayofweek, index=index, name="weekday")
    return pd.get_dummies(weekday, prefix="weekday", drop_first=True, dtype=float)

def compute_stationarity_test(series: pd.Series) -> pd.DataFrame:
    """
    Run a basic ADF test on a given series
    """
    data = series.dropna().astype(float)
    if len(data) < 20:
        raise ValueError("The series is too short to run the ADF stationarity test.")
    statistic, p_value, used_lag, nobs, critical_values, icbest = adfuller(
        data, regression="c", autolag="AIC", result_object=False)
    return pd.DataFrame({"value": pd.Series({
            "test": "Augmented Dickey-Fuller",
            "series": series.name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "used_lag": int(used_lag),
            "nobs": int(nobs),
            "critical_value_1pct": float(critical_values["1%"]),
            "critical_value_5pct": float(critical_values["5%"]),
            "critical_value_10pct": float(critical_values["10%"]),
            "stationary_at_5pct": bool(p_value < 0.05),
            "autolag": "AIC",
            "icbest": float(icbest)}, dtype="object")})

def select_arma_order(gpr: pd.Series, max_ar_order: int = 5, max_ma_order: int = 2, selection_criterion: str = "bic", 
                      calendar_effects: bool = True, trend: str = "c", difference_order: int = 1) -> tuple[int, int]:
    """
    Select the autoregressive (p) and moving average (q) orders for the ARMA model based on the specified selection criterion (AIC or BIC).
    It works by estimating the models and computing AIC/BIC for each (p, q) combination, returning the  one that mninimizes the criterion.

    Inputs
    ------
        :gpr: GPR series to be modeled.
        :max_ar_order: maximum AR order to consider.
        :max_ma_order: maximum MA order to consider.
        :selection_criterion: criterion to use for model selection ("aic" or "bic").
        :calendar_effects: whether to include day-of-week fixed effects.
        :trend: trend specification for the SARIMAX model. "c" = constant, "t" = linear trend, "ct" = both, "n" = none
        :difference_order: order of differencing to apply to the GPR series

    Output
    ------
        Tuple containing the selected AR and MA orders (p, q) based on the specified selection criterion.
    """
    criterion = selection_criterion.lower()
    # Input validation
    if criterion not in {"aic", "bic"}:
        raise ValueError("selection_criterion must be either 'aic' or 'bic'.")
    if max_ar_order < 0:
        raise ValueError("max_ar_order must be non-negative.")
    if max_ma_order < 0:
        raise ValueError("max_ma_order must be non-negative.")

    # If asked, we differenciate the series, then add calendar days fixed effects
    transformed = transform_gpr(gpr, difference_order=difference_order)
    exog = build_calendar_fixed_effects(transformed.index) if calendar_effects else None
    rows = []
    # Loop over all combinations of AR and MA orders
    for p in range(max_ar_order + 1):
        for q in range(max_ma_order + 1):
            # If the model converges, report results, otherwise we report the failure but carry on
            try:
                model = SARIMAX(transformed, exog=exog, order=(p, 0, q), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
                fitted = model.fit(method="lbfgs", disp=False, maxiter=200)
                rows.append({"ar_order": p,
                    "difference_order": difference_order,
                    "ma_order": q,
                    "nobs": int(fitted.nobs),
                    "converged": bool(fitted.mle_retvals.get("converged", False)),
                    "log_likelihood": float(fitted.llf),
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                    "hqic": float(fitted.hqic),
                    "error": None})
            except Exception as error:
                rows.append({"ar_order": p,
                            "difference_order": difference_order,
                            "ma_order": q,
                            "nobs": np.nan,
                            "converged": False,
                            "log_likelihood": np.nan,
                            "aic": np.nan,
                            "bic": np.nan,
                            "hqic": np.nan,
                            "error": str(error)})

    model_selection = pd.DataFrame(rows)
    valid = model_selection.loc[model_selection[criterion].notna() & model_selection["converged"]].copy()
    if valid.empty:
        raise RuntimeError("No ARMA candidate converged successfully.")
    # We select the combination minimizing AIC or BIC
    selected = valid.loc[valid[criterion].idxmin()]
    orders = {"p": int(selected["ar_order"]),
              "q": int(selected["ma_order"])}
    model_selection["delta_aic"] = model_selection["aic"] - valid["aic"].min()
    model_selection["delta_bic"] = model_selection["bic"] - valid["bic"].min()
    model_selection["selected"] = ((model_selection["ar_order"] == orders["p"])
                                    & (model_selection["ma_order"] == orders["q"]))
    return orders, model_selection

def fit_gpr_arma(gpr: pd.Series, p: int, q: int, calendar_effects: bool = True, trend: str = "c", difference_order: int = 1) -> tuple[SARIMAX, pd.DataFrame]:
    """
    Fit an ARMA model to the given series and returns the fitted values and residuals, accounting for any initial burn-in period.

    Inputs
    ------
        :gpr: GPR series to be modeled.
        :p: AR order of the ARMA model.
        :q: MA order of the ARMA model.
        :calendar_effects: whether to include day-of-week fixed effects.
        :trend: trend specification for the SARIMAX model. "c" = constant, "t" = linear trend, "ct" = both, "n" = none
        :difference_order: order of differencing to apply to the GPR series

    Output
    ------
        Tuple containing the fitted ARMA model and a DataFrame with the original GPR series, the transformed series used for modeling, 
          fitted values from the ARMA model, and the resulting innovations (residuals).
    """
    # We prepare the regression: do we transform the series and include calendar effects
    transformed = transform_gpr(gpr, difference_order=difference_order)
    exog = build_calendar_fixed_effects(transformed.index) if calendar_effects else None
    # Estimation
    model = SARIMAX(transformed, exog=exog, order=(p, 0, q), trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    fitted = model.fit(method="lbfgs", disp=False, maxiter=200)
    burn = int(getattr(fitted, "loglikelihood_burn", 0))
    fitted_values = pd.Series(fitted.fittedvalues, index=transformed.index)
    residuals = pd.Series(fitted.resid, index=transformed.index)
    if burn > 0:
        fitted_values.iloc[:burn] = np.nan
        residuals.iloc[:burn] = np.nan
    # Clean save
    innovations = pd.DataFrame(index=gpr.index)
    innovations["gpr"] = gpr.astype(float)
    innovations["model_gpr"] = np.nan
    innovations["fitted_model_gpr"] = np.nan
    innovations["innovation"] = np.nan
    innovations.loc[transformed.index, "model_gpr"] = transformed
    innovations.loc[transformed.index, "fitted_model_gpr"] = fitted_values
    innovations.loc[transformed.index, "innovation"] = residuals
    return fitted, innovations

def build_regression_table(model) -> pd.DataFrame:
    """
    Create a compact ARMA parameter table for export
    Inputs
    ------
        :model: Fitted ARMA/SARIMAX model.
    Output
    ------
        DataFrame containing the coefficient, standard error, z-statistic, p-value, and 95% confidence interval for each term in the model.
    """

    ci = model.conf_int(alpha=0.05)
    regression = pd.DataFrame({"coefficient": model.params,
                                "std_error": model.bse,
                                "z_stat": model.zvalues,
                                "p_value": model.pvalues,
                                "ci_lower_95": ci.iloc[:, 0],
                                "ci_upper_95": ci.iloc[:, 1]})
    regression.index.name = "term"
    return regression

def compute_residual_autocorrelation(model, p: int, q: int, lags: Iterable[int] = (1, 5, 10, 20, 30)) -> pd.DataFrame:
    """"
    Perform the Ljung-Box test on the residuals of an ARMA model to check for autocorrelation.
    """
    burn = int(getattr(model, "loglikelihood_burn", 0))
    residuals = pd.Series(model.resid).iloc[burn:].dropna()
    model_df = p + q
    valid_lags = sorted({int(lag) for lag in lags if 0 < int(lag) < len(residuals)})
    if not valid_lags:
        raise ValueError("No valid Ljung-Box lag remains for the available residual sample.")
    unadjusted = acorr_ljungbox(residuals, lags=valid_lags, return_df=True)
    adjusted = acorr_ljungbox(residuals, lags=valid_lags, model_df=model_df, return_df=True)
    autocorrelation = pd.DataFrame({"lag": valid_lags,
                                    "statistic": unadjusted["lb_stat"].to_numpy(),
                                    "p_value_unadjusted": unadjusted["lb_pvalue"].to_numpy(),
                                    "p_value_adjusted": adjusted["lb_pvalue"].to_numpy()})
    autocorrelation.insert(0, "test", "Ljung-Box")
    autocorrelation["reject_unadjusted_5pct"] = autocorrelation["p_value_unadjusted"] < 0.05
    autocorrelation["reject_adjusted_5pct"] = autocorrelation["p_value_adjusted"] < 0.05
    autocorrelation["adjusted_pvalue_available"] = autocorrelation["lag"] > model_df
    return autocorrelation

def select_large_positive_innovations(innovations: pd.DataFrame, threshold: float = 0.99, threshold_sample: str = "positive",
                                      rolling_window: int = 756, rolling_min_periods: int = 504, selection_start_date: pd.Timestamp | None = None,
                                      selection_end_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Select q% largest positive GPR innovations (with q = 1-threshold) after adjusting them for time-varying volatility.
    The rolling volatility is estimated using past observations only. A three-year window corresponds approximately to 3 * 252 = 756 trading observations.
    We restrict the q% largest positive innovations to the selected sample (ex: ILS one)
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1.")
    if threshold_sample not in {"positive", "all"}:
        raise ValueError("threshold_sample must be either 'positive' or 'all'.")
    if rolling_window <= 1:
        raise ValueError("rolling_window must be greater than 1.")
    if not 1 < rolling_min_periods <= rolling_window:
        raise ValueError("rolling_min_periods must lie between 2 and rolling_window.")
    data = innovations.copy()

    # We compute the rolling volatility of innovation
    data["innovation_rolling_vol"] = data["innovation"].shift(1).rolling(
                                    window=rolling_window,
                                    min_periods=rolling_min_periods).std(ddof=1)
    # Innovation expressed in rolling standard deviations
    data["innovation_rolling_std"] = data["innovation"] / data["innovation_rolling_vol"]
    # We restrict to correct sample
    selection_mask = pd.Series(True, index=data.index)
    if selection_start_date is not None:
        selection_mask &= (data.index >= pd.Timestamp(selection_start_date))
    if selection_end_date is not None:
        selection_mask &= (data.index <= pd.Timestamp(selection_end_date))
    # Of course we need to ensure that we have enough positive errors (= deterioreting geopolitical risk)
    # Then we select the valid innovations that will be used to determine the threshold (either only positive or all)
    valid_innovations = data.loc[selection_mask, "innovation_rolling_std"].replace([np.inf, -np.inf], np.nan).dropna()
    positive_innovations = valid_innovations.loc[valid_innovations > 0]
    if positive_innovations.empty:
        raise ValueError("No positive rolling-vol-adjusted GPR innovations are available to construct shocks.")
    threshold_source = positive_innovations if threshold_sample == "positive" else valid_innovations
    threshold_value = float(threshold_source.quantile(threshold))
    data["positive_percentile_rank"] = np.nan
    data.loc[positive_innovations.index,"positive_percentile_rank"] = positive_innovations.rank(method="average", pct=True)
    data["is_shock"] = (selection_mask & (data["innovation_rolling_std"] > 0) & (
                                          data["innovation_rolling_std"] >= threshold_value))
    
    data["gpr_shock"] = (data["innovation"].where(data["is_shock"], 0.0))
    shocks = data.loc[data["is_shock"], ["gpr", "model_gpr", "fitted_model_gpr", "innovation",
                                        "innovation_rolling_vol", "innovation_rolling_std", 
                                        "positive_percentile_rank", "gpr_shock"]].copy()
    info = {"threshold_quantile": threshold,
            "threshold_sample": threshold_sample,
            "threshold_variable": "innovation_rolling_std",
            "threshold_value": threshold_value,
            "rolling_window_observations": rolling_window,
            "rolling_min_periods": rolling_min_periods,
            "n_valid_innovations": int(valid_innovations.shape[0]),
            "n_positive_innovations": int(positive_innovations.shape[0]),
            "n_selected_shocks": int(data["is_shock"].sum()),
            "selected_share_all_innovations": float(data["is_shock"].sum() / valid_innovations.shape[0]),
            "selected_share_positive_innovations": float(data["is_shock"].sum() / positive_innovations.shape[0])}
    return data, shocks, info

def classify_gpr_shocks_with_brent(shocks: pd.DataFrame, brent: pd.Series):
    """
    We classify GPR shocks based on the sign of the Brent return around the event.
    Similar to the "poor's man" method of Pinchetti (2026) where we compute the 3 day log change of Brent around the event.
    Following the three-day window used by Pinchetti: Brent surprise_t = 100 * log(P_{t+1} / P_{t-2})
    A positive Brent log change implies is_energy=True.

    Inputs
    ------
        :shocks: DataFrame containing GPR shocks.
        :brent: Brent prices indexed by date.

    Output
    -------
        Tuple with the classified DataFrame and the classification info dictionary.
    """
    # Imports and cleaning
    classified = shocks.copy()
    classified["brent_log_change"] = np.nan
    classified["is_energy"] = pd.Series(pd.NA, index=classified.index, dtype="boolean")
    brent = brent.dropna().sort_index()
    brent_dates = brent.index

    for event_date in classified.index:
        event_date = pd.Timestamp(event_date)
        # First Brent trading date on or after the event date
        position = brent_dates.searchsorted(event_date)
        if position >= len(brent_dates):
            continue
        # If the event itself is a Brent trading date, this is t.
        # Otherwise, the first available Brent trading date after the event becomes the trading-day anchor.
        anchor_position = position
        before_position = anchor_position - 2
        after_position = anchor_position + 1
        if (before_position < 0 or after_position >= len(brent_dates)):
            continue

        price_before = float(brent.iloc[before_position])
        price_after = float(brent.iloc[after_position])
        log_change = 100.0 * (np.log(price_after) - np.log(price_before))
        classified.loc[event_date, "brent_log_change"] = log_change
        classified.loc[event_date, "is_energy"] = bool(log_change > 0)

    valid = classified["brent_log_change"].notna()
    energy = classified["is_energy"].fillna(False) & valid
    non_energy = (~classified["is_energy"].fillna(False)) & valid
    classification_info = {"brent_classification_window": "three trading returns: t-2 close to t+1 close",
                            "n_brent_classified_shocks": int(valid.sum()),
                            "n_brent_unclassified_shocks": int((~valid).sum()),
                            "n_energy_shocks": int(energy.sum()),
                            "n_non_energy_shocks": int(non_energy.sum()),
                            "share_energy_among_classified": float(energy.sum() / valid.sum()) if valid.any() else np.nan}
    return classified, classification_info

def build_summary_table(model, orders: dict, difference_order: int, selection_criterion: str, calendar_effects: bool, 
                        stationarity: pd.DataFrame, threshold_info: dict, classification_info: dict):
    """
    Provides a summary table of the model's characteristics, diagnostics,...    
    """
    burn = int(getattr(model, "loglikelihood_burn", 0))
    residuals = pd.Series(model.resid).iloc[burn:].dropna()
    model_description = "ARMA on GPR levels" if difference_order == 0 else "ARMA on first-differenced GPR"
    values = {"model": model_description,
                "selection_criterion": selection_criterion,
                "ar_order_p": orders["p"],
                "difference_order_d": difference_order,
                "ma_order_q": orders["q"],
                "calendar_effects": calendar_effects,
                "nobs": int(model.nobs),
                "loglikelihood_burn": burn,
                "converged": bool(model.mle_retvals.get("converged", False)),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "hqic": float(model.hqic),
                "residual_std": float(residuals.std(ddof=1)),
                "durbin_watson": float(durbin_watson(residuals)),
                "adf_series": stationarity.loc["series", "value"],
                "adf_statistic": float(stationarity.loc["statistic", "value"]),
                "adf_p_value": float(stationarity.loc["p_value", "value"]),
                "adf_stationary_at_5pct": bool(stationarity.loc["stationary_at_5pct", "value"]),
                **threshold_info,
                **classification_info}
    return pd.DataFrame({"value": pd.Series(values, dtype="object")})

def compute_gpr_shocks_orchestrator(gpr_config: dict, trading_dates: pd.DatetimeIndex | None = None) -> GPRShockResult:
    """
    Compute GPR shocks, classify them and save the results. 
    NOTE: we estimate the ARMA on 2002-2026 (so not restricted to the ILS trading dates) 
    because we need the 3y rolling window for volatility estimation
    HOWEVER the first 3y innovations are NOT used in our shocks identification.

    Inputs:
    -----------
        :gpr_config: Dictionary containing GPR configuration parameters.
        :trading_dates: Optional DatetimeIndex of ILS trading dates to restrict the analysis to.

    Output:
    ----------
        GPRShockResult containing estimated GPR shocks, classification, and summary statistics.
    """
    # We first load cleaned GPR and Brent data
    if trading_dates is not None:
        ils_dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize().sort_values().unique()
        ils_start_date = pd.Timestamp(ils_dates.min())
        ils_end_date = pd.Timestamp(ils_dates.max())
        pre_sample_years = int(gpr_config.get("pre_sample_years", 3))
        estimation_start_date = ils_start_date - pd.DateOffset(years=pre_sample_years)
    else:
        ils_dates = None
        ils_start_date = None
        ils_end_date = None
        estimation_start_date = pd.Timestamp(gpr_config["start_date"]) if gpr_config.get("start_date") is not None else None

    gpr = load_gpr_data(file=gpr_config["input"], sheet=gpr_config["sheet"], date_column=gpr_config["date_column"],
                        gpr_column=gpr_config["gpr_column"], start_date=estimation_start_date, end_date=gpr_config.get("end_date"))
    brent = load_brent_data(file=gpr_config["brent_input"], sheet=gpr_config["brent_sheet"], date_column=gpr_config["brent_date_column"],
                            price_column=gpr_config["brent_price_column"])
    
    # We now need to build the GPR estimation trading calendar
    # Before the daily ILS sample (2005), we use the Brent trading dates as a proxy for the financial-market calendar
    # Then we use the ILS trading dates
    if ils_dates is not None:
        brent_dates = pd.DatetimeIndex(brent.index).normalize()
        pre_sample_trading_dates = brent_dates[(brent_dates >= estimation_start_date) & (brent_dates < ils_start_date)]
        estimation_trading_dates = pre_sample_trading_dates.union(ils_dates).sort_values()
        gpr = restrict_to_trading_dates(gpr, estimation_trading_dates)
    if gpr.empty:
        raise ValueError("No GPR observations remain in the estimation sample.")
    if brent.empty:
        raise ValueError("No valid Brent observations are available.")
    
    # If asked we transform the GPR data
    model_gpr = transform_gpr(gpr, difference_order=gpr_config["difference_order"])
    stationarity = compute_stationarity_test(model_gpr)
    # Best ARMA model selection
    orders, model_selection = select_arma_order(gpr=gpr, max_ar_order=gpr_config["max_ar_order"], max_ma_order=gpr_config["max_ma_order"],
                                                selection_criterion=gpr_config["selection_criterion"], calendar_effects=gpr_config["calendar_effects"],
                                                trend=gpr_config["trend"], difference_order=gpr_config["difference_order"])
    # ARMA estimation
    model, innovations = fit_gpr_arma(gpr=gpr, p=orders["p"], q=orders["q"], calendar_effects=gpr_config["calendar_effects"], 
                                      trend=gpr_config["trend"], difference_order=gpr_config["difference_order"])
    # We extract our "shocks". Note that the first 3 years of innovations are not used in the shocks identification.
    innovations, shocks, threshold_info = select_large_positive_innovations(innovations=innovations, threshold=gpr_config["threshold"],
                                                                            threshold_sample=gpr_config["threshold_sample"], 
                                                                            rolling_window=gpr_config.get("rolling_vol_window", 756),
                                                                            rolling_min_periods=gpr_config.get("rolling_vol_min_periods", 504),
                                                                            selection_start_date=ils_start_date, selection_end_date=ils_end_date)
    # Energy classification based on Brent data
    shocks, classification_info = classify_gpr_shocks_with_brent(shocks=shocks, brent=brent)

    # Diagnostics
    regression = build_regression_table(model=model)
    autocorrelation = compute_residual_autocorrelation(model=model, p=orders["p"], q=orders["q"], lags=gpr_config["autocorrelation_lags"])
    summary = build_summary_table(model=model, orders=orders, difference_order=gpr_config["difference_order"], 
                                  selection_criterion=gpr_config["selection_criterion"].lower(), 
                                  calendar_effects=gpr_config["calendar_effects"], stationarity=stationarity,
                                  threshold_info=threshold_info, classification_info=classification_info)
    
    return GPRShockResult(
        innovations=innovations,
        shocks=shocks,
        regression=regression,
        model_selection=model_selection,
        stationarity=stationarity,
        autocorrelation=autocorrelation,
        summary=summary,
    )

def export_gpr_shocks(result: GPRShockResult, output: str):

    """
    Export GPR shocks, Brent classification and model diagnostics in Excel
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output) as writer:
        result.innovations.to_excel(writer, sheet_name="innovations", index=True, index_label="date")
        result.shocks.to_excel(writer, sheet_name="shocks", index=True, index_label="date")
        result.regression.to_excel(writer, sheet_name="regression", index=True)
        result.model_selection.to_excel(writer, sheet_name="model_selection", index=False)
        result.stationarity.to_excel(writer, sheet_name="stationarity", index=True, index_label="statistic")
        result.autocorrelation.to_excel(writer, sheet_name="autocorrelation", index=False)
        result.summary.to_excel(writer, sheet_name="summary", index=True, index_label="statistic")
