import pandas as pd
from atsm.decomposition import decompose_ils, forward_decomposition_frame, spot_decomposition_frame

def decompose_model(model, yields, dates, maturities, dec_cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute spot and forward ILS decompositions for one estimated ATSM.
    Inputs:
    -----------
        :model: ATSMModel Dataclass containing the estimated parameters of the ATSM.
        :yields: ILS rates
        :dates: observed dates
        :dec_cfg: decomposition booleans in the configuration yaml (compute_forwards, compute_5y5y)    
                
    Output:
    ----------
        Dataframes with the observed, fitted, expected inflation and IRP at each date for each maturity.
        Spots and forwards are two distinct dataframes.
    """
    decomposition = decompose_ils(model=model, yields=yields, compute_forwards=bool(dec_cfg.get("compute_forwards", True)),
                                  compute_5y5y=bool(dec_cfg.get("compute_5y5y", True)))

    spot = spot_decomposition_frame(dates=dates,maturities=maturities,decomposition=decomposition)
    forward = forward_decomposition_frame(dates=dates,decomposition=decomposition)
    return spot, forward


def average_decomposition_frames(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Average the standard and bias-adjusted ATSM decompositions. Identification columns, such as date and maturity / forward label, are preserved unchanged.
    Inputs:
    -----------
        :left: Either standard ATSM decomposition or bias-adjusted ATSM decomposition.
        :right: Other decomposition than left.
        :keys: on what to merge the two datasets (fitted, expected...)

    Output:
    ----------
        Dataframe averaging the two decompositions.
    """
    merged = left.merge(right, on=keys, how="inner", suffixes=("_standard", "_bias"), validate="one_to_one")
    out = merged[keys].copy()
    numerical_columns = ["observed", "fitted", "expected", "irp", "pricing_error"]

    for column in numerical_columns:
        standard_col = f"{column}_standard"
        bias_col = f"{column}_bias"
        if (standard_col in merged.columns and bias_col in merged.columns):
            out[column] = 0.5 * (merged[standard_col] + merged[bias_col])
    return out
