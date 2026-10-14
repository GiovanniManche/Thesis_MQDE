################################################
# raw_ils.py
# Loading and validation of ILS panels.
# The loader:
#   1. resolves the input path relative to the project root;
#   2. reads CSV or Excel input files;
#   3. identifies and orders ILS maturities;
#   4. converts dates to a DatetimeIndex;
#   5. applies the requested sample end date;
#   6. checks that the panel is rectangular and contains no missing rates;
#   7. converts ILS rates to the scale used by the ATSM.
# Meanwhile, load_ils_trading_dates is responsible for loading the ILS trading calendar and removing TARGET closing days.
################################################

from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from utils.paths import project_path
from utils.helpers import build_target_holidays



def _read_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """
    Read an input data file into a pandas DataFrame. Supported formats are CSV and Excel.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    raise ValueError(f"Unsupported ILS file format: {suffix}. Supported formats are .csv, .xlsx and .xls.")

# ILS PANEL LOADER
def load_ils_panel(path: str | Path, date_column: str, maturity_columns: Mapping[int | str, str],
                   scale: float = 1.0, sheet_name: str | None = None, end_date: str | pd.Timestamp | None = None,
                    ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and validate a rectangular panel of ILS rates.

    Inputs
    ------------
        :path: Path to the input CSV or Excel file
        :date_column: Name of the column containing observation dates.
        :maturity_columns: Mapping between maturities in years and column names in the input file. For example:
        :scale: Multiplicative factor applied to ILS rates. scale = 0.01 if 2% is stored as 2.0, scale = 1.0 if 2% is stored as 0.02.
        :sheet_name: Excel sheet to read. Ignored for CSV files. If None, the first Excel sheet is used.
        :end_date: Optional final date of the estimation sample.

    Output
    ----------
    Tuple containing:
        :dates: DatetimeIndex of observation dates.
        :yields: Array of ILS rates in decimal annualized units.
        :maturities: Vector containing maturities in years, ordered increasingly.
        :frame: Clean DataFrame containing dates and ordered maturity columns.
    """
    path = project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"ILS input file not found: {path}")
    frame = _read_table(path, sheet_name=sheet_name).copy()

    if date_column not in frame.columns:
        raise KeyError(f"Date column {date_column!r} not found in {path}.")

    normalized = sorted(((int(maturity), column)for maturity, column in maturity_columns.items()),key=lambda item: item[0])
    maturities = np.array([maturity for maturity, _ in normalized],dtype=int)
    columns = [column for _, column in normalized]
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing maturity columns in {path}: {missing_columns}")

    frame = frame[[date_column, *columns]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    # Remove observations without a valid date.
    frame = frame.dropna(subset=[date_column])
    # Remove rows where every ILS maturity is missing.
    frame = frame.dropna(how="all",subset=columns)
    # Sort chronologically
    frame = frame.sort_values(date_column)

    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        frame = frame.loc[frame[date_column] <= end_date]
    if frame.empty:
        raise ValueError("No ILS observations remain after applying the sample restrictions.")
    
    duplicate_dates = frame.loc[frame[date_column].duplicated(keep=False),date_column,]
    if not duplicate_dates.empty:
        raise ValueError("Duplicate dates found in the ILS panel, including: "+ ", ".join(duplicate_dates.astype(str).head(5).tolist()))

    # The ATSM requires every maturity to be available at every
    # estimation date. Partially missing rows are therefore rejected.
  
    missing_rows = frame[columns].isna().any(axis=1)
    if missing_rows.any():
        rows = frame.loc[missing_rows,date_column]
        raise ValueError("The ILS panel contains missing maturities on dates including: " + ", ".join(rows.astype(str).head(5).tolist()))

    values = frame[columns].to_numpy(dtype=float)
    values *= float(scale)
    if not np.isfinite(values).all():
        raise ValueError("The ILS panel contains non-finite values.")
    dates = pd.DatetimeIndex(frame[date_column])

    return (dates,values,maturities,frame.reset_index(drop=True))

def load_ils_trading_dates(file: str | Path, date_column: str = "date"):
    """
    Helper function that loads the ILS trading dates from a CSV file and removes TARGET closing days.
    """
    data = pd.read_csv(file, usecols=[date_column])
    dates = pd.to_datetime(data[date_column], errors="coerce").dropna()
    dates = pd.DatetimeIndex(dates).normalize().unique().sort_values()
    target_holidays = build_target_holidays(start_year=dates.min().year, end_year=dates.max().year)
    return dates[~dates.isin(target_holidays)]