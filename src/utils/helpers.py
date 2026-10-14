
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from dateutil.easter import easter
from utils.paths import project_path

def save_csv(frame: pd.DataFrame | None, path: str | Path) -> None:
    """
    Save a DataFrame to CSV.
    """
    if frame is None:
        return
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def flatten_matrix(prefix: str, matrix: np.ndarray) -> dict:
    """
    Flatten a matrix into named scalar coefficients.
    """
    matrix = np.asarray(matrix)

    return {
        f"{prefix}_{i + 1}{j + 1}": float(matrix[i, j])
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
    }

def flatten_vector(prefix: str, vector: np.ndarray) -> dict:
    """
    Flatten a vector into named scalar coefficients.
    """
    vector = np.asarray(vector).reshape(-1)

    return {
        f"{prefix}_{i + 1}": float(vector[i])
        for i in range(len(vector))
    }

def build_target_holidays(start_year: int, end_year: int):
    """
    Build TARGET closing days for each year.
    These are 1st January, Good Friday, Easter Monday, 1st May, 25th December, and 26th December.
    """
    holidays = []
    for year in range(start_year, end_year + 1):
        easter_sunday = easter(year)
        holidays.extend([pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(easter_sunday - timedelta(days=2)),
                        pd.Timestamp(easter_sunday + timedelta(days=1)), pd.Timestamp(year=year, month=5, day=1), 
                        pd.Timestamp(year=year, month=12, day=25), pd.Timestamp(year=year, month=12, day=26)])
    return pd.DatetimeIndex(holidays)