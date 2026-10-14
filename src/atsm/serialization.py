import json
from pathlib import Path
import numpy as np
from .model import ATSMModel


ARRAY_FIELDS = {
    "W",
    "maturities",
    "K0P_P",
    "K1P_P",
    "K0Q_P",
    "K1Q_P",
    "Sigma_P",
    "rho1_P",
    "AP",
    "BP",
    "AX",
    "BX",
    "K0Q_X",
    "K1Q_X",
    "rho1_X",
    "llk_series",
    "llkP",
    "llkQ",
    "P_monthly",
    "optimizer_parameters",
}


def _jsonify(value):
    """
    Convert NumPy/Python objects to JSON-compatible objects.
    """
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, complex):
        return {
            "__complex__": True,
            "real": float(value.real),
            "imag": float(value.imag),
        }
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value

def model_to_dict(model: ATSMModel) -> dict:
    """
    Changes the ATSMModel dataclass object into a dictionary then used to have a JSON
    """
    return {
        "format_version": 1,
        "W": _jsonify(model.W),
        "maturities": _jsonify(model.maturities),
        "dt": float(model.dt),
        "K0P_P": _jsonify(model.K0P_P),
        "K1P_P": _jsonify(model.K1P_P),
        "K0Q_P": _jsonify(model.K0Q_P),
        "K1Q_P": _jsonify(model.K1Q_P),
        "Sigma_P": _jsonify(model.Sigma_P),
        "rho0_P": float(model.rho0_P),
        "rho1_P": _jsonify(model.rho1_P),
        "AP": _jsonify(model.AP),
        "BP": _jsonify(model.BP),
        "AX": _jsonify(model.AX),
        "BX": _jsonify(model.BX),
        "kinfQ": float(model.kinfQ),
        "K0Q_X": _jsonify(model.K0Q_X),
        "K1Q_X": _jsonify(model.K1Q_X),
        "rho0_X": float(model.rho0_X),
        "rho1_X": _jsonify(model.rho1_X),
        "sigma_e": float(model.sigma_e),
        "llk_series": _jsonify(model.llk_series),
        "llkP": _jsonify(model.llkP),
        "llkQ": _jsonify(model.llkQ),
        "P_monthly": _jsonify(model.P_monthly),
        "optimizer_success": bool(model.optimizer_success),
        "optimizer_message": str(model.optimizer_message),
        "objective_value": float(model.objective_value),
        "optimizer_parameters": _jsonify(
            model.optimizer_parameters
        ),
        "metadata": _jsonify(model.metadata),
    }


def save_model_json(model: ATSMModel, path: str | Path) -> None:
    """
    Converts the dictionary that contains the model estimation information into a JSON file,
    and saves it.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = model_to_dict(model)

    with path.open("w",encoding="utf-8") as f:
        json.dump(payload, f,indent=2)


def load_model_json(path: str | Path) -> ATSMModel:
    """
    Converts the JSON file containing the estimation information into a ATSMModel class
    to then be used in the rest of the code
    Cf ATSMModel dataclass code for more details
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    def arr(name):
        return np.asarray(data[name], dtype=float)

    return ATSMModel(W=arr("W"),
        maturities=arr("maturities"),
        dt=float(data["dt"]),
        K0P_P=arr("K0P_P"),
        K1P_P=arr("K1P_P"),
        K0Q_P=arr("K0Q_P"),
        K1Q_P=arr("K1Q_P"),
        Sigma_P=arr("Sigma_P"),
        rho0_P=float(data["rho0_P"]),
        rho1_P=arr("rho1_P"),
        AP=arr("AP"),
        BP=arr("BP"),
        AX=arr("AX"),
        BX=arr("BX"),
        kinfQ=float(data["kinfQ"]),
        K0Q_X=arr("K0Q_X"),
        K1Q_X=arr("K1Q_X"),
        rho0_X=float(data["rho0_X"]),
        rho1_X=arr("rho1_X"),
        sigma_e=float(data["sigma_e"]),
        llk_series=arr("llk_series"),
        llkP=arr("llkP"),
        llkQ=arr("llkQ"),
        P_monthly=arr("P_monthly"),
        optimizer_success=bool(data["optimizer_success"]),
        optimizer_message=str(data["optimizer_message"]),
        objective_value=float(data["objective_value"]),
        optimizer_parameters=arr("optimizer_parameters"),
        metadata=data.get("metadata", {},)
    )