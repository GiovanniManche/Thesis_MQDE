import numpy as np
from .model import ATSMModel

def model_diagnostics(model: ATSMModel) -> dict:
    """
    Function to put in a dictionary core diagnostics and long run elements.
    Used to then put them in a JSON.
    """
    N = model.K1P_P.shape[0]
    eig_P = np.linalg.eigvals(np.eye(N) + model.K1P_P)
    eig_Q = np.linalg.eigvals(np.eye(N) + model.K1Q_P)
    long_run_factors = np.linalg.solve(-model.K1P_P, model.K0P_P)
    long_run_inflation = float(model.rho0_P + model.rho1_P @ long_run_factors)

    return {
        "optimizer_success": bool(model.optimizer_success),
        "optimizer_message": model.optimizer_message,
        "objective_value": float(model.objective_value),
        "sigma_e": float(model.sigma_e),
        "eigenvalues_P_real": eig_P.real.tolist(),
        "eigenvalues_P_imag": eig_P.imag.tolist(),
        "eigenvalues_Q_real": eig_Q.real.tolist(),
        "eigenvalues_Q_imag": eig_Q.imag.tolist(),
        "max_abs_eigenvalue_P": float(np.max(np.abs(eig_P))),
        "max_abs_eigenvalue_Q": float(np.max(np.abs(eig_Q))),
        "long_run_factors": long_run_factors.tolist(),
        "long_run_inflation_decimal": long_run_inflation,
        "long_run_inflation_percent": 100.0 * long_run_inflation,
        "mean_negative_log_likelihood": float(np.mean(model.llk_series)),
    }
