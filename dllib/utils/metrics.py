import numpy as np

from .device import xp


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return xp.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_mean = xp.mean(y_true)
    ss_res = xp.sum((y_true - y_pred) ** 2)
    ss_tot = xp.sum((y_true - y_true_mean) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


__all__ = ["mse", "rmse", "r2_score"]

