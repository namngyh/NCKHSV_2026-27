"""Tieu chi danh gia nghien cuu mo phong (Muc 6.4): Bias, RMSE, Coverage, Brier.

eq:bias, eq:parameter-rmse, eq:coverage, eq:brier-state.
"""
from __future__ import annotations

import numpy as np


def bias(theta_hat: np.ndarray, theta0: float) -> float:
    return float(np.mean(np.asarray(theta_hat) - theta0))


def parameter_rmse(theta_hat: np.ndarray, theta0: float) -> float:
    return float(np.sqrt(np.mean((np.asarray(theta_hat) - theta0) ** 2)))


def coverage(lower: np.ndarray, upper: np.ndarray, theta0: float) -> float:
    lower, upper = np.asarray(lower), np.asarray(upper)
    return float(np.mean((lower <= theta0) & (theta0 <= upper)))


def brier_state(xi_hat: np.ndarray, true_state: np.ndarray) -> float:
    """xi_hat: (T, 2) xac suat uoc luong (loc hoac lam tron); true_state: (T,) trong {0,1}."""
    xi_hat = np.asarray(xi_hat)
    true_state = np.asarray(true_state)
    T, n_states = xi_hat.shape
    indicator = np.eye(n_states)[true_state]
    return float(np.mean(np.sum((xi_hat - indicator) ** 2, axis=1)))


def state_classification_accuracy(xi_hat: np.ndarray, true_state: np.ndarray) -> dict:
    """Do chinh xac phan lop sau khi anh xa nhan (argmax xac suat), balanced accuracy."""
    pred_state = np.argmax(np.asarray(xi_hat), axis=1)
    true_state = np.asarray(true_state)
    accuracy = float(np.mean(pred_state == true_state))
    balanced = []
    for s in np.unique(true_state):
        mask = true_state == s
        balanced.append(np.mean(pred_state[mask] == s))
    return {"accuracy": accuracy, "balanced_accuracy": float(np.mean(balanced))}


def transition_matrix_error(P_hat: np.ndarray, P0: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(P_hat) - np.asarray(P0)) ** 2)))
