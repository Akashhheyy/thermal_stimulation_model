"""Regression evaluation, baselines, sanity checks, and feature importance."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import ML_TARGETS

__all__ = [
    "regression_metrics",
    "baseline_metrics",
    "physical_sanity_report",
    "feature_importance",
    "evaluate_target_model",
]

# Targets that can legitimately be exactly zero must never be scored with
# percentage-based metrics.
ZERO_POSSIBLE_TARGETS = {
    "percent_time_comfortable",
    "degree_hours_below_comfort",
    "degree_hours_above_comfort",
    "thermal_mass_net_kwh",
}

# Documented physical bounds used for sanity REPORTING only.  Predictions are
# never clipped to improve metrics.
TARGET_BOUNDS = {
    "percent_time_comfortable": (0.0, 100.0),
    "percent_time_below_comfort": (0.0, 100.0),
    "percent_time_above_comfort": (0.0, 100.0),
    "total_heat_loss_kwh": (0.0, None),
    "total_solar_gain_kwh": (0.0, None),
    "thermal_mass_absorbed_kwh": (0.0, None),
    "thermal_mass_released_kwh": (0.0, None),
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R2, explained variance; MAPE only when every value > 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("metric inputs must be same-shape and non-empty")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("metric inputs must be finite")
    error = y_pred - y_true
    metrics = {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    metrics["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    variance = float(np.var(y_true))
    metrics["explained_variance"] = (
        float(1.0 - np.var(error) / variance) if variance > 0.0 else float("nan")
    )
    if bool((y_true > 0.0).all()):
        metrics["mape_percent"] = float(100.0 * np.mean(np.abs(error / y_true)))
    else:
        metrics["mape_percent"] = None  # type: ignore[assignment]
    return metrics


def baseline_metrics(
    y_train: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Training-set mean predictor: the honest reference every model must beat."""
    mean = float(np.mean(np.asarray(y_train, dtype=float)))
    return regression_metrics(y_test, np.full(len(y_test), mean))


def physical_sanity_report(target: str, y_pred: np.ndarray) -> dict[str, Any]:
    """Count physically questionable RAW predictions; never clip silently."""
    y_pred = np.asarray(y_pred, dtype=float)
    report: dict[str, Any] = {
        "target": target,
        "n": int(y_pred.size),
        "non_finite_count": int((~np.isfinite(y_pred)).sum()),
        "bounds": None,
        "out_of_bounds_count": 0,
        "clipping_applied": False,
    }
    bounds = TARGET_BOUNDS.get(target)
    if bounds is not None:
        low, high = bounds
        violations = np.zeros(y_pred.shape, dtype=bool)
        if low is not None:
            violations |= y_pred < low
        if high is not None:
            violations |= y_pred > high
        report["bounds"] = {"low": low, "high": high}
        report["out_of_bounds_count"] = int(violations.sum())
    return report


def _clean_feature_name(raw: str) -> str:
    name = raw
    for prefix in ("categorical__", "numeric__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def feature_importance(pipeline, model_name: str) -> list[dict[str, Any]]:
    """Model-derived importance for one fitted pipeline.

    Tree models expose ``feature_importances_``; the linear model exposes the
    absolute regression coefficients.  These are model-derived importances,
    NOT causal effects.
    """
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessing"]
    names = [
        _clean_feature_name(raw) for raw in preprocessor.get_feature_names_out()
    ]
    if hasattr(regressor, "feature_importances_"):
        values = np.asarray(regressor.feature_importances_, dtype=float)
        kind = "tree_feature_importance"
    elif hasattr(regressor, "coef_"):
        values = np.abs(np.asarray(regressor.coef_, dtype=float).ravel())
        kind = "linear_coefficient_absolute"
    else:
        raise ValueError(f"model {model_name!r} exposes no importance attribute")
    if len(names) != len(values):
        raise ValueError("importance length does not match feature count")
    rows = [
        {"feature": name, "importance": float(value), "importance_kind": kind}
        for name, value in zip(names, values)
    ]
    rows.sort(key=lambda row: (-row["importance"], row["feature"]))
    return rows


def evaluate_target_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray
) -> dict[str, Any]:
    """Metrics plus baseline comparison and the sanity report for one pair."""
    metrics = regression_metrics(y_true, y_pred)
    baseline = baseline_metrics(y_train, y_true)
    return {
        "metrics": metrics,
        "baseline_train_mean": baseline,
        "beats_baseline_mae": bool(metrics["mae"] < baseline["mae"]),
        "sanity": physical_sanity_report("target", y_pred),
    }
