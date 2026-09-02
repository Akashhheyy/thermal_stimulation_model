"""Supported surrogate regression models with reproducible configuration.

Three model families are provided:

- ``linear_regression``: one-hot categoricals plus standardised numerics.
- ``random_forest``: sklearn RandomForestRegressor with a fixed random_state.
- ``gradient_boosting``: sklearn GradientBoostingRegressor with a fixed
  random_state.

XGBoost is supported only when it is ALREADY installed; this module detects it
and never installs anything.  Every model wraps its own preprocessing
(ColumnTransformer) so the fitted artefact is self-contained and prediction
reuses exactly the training-time preprocessing and feature order.
"""
from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

__all__ = ["MODEL_NAMES", "XGBOOST_AVAILABLE", "build_models", "model_library"]

MODEL_NAMES = ("linear_regression", "random_forest", "gradient_boosting")


def _xgboost_available() -> bool:
    try:  # pragma: no cover - depends on the environment
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


XGBOOST_AVAILABLE = _xgboost_available()

_DEFAULT_RANDOM_FOREST = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1}
_DEFAULT_GRADIENT_BOOSTING = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
}
_FAST_RANDOM_FOREST = {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 2}
_FAST_GRADIENT_BOOSTING = {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 2}


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_step = (
        StandardScaler() if scale_numeric else "passthrough"
    )
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", numeric_step, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )


def build_models(seed: int = 42, fast: bool = False) -> dict[str, Pipeline]:
    """Build the supported model pipelines for one seed.

    ``fast=True`` trades accuracy for speed (used by the offline test suite);
    the command-line training run uses the full configuration.
    """
    forest_args = _FAST_RANDOM_FOREST if fast else _DEFAULT_RANDOM_FOREST
    boosting_args = _FAST_GRADIENT_BOOSTING if fast else _DEFAULT_GRADIENT_BOOSTING
    models: dict[str, Pipeline] = {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessing", _preprocessor(scale_numeric=True)),
                ("regressor", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessing", _preprocessor(scale_numeric=False)),
                (
                    "regressor",
                    RandomForestRegressor(
                        random_state=seed,
                        n_jobs=1,
                        **forest_args,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessing", _preprocessor(scale_numeric=False)),
                (
                    "regressor",
                    GradientBoostingRegressor(random_state=seed, **boosting_args),
                ),
            ]
        ),
    }
    if XGBOOST_AVAILABLE:  # pragma: no cover - depends on the environment
        import xgboost

        models["xgboost"] = Pipeline(
            steps=[
                ("preprocessing", _preprocessor(scale_numeric=False)),
                (
                    "regressor",
                    xgboost.XGBRegressor(
                        random_state=seed,
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    return models


def model_library(seed: int = 42, fast: bool = False) -> dict[str, Any]:
    """Model configuration summary for the training metadata report."""
    models = build_models(seed=seed, fast=fast)
    summary: dict[str, Any] = {}
    for name, pipeline in models.items():
        regressor = pipeline.named_steps["regressor"]
        params = regressor.get_params(deep=False)
        summary[name] = {
            "class": f"{regressor.__class__.__module__}.{regressor.__class__.__name__}",
            "parameters": {
                key: value
                for key, value in params.items()
                if isinstance(value, (int, float, str, bool, type(None)))
            },
        }
    return summary
