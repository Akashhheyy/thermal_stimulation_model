"""Load trained surrogate models and predict from new design/weather inputs.

The loaded artefacts are the SAME fitted pipelines produced during training
(each contains its preprocessing), so prediction always applies the exact
training-time feature ordering and encodings without touching the thermal
engine.  This is a surrogate/predictive layer on top of the existing
simulation dataset.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .features import ML_TARGETS, build_feature_row, prepare_features
from .train import MODEL_FILE_PATTERN

__all__ = [
    "load_trained_models",
    "predict_features",
    "predict_design",
    "load_dataset_row_features",
]

VALID_MODEL_NAMES = ("linear_regression", "random_forest", "gradient_boosting", "xgboost")


def load_trained_models(models_dir: Path | str) -> dict[tuple[str, str], object]:
    """Load every ``{target}__{model}.joblib`` artefact from ``models_dir``.

    Returns a mapping keyed by ``(target, model_name)``.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"trained models directory not found: {models_dir}")
    models: dict[tuple[str, str], object] = {}
    for path in sorted(models_dir.glob("*.joblib")):
        stem = path.stem
        if "__" not in stem:
            continue
        target, model = stem.split("__", 1)
        if target not in ML_TARGETS:
            continue
        if model not in VALID_MODEL_NAMES:
            continue
        models[(target, model)] = joblib.load(path)
    if not models:
        raise FileNotFoundError(
            f"no trained models matching '{MODEL_FILE_PATTERN}' found in {models_dir}"
        )
    return models


def predict_features(
    models: dict[tuple[str, str], object],
    features: pd.DataFrame,
    target: str | None = None,
) -> pd.DataFrame:
    """Predict one or every target; returns a target x model DataFrame.

    The same fitted pipelines (with their own preprocessing) are reused, so
    nothing here re-encodes or reorders features differently from training.
    """
    if target is not None and target not in ML_TARGETS:
        raise ValueError(f"unsupported target {target!r}; supported: {list(ML_TARGETS)}")
    targets = [target] if target is not None else list(ML_TARGETS)
    rows: list[dict] = []
    for current in targets:
        row: dict[str, object] = {"target": current}
        for (model_target, model_name), pipeline in models.items():
            if model_target != current:
                continue
            predicted = pipeline.predict(features)
            if predicted.ndim > 1 and predicted.shape[1] != 1:
                raise ValueError(f"model {model_name!r} returned a multidimensional prediction")
            row[model_name] = float(predicted.ravel()[0])
        rows.append(row)
    result = pd.DataFrame(rows, columns=["target"] + sorted(
        {model for (_, model) in models}
    ))
    return result


def predict_design(
    config_features: dict,
    weather_features: dict,
    models_dir: Path | str,
    target: str | None = None,
) -> pd.DataFrame:
    """Predict targets for one new design plus one weather scenario.

    ``config_features`` carries the design feature values and
    ``weather_features`` the NASA scenario summary features.  Returns the same
    target x model DataFrame as :func:`predict_features`.
    """
    frame = build_feature_row(config_features, weather_features)
    models = load_trained_models(models_dir)
    return predict_features(models, frame, target=target)


def load_dataset_row_features(
    design_id: str,
    weather_scenario_id: str,
    dataset_path: Path | str,
) -> pd.DataFrame:
    """Extract the feature row for an existing ``design_id`` x scenario pair."""
    frame = pd.read_csv(dataset_path)
    subset = frame[
        (frame["design_id"] == design_id)
        & (frame["weather_scenario_id"] == weather_scenario_id)
    ]
    if subset.empty:
        raise ValueError(
            f"no row for design {design_id!r} and scenario {weather_scenario_id!r}"
        )
    return prepare_features(subset.head(1))