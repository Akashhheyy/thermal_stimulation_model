"""Reproducible training of one fitted pipeline per target and model."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from .features import ML_TARGETS
from .models import MODEL_NAMES

__all__ = [
    "train_models",
    "train_target",
    "save_model",
    "MODEL_FILE_PATTERN",
]

MODEL_FILE_PATTERN = "{target}__{model}.joblib"


def train_target(
    pipeline,
    features: pd.DataFrame,
    target: str,
    target_values: pd.Series,
    train_index: np.ndarray,
) -> object:
    """Fit preprocessing plus regressor on the training rows only."""
    if target not in ML_TARGETS:
        raise ValueError(
            f"{target!r} is not a supported ML target; supported: {list(ML_TARGETS)}"
        )
    x_train = features.iloc[train_index]
    y_train = target_values.iloc[train_index]
    pipeline.fit(x_train, y_train)
    return pipeline


def train_models(
    build_pipelines: Callable[[], dict[str, object]],
    features: pd.DataFrame,
    targets: dict[str, pd.Series],
    train_index: np.ndarray,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> dict[tuple[str, str], object]:
    """Train every requested model on every requested target.

    Returns a mapping keyed by ``(target, model_name)``.  Each (target, model)
    pair gets a FRESH unfitted pipeline so no fitted state is shared.  Iteration
    order is fixed (targets then models) so output ordering is deterministic.
    """
    fitted: dict[tuple[str, str], object] = {}
    templates = build_pipelines()
    unknown = [name for name in model_names if name not in templates]
    if unknown:
        raise KeyError(f"unknown models: {unknown}; available: {sorted(templates)}")
    for target in ML_TARGETS:
        if target not in targets:
            continue
        for model_name in model_names:
            fresh = copy.deepcopy(templates[model_name])
            fitted[(target, model_name)] = train_target(
                fresh, features, target, targets[target], train_index
            )
    return fitted


def save_model(pipeline, directory: Path | str, target: str, model_name: str) -> Path:
    """Persist one fitted pipeline; the artefact contains its preprocessing."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE_PATTERN.format(target=target, model=model_name)
    joblib.dump(pipeline, path)
    return path
