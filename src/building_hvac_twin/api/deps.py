"""Shared request-scoped helpers for the API routes.

The helpers only read state that the application lifespan loaded once onto
``app.state`` and translate the existing packages' errors into HTTP errors.
All business logic stays in the existing recommendation, ML and shelter
packages.
"""
from __future__ import annotations

import pandas as pd
from fastapi import HTTPException, Request

from ..recommendation.predictor import PredictorBundle

__all__ = [
    "get_bundle",
    "get_dataset",
    "get_metadata",
    "require_scenario",
    "require_design_id",
]


def get_bundle(request: Request) -> PredictorBundle:
    """Return the predictor bundle loaded once at application startup."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="trained models are not loaded; check server startup logs",
        )
    return bundle


def get_dataset(request: Request) -> pd.DataFrame:
    """Return the existing ML dataset frame loaded at startup."""
    dataset = getattr(request.app.state, "dataset", None)
    if dataset is None:
        raise HTTPException(
            status_code=503,
            detail="ML dataset is not loaded; check server startup logs",
        )
    return dataset


def get_metadata(request: Request) -> dict:
    """Return the existing dataset metadata loaded at startup."""
    metadata = getattr(request.app.state, "metadata", None)
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="dataset metadata is not loaded; check server startup logs",
        )
    return metadata


def require_scenario(metadata: dict, scenario_id: str) -> dict:
    """Return the metadata record for ``scenario_id`` or raise 404.

    Only scenarios that exist in the dataset metadata (retrieved from NASA
    POWER during dataset generation) are accepted; the API never invents
    weather.
    """
    scenarios = metadata.get("weather_scenarios", {}).get("used", [])
    for scenario in scenarios:
        if scenario.get("scenario_id") == scenario_id:
            return scenario
    available = [entry.get("scenario_id") for entry in scenarios]
    raise HTTPException(
        status_code=404,
        detail=(
            f"unknown weather scenario {scenario_id!r}; "
            f"available scenarios: {available}"
        ),
    )


def require_design_id(dataset: pd.DataFrame, design_id: str) -> None:
    """Raise 404 unless ``design_id`` exists in the existing ML dataset."""
    known = set(dataset["design_id"].unique())
    if design_id not in known:
        sample = sorted(known)[:5]
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown design_id {design_id!r}; the dataset contains "
                f"{len(known)} designs, e.g. {sample}"
            ),
        )
