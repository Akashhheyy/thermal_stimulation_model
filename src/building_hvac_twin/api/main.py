"""FastAPI application for the building energy HVAC digital twin.

A thin transport layer over the EXISTING packages:

- ``building_hvac_twin.recommendation`` for ML prediction, ranking and the
  ML-vs-physics cross-check;
- ``building_hvac_twin.shelter`` for the thermal engine, the design space and
  the NASA POWER scenario catalog.

The trained models, the ML dataset and the dataset metadata are loaded once
at startup and stored on ``app.state``; request handlers never retrain and
never mutate that state.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI

from ..recommendation import (
    DEFAULT_METRICS_REPORT,
    DEFAULT_MODELS_DIR,
    load_predictors,
)
from ..recommendation.predictor import DEFAULT_DATASET_PATH, DEFAULT_METADATA_PATH
from ..recommendation.schemas import PHYSICAL_TARGETS
from .routes import prediction, recommendation, simulation
from .schemas import HealthResponse

# Repository root: src/building_hvac_twin/api/main.py -> parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(env_var: str, default: Path) -> Path:
    """Resolve a data path against the CWD first, then the repository root.

    This keeps the app working whether uvicorn is launched from the repo
    root, another directory, or with an explicit environment override.
    """
    configured = os.environ.get(env_var)
    if configured:
        return Path(configured)
    for base in (Path.cwd(), REPO_ROOT):
        candidate = base / default
        if candidate.exists():
            return candidate
    return Path.cwd() / default


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the existing artifacts once and expose them on ``app.state``."""
    models_dir = _resolve_path("BHVAC_MODELS_DIR", DEFAULT_MODELS_DIR)
    dataset_path = _resolve_path("BHVAC_DATASET_PATH", DEFAULT_DATASET_PATH)
    metadata_path = _resolve_path("BHVAC_METADATA_PATH", DEFAULT_METADATA_PATH)
    metrics_report = _resolve_path("BHVAC_METRICS_REPORT", DEFAULT_METRICS_REPORT)

    app.state.dataset_path = dataset_path
    app.state.metadata_path = metadata_path
    # The existing loader reads only trained joblib artifacts; it never trains.
    app.state.bundle = load_predictors(
        models_dir,
        metrics_report=metrics_report,
    )
    app.state.dataset = pd.read_csv(dataset_path)
    app.state.metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    try:
        yield
    finally:
        app.state.bundle = None
        app.state.dataset = None
        app.state.metadata = None


def create_app() -> FastAPI:
    """Create the FastAPI application (factory keeps tests simple)."""
    app = FastAPI(
        title="Building Energy HVAC Digital Twin API",
        description=(
            "ML surrogate predictions, design recommendations and ML-vs-physics "
            "comparisons for passive shelters, driven by NASA POWER weather "
            "scenarios. All model outputs are estimates, not measurements."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(prediction.router)
    app.include_router(recommendation.router)
    app.include_router(simulation.router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        bundle = getattr(app.state, "bundle", None)
        loaded_models = len(bundle.models) if bundle is not None else 0
        return HealthResponse(
            status="ok",
            targets_loaded=len(PHYSICAL_TARGETS) if bundle is not None else 0,
            models_loaded=loaded_models,
        )

    return app


app = create_app()
