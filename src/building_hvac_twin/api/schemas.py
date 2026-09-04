"""Pydantic request and response schemas for the HTTP API layer.

The API is a thin transport on top of the existing recommendation, ML and
thermal packages.  These schemas validate transport input and shape output;
they never define new targets, new designs or new ranking logic.  All 9
targets are the existing physical ML targets from
``building_hvac_twin.recommendation.schemas``; ``performance_score`` is
deliberately absent.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..recommendation.schemas import (
    NASA_PROVENANCE_STATEMENT,
    PHYSICAL_TARGETS,
    SURROGATE_MODEL_DISCLAIMER,
)

__all__ = [
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "RecommendRequest",
    "RankedCandidate",
    "RecommendResponse",
    "CompareRequest",
    "ComparisonRow",
    "CompareResponse",
    "ScenarioSummary",
    "ScenariosResponse",
    "DesignSummary",
    "DesignsResponse",
]

# Shared provenance text, imported from the existing recommendation layer so
# the API can never drift from it.
NASA_PROVENANCE = NASA_PROVENANCE_STATEMENT
SURROGATE_DISCLAIMER = SURROGATE_MODEL_DISCLAIMER


class HealthResponse(BaseModel):
    """Liveness plus a minimal artifact inventory."""

    status: str
    targets_loaded: int
    models_loaded: int


class PredictRequest(BaseModel):
    """One existing dataset design under one existing NASA POWER scenario."""

    design_id: str = Field(
        min_length=1,
        description="Existing shelter design id from the ML dataset, e.g. D0002.",
    )
    scenario_id: str = Field(
        min_length=1,
        description="Existing NASA POWER weather scenario id, e.g. S01_winter.",
    )


class PredictResponse(BaseModel):
    """Result of the existing ``predict_design`` for one design and scenario."""

    design_id: str | None
    scenario_id: str | None
    input_mode: str
    targets: list[str]
    raw_predictions: dict[str, dict[str, float]]
    primary_predictions: dict[str, dict[str, Any]]
    out_of_bounds: dict[str, list[str]]
    provenance: str
    nasa_provenance_statement: str = NASA_PROVENANCE
    surrogate_model_disclaimer: str = SURROGATE_DISCLAIMER
    artifact_info: dict[str, Any]


class RecommendRequest(BaseModel):
    """Rank candidate designs for one scenario using the existing ranking."""

    scenario_id: str = Field(
        min_length=1,
        description="Existing NASA POWER weather scenario id, e.g. S01_winter.",
    )
    count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of candidate designs to generate and rank.",
    )
    seed: int = Field(
        default=42,
        description="Seed for the existing deterministic design generator.",
    )


class RankedCandidate(BaseModel):
    """One entry of the existing deterministic ranking output."""

    design_id: str
    rank: int
    recommendation_score: float
    components: dict[str, float]
    primary_predictions: dict[str, dict[str, Any]]


class RecommendResponse(BaseModel):
    """Ranked candidates from the existing ``rank_designs`` implementation."""

    scenario_id: str
    count: int
    objectives: list[dict[str, Any]]
    ranking: list[RankedCandidate]
    provenance: str
    nasa_provenance_statement: str = NASA_PROVENANCE
    surrogate_model_disclaimer: str = SURROGATE_DISCLAIMER


class CompareRequest(BaseModel):
    """Cross-check one existing design between ML and the thermal engine."""

    design_id: str = Field(
        min_length=1,
        description="Existing shelter design id from the ML dataset, e.g. D0002.",
    )
    scenario_id: str = Field(
        min_length=1,
        description="Existing NASA POWER weather scenario id, e.g. S01_winter.",
    )


class ComparisonRow(BaseModel):
    """Per-target ML-vs-physics row from the existing comparison function."""

    target: str
    ml_prediction: float
    ml_model: str | None
    physics_result: float
    absolute_error: float
    relative_error: float | None


class CompareResponse(BaseModel):
    """Existing ``compare_prediction_with_physics`` result over HTTP."""

    design_id: str
    scenario_id: str
    compared_targets: list[str]
    rows: list[ComparisonRow]
    provenance: str
    nasa_provenance_statement: str = NASA_PROVENANCE
    surrogate_model_disclaimer: str = SURROGATE_DISCLAIMER


class ScenarioSummary(BaseModel):
    """One NASA POWER weather scenario as recorded in the dataset metadata."""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    name: str | None = None
    season: str | None = None
    requested_date: str | None = None
    effective_date: str | None = None
    date_was_replaced: bool | None = None
    retrieval_status: str | None = None
    transport_kind: str | None = None
    weather_record_count: int | None = None
    mean_outdoor_temperature_c: float | None = None
    minimum_outdoor_temperature_c: float | None = None
    maximum_outdoor_temperature_c: float | None = None
    daily_solar_sum_wh_m2: float | None = None
    mean_wind_speed_m_s: float | None = None
    mean_relative_humidity_percent: float | None = None


class ScenariosResponse(BaseModel):
    """The existing NASA POWER scenario catalog; nothing synthetic is added."""

    count: int
    location_name: str
    latitude: float
    longitude: float
    nasa_power_source: str
    nasa_provenance_statement: str = NASA_PROVENANCE
    scenarios: list[ScenarioSummary]


class DesignSummary(BaseModel):
    """One shelter design as represented in the existing ML dataset."""

    design_id: str
    design_parameters: dict[str, Any]


class DesignsResponse(BaseModel):
    """The existing shelter design catalog from the ML dataset."""

    count: int
    designs: list[DesignSummary]
    provenance: str = (
        "Design catalog read from the existing shelter ML dataset; "
        "no designs are invented by the API."
    )
