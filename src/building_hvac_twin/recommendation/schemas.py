"""Shared schemas for the ML prediction and design recommendation layer.

This package is an application layer on top of the existing trained surrogate
models.  It never retrains, never touches the thermal engine, and never
invents targets.  All provenance and disclaimer text is centralised here so
every public result carries it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "NASA_PROVENANCE_STATEMENT",
    "SURROGATE_MODEL_DISCLAIMER",
    "DISPLAY_BOUNDS",
    "PHYSICAL_TARGETS",
    "FastObjective",
    "RecommendationObjective",
    "RankedRecommendation",
    "ModelArtifactInfo",
    "PredictionOutcome",
]

# The exact provenance statement used by the dataset phase.  Every result from
# this package carries it so NASA POWER values are never presented as ground
# measurements.
NASA_PROVENANCE_STATEMENT = (
    "NASA POWER data are satellite/reanalysis-derived estimates and are not "
    "ground measurements."
)

# Explicit disclaimer: ML predictions approximate the thermal engine over the
# represented design space and NASA weather scenarios.  They are NOT measured
# building performance.
SURROGATE_MODEL_DISCLAIMER = (
    "ML models are surrogate models of the thermal engine over the represented "
    "design space and NASA weather scenarios.  Predictions are not measured "
    "building performance."
)

# The exact physical targets with trained artifacts.  performance_score and
# auxiliary heating/cooling are deliberately absent (no artifacts, no physics).
PHYSICAL_TARGETS = (
    "percent_time_comfortable",
    "degree_hours_below_comfort",
    "degree_hours_above_comfort",
    "minimum_indoor_temperature_c",
    "mean_indoor_temperature_c",
    "indoor_temperature_range_c",
    "total_heat_loss_kwh",
    "total_solar_gain_kwh",
    "thermal_mass_net_kwh",
)

# Documented bounds used ONLY for a physically bounded presentation value.
# Raw model predictions are always preserved and reported separately; nothing
# is clipped silently.  Bounds here are deliberately conservative.
DISPLAY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "percent_time_comfortable": (0.0, 100.0),
    "degree_hours_below_comfort": (0.0, None),
    "degree_hours_above_comfort": (0.0, None),
    "minimum_indoor_temperature_c": (-273.15, None),
    "mean_indoor_temperature_c": (-273.15, None),
    "indoor_temperature_range_c": (0.0, None),
    "total_heat_loss_kwh": (0.0, None),
    "total_solar_gain_kwh": (0.0, None),
    "thermal_mass_net_kwh": (None, None),
}
class FastObjective(Enum):
    """Direction of a recommendation objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class RecommendationObjective:
    """One ranking objective referencing a physical ML target.

    ``direction`` is ``maximize`` or ``minimize`` and ``weight`` is a
    nonnegative importance for the weighted decision score.  This is an
    application-level decision metric; it is NOT ``performance_score`` from
    ``comparison.py``.
    """

    target: str
    direction: FastObjective | str = FastObjective.MINIMIZE
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.target not in PHYSICAL_TARGETS:
            raise ValueError(
                f"{self.target!r} is not a supported physical ML target; "
                f"supported: {list(PHYSICAL_TARGETS)}"
            )
        direction = (
            self.direction
            if isinstance(self.direction, FastObjective)
            else FastObjective(str(self.direction).lower())
        )
        object.__setattr__(self, "direction", direction)
        if self.weight < 0.0:
            raise ValueError("objective weight must be nonnegative")


@dataclass
class RankedRecommendation:
    """One candidate's position in a deterministic ranking."""

    design_id: str
    rank: int
    recommendation_score: float  # 0..100, application-level decision metric
    components: dict[str, float]  # per-objective normalized contribution 0..1
    primary_predictions: dict[str, dict[str, Any]]
    provenance: str


@dataclass
class ModelArtifactInfo:
    """Version and provenance information about loaded model artifacts."""

    models_dir: str
    file_count: int
    targets: tuple[str, ...]
    model_names: tuple[str, ...]
    training_metadata: dict[str, Any] = field(default_factory=dict)
    primary_models: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models_dir": self.models_dir,
            "file_count": self.file_count,
            "targets": list(self.targets),
            "model_names": list(self.model_names),
            "primary_models": dict(self.primary_models),
            "training_metadata": self.training_metadata,
        }


@dataclass
class PredictionOutcome:
    """Structured result for one ML prediction.

    ``raw_predictions`` carries the unmodified model values per target and
    model.  ``primary_predictions`` carries the selected model value per
    target plus a bounded display value; when a raw value leaves the
    documented physical range, it is flagged in ``out_of_bounds`` instead of
    being hidden.
    """

    design_id: str | None
    weather_scenario_id: str | None
    input_mode: str
    raw_predictions: dict[str, dict[str, float]]  # target -> model -> value
    primary_predictions: dict[str, dict[str, Any]]  # target -> {value, model, in_bounds, display_value}
    out_of_bounds: dict[str, list[str]]  # target -> models with raw values outside bounds
    provenance: str
    artifact_info: ModelArtifactInfo

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "weather_scenario_id": self.weather_scenario_id,
            "input_mode": self.input_mode,
            "raw_predictions": self.raw_predictions,
            "primary_predictions": self.primary_predictions,
            "out_of_bounds": self.out_of_bounds,
            "provenance": self.provenance,
            "surrogate_model_disclaimer": SURROGATE_MODEL_DISCLAIMER,
            "artifact_info": self.artifact_info.to_dict(),
        }