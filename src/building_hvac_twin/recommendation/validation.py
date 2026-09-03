"""Cross-check ML predictions against the physics engine.

The same ``ShelterConfig`` and weather series are sent through both the ML
surrogate and the real thermal engine so the two can be compared target by
target.  This is a diagnostic, not a replacement for measured validation:
both columns are model outputs, and that is stated explicitly in every
result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from building_hvac_twin.shelter import weather

from ..shelter.comparison import design_metrics
from ..shelter.ml_dataset import (
    fetch_or_load_weather,
    get_weather_scenarios,
    summarize_weather,
)
from ..shelter.models import ComfortRange, ShelterConfig
from ..shelter.simulation import simulate_shelter
from .schemas import PHYSICAL_TARGETS
from .predictor import (
    DEFAULT_METADATA_PATH,
    PredictorBundle,
    design_features_from_config,
    weather_features_from_scenario,
)

__all__ = [
    "SCENARIO_WEATHER_CACHE_DIR",
    "load_scenario_weather",
    "compare_prediction_with_physics",
]

# Disk cache holding the raw NASA POWER payloads retrieved during dataset
# generation.  Reusing it keeps this diagnostic offline where present.
SCENARIO_WEATHER_CACHE_DIR = Path("data") / "nasa_weather_raw"

_TARGET_COMPARED = (
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


def load_scenario_weather(scenario_id: str) -> pd.DataFrame:
    """Load one scenario's hourly weather, preferring the disk cache.

    Raises ``ValueError`` when the scenario is unknown.  When the cache file
    is missing and no transport is injected, the underlying retrieval function
    raises its own documented error rather than inventing data.
    """
    scenario = None
    for candidate in get_weather_scenarios():
        if candidate.scenario_id == scenario_id:
            scenario = candidate
            break
    if scenario is None:
        available = [candidate.scenario_id for candidate in get_weather_scenarios()]
        raise ValueError(f"unknown scenario {scenario_id!r}; available: {available}")
    weather, _ = fetch_or_load_weather(scenario, SCENARIO_WEATHER_CACHE_DIR)
    return weather


def _physics_metrics(
    config: ShelterConfig,
    weather: pd.DataFrame,
    comfort_range: ComfortRange | None = None,
) -> dict[str, float]:
    result = simulate_shelter(config, weather)
    metrics = design_metrics(result, comfort_range)
    metrics.pop("design", None)
    return metrics


def compare_prediction_with_physics(
    bundle: PredictorBundle,
    config: ShelterConfig,
    weather: pd.DataFrame,
    *,
    comfort_range: ComfortRange | None = None,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Compare the ML surrogate against the thermal engine for one design.

    Returns a table of per-target ML prediction, physics result, absolute
    error and relative error.  Every value is a model output: the ML column is
    the surrogate and the physics column is the thermal engine.  Neither is a
    measurement, and the returned provenance statement says so.
    """
    scenario_features = summarize_weather(weather)
    scenario_features = {key: float(scenario_features[key]) for key in (
        "mean_outdoor_temperature_c",
        "minimum_outdoor_temperature_c",
        "maximum_outdoor_temperature_c",
        "daily_solar_sum_wh_m2",
    )}

    config_features = design_features_from_config(config)
    from ..ml.features import build_feature_row
    features = build_feature_row(config_features, scenario_features)
    prediction = bundle.predict(features)

    primary_lookup: dict[str, float] = {}
    for row in prediction.itertuples():
        target = str(row.target)
        chosen_model = bundle.primary_models.get(target)
        value = None
        for model_name in [chosen_model] + [m for (_, m) in bundle.models if m != chosen_model]:
            if model_name is None:
                continue
            attr = getattr(row, model_name, None)
            if attr is not None:
                value = float(attr)
                break
        if value is not None:
            primary_lookup[target] = value

    physics = _physics_metrics(config, weather, comfort_range)

    rows: list[dict[str, Any]] = []
    for target in _TARGET_COMPARED:
        ml_value = primary_lookup.get(target)
        physics_value = physics.get(target)
        if ml_value is None or physics_value is None:
            continue
        absolute_error = ml_value - physics_value
        relative_error = (
            absolute_error / physics_value if physics_value != 0.0 else None
        )
        rows.append(
            {
                "target": target,
                "ml_prediction": ml_value,
                "ml_model": bundle.primary_models.get(target),
                "physics_result": physics_value,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
        )

    return {
        "design_id": config.name,
        "compared_targets": _TARGET_COMPARED,
        "rows": rows,
        "provenance": (
            "ML column = surrogate prediction; physics column = thermal engine. "
            "Both are model outputs and neither is a measurement."
        ),
        "surrogate_model_disclaimer": (
            "ML models are surrogate models of the thermal engine over the "
            "represented design space and NASA weather scenarios."
        ),
    }