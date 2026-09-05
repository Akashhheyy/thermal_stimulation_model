"""Collection names and MongoDB document builders.

Document builders convert EXISTING project representations (dataset design
rows, dataset metadata scenario entries, and the application results of the
existing prediction, ranking and comparison functions) into plain BSON-safe
dicts.  No new design parameters, targets or scenarios are invented here.

All timestamps are UTC.  Provenance statements are carried into every stored
document so database records never present model outputs as measurements.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

__all__ = [
    "DESIGNS_COLLECTION",
    "WEATHER_SCENARIOS_COLLECTION",
    "PREDICTIONS_COLLECTION",
    "RECOMMENDATIONS_COLLECTION",
    "COMPARISONS_COLLECTION",
    "COLLECTION_NAMES",
    "now_utc",
    "design_document",
    "scenario_document",
    "prediction_document",
    "recommendation_document",
    "comparison_document",
]

DESIGNS_COLLECTION = "designs"
WEATHER_SCENARIOS_COLLECTION = "weather_scenarios"
PREDICTIONS_COLLECTION = "predictions"
RECOMMENDATIONS_COLLECTION = "recommendations"
COMPARISONS_COLLECTION = "comparisons"

COLLECTION_NAMES = (
    DESIGNS_COLLECTION,
    WEATHER_SCENARIOS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RECOMMENDATIONS_COLLECTION,
    COMPARISONS_COLLECTION,
)


def now_utc() -> datetime:
    """Current UTC time (MongoDB stores datetimes natively)."""
    return datetime.now(timezone.utc)


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars (from the dataset CSV) to BSON-safe builtins."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def design_document(design_row: dict[str, Any], design_parameters: tuple[str, ...]) -> dict[str, Any]:
    """Build a design document from one existing dataset design row.

    Only the columns in ``design_parameters`` (the shelter design parameter
    columns already defined by ``shelter.ml_dataset``) are stored; weather,
    target and other columns are deliberately excluded.
    """
    parameters: dict[str, Any] = {}
    for column in design_parameters:
        if column in design_row and design_row[column] is not None:
            parameters[column] = _to_builtin(design_row[column])
    design_id = str(parameters.pop("design_id"))
    return {
        "design_id": design_id,
        "design_parameters": parameters,
        "source": "shelter_ml_dataset",
        "updated_at_utc": now_utc(),
    }


def scenario_document(
    scenario: dict[str, Any],
    nasa_power: dict[str, Any],
) -> dict[str, Any]:
    """Build a scenario document from the existing dataset metadata entry."""
    return {
        "scenario_id": str(scenario["scenario_id"]),
        "name": scenario.get("name"),
        "season": scenario.get("season"),
        "requested_date": scenario.get("requested_date"),
        "effective_date": scenario.get("effective_date"),
        "date_was_replaced": scenario.get("date_was_replaced"),
        "retrieval_status": scenario.get("retrieval_status"),
        "transport_kind": scenario.get("transport_kind"),
        "weather_record_count": scenario.get("weather_record_count"),
        "mean_outdoor_temperature_c": scenario.get("mean_outdoor_temperature_c"),
        "minimum_outdoor_temperature_c": scenario.get("minimum_outdoor_temperature_c"),
        "maximum_outdoor_temperature_c": scenario.get("maximum_outdoor_temperature_c"),
        "daily_solar_sum_wh_m2": scenario.get("daily_solar_sum_wh_m2"),
        "mean_wind_speed_m_s": scenario.get("mean_wind_speed_m_s"),
        "mean_relative_humidity_percent": scenario.get("mean_relative_humidity_percent"),
        "location_name": nasa_power.get("location_name"),
        "latitude": nasa_power.get("latitude"),
        "longitude": nasa_power.get("longitude"),
        "nasa_power_source": nasa_power.get("source"),
        "nasa_provenance_statement": nasa_power.get("provenance_statement"),
        "payload_sha256": scenario.get("payload_sha256"),
        "source": "shelter_ml_dataset_metadata",
        "updated_at_utc": now_utc(),
    }


def prediction_document(outcome: dict[str, Any]) -> dict[str, Any]:
    """Build a prediction document from ``PredictionOutcome.to_dict()``.

    Stores the 9 existing physical ML target predictions and the primary
    model per target.  ``performance_score`` is not an ML target and is never
    stored.
    """
    return {
        "design_id": outcome.get("design_id"),
        "scenario_id": outcome.get("weather_scenario_id"),
        "input_mode": outcome.get("input_mode"),
        "primary_predictions": outcome.get("primary_predictions"),
        "raw_predictions": outcome.get("raw_predictions"),
        "out_of_bounds": outcome.get("out_of_bounds"),
        "primary_models": outcome.get("artifact_info", {}).get("primary_models"),
        "provenance": outcome.get("provenance"),
        "surrogate_model_disclaimer": outcome.get("surrogate_model_disclaimer"),
        "created_at_utc": now_utc(),
    }


def recommendation_document(response: dict[str, Any]) -> dict[str, Any]:
    """Build a recommendation document from a ``RecommendResponse`` body."""
    return {
        "scenario_id": response.get("scenario_id"),
        "count": response.get("count"),
        "objectives": response.get("objectives"),
        "ranking": [
            {
                "rank": candidate["rank"],
                "design_id": candidate["design_id"],
                "recommendation_score": candidate["recommendation_score"],
                "components": candidate["components"],
                "primary_predictions": candidate["primary_predictions"],
            }
            for candidate in response.get("ranking", [])
        ],
        "provenance": response.get("provenance"),
        "surrogate_model_disclaimer": response.get("surrogate_model_disclaimer"),
        "created_at_utc": now_utc(),
    }


def comparison_document(response: dict[str, Any]) -> dict[str, Any]:
    """Build a comparison document from a ``CompareResponse`` body."""
    return {
        "design_id": response.get("design_id"),
        "scenario_id": response.get("scenario_id"),
        "compared_targets": response.get("compared_targets"),
        "rows": response.get("rows"),
        "provenance": response.get("provenance"),
        "surrogate_model_disclaimer": response.get("surrogate_model_disclaimer"),
        "created_at_utc": now_utc(),
    }
