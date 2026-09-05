"""Repository, document-builder and seed tests using in-memory fakes.

The real dataset CSV and metadata JSON are read (never modified) so the
documents under test are exactly what the seed command would store.
"""
from __future__ import annotations

from pathlib import Path

from building_hvac_twin.database.collections import (
    comparison_document,
    design_document,
    prediction_document,
    recommendation_document,
    scenario_document,
)
from building_hvac_twin.database.seed import seed_database
from building_hvac_twin.shelter.ml_dataset import DESIGN_PARAMETER_COLUMNS

from .fakes import FakeCollection, make_fake_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "shelter_ml_dataset.csv"
METADATA_PATH = REPO_ROOT / "data" / "shelter_ml_dataset_metadata.json"


def _first_design_row() -> dict:
    import pandas as pd

    frame = pd.read_csv(DATASET_PATH)
    return frame.sort_values("design_id").iloc[0].to_dict()


def _first_scenario_entry() -> tuple[dict, dict]:
    import json

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return (
        metadata["weather_scenarios"]["used"][0],
        metadata["nasa_power"],
    )


def test_design_document_uses_only_design_parameters():
    row = _first_design_row()
    document = design_document(row, DESIGN_PARAMETER_COLUMNS)
    assert document["design_id"] == row["design_id"]
    assert set(document["design_parameters"]) == set(DESIGN_PARAMETER_COLUMNS) - {
        "design_id"
    }
    # Weather and target columns never leak into a design document.
    assert "weather_scenario_id" not in document["design_parameters"]
    assert "percent_time_comfortable" not in document["design_parameters"]
    assert document["source"] == "shelter_ml_dataset"
    # BSON-safe builtin types only.
    assert all(
        isinstance(value, (int, float, str, bool))
        for value in document["design_parameters"].values()
    )


def test_scenario_document_carries_nasa_provenance():
    scenario, nasa_power = _first_scenario_entry()
    document = scenario_document(scenario, nasa_power)
    assert document["scenario_id"] == scenario["scenario_id"]
    assert "NASA POWER" in document["nasa_provenance_statement"]
    assert document["location_name"] == nasa_power["location_name"]
    assert document["weather_record_count"] == scenario["weather_record_count"]


def test_design_upsert_is_idempotent():
    bundle = make_fake_bundle()
    row = _first_design_row()
    document = design_document(row, DESIGN_PARAMETER_COLUMNS)
    bundle.designs.ensure_indexes()
    assert bundle.designs.upsert_many([document]) == 1
    assert bundle.designs.upsert_many([document]) == 1
    assert bundle.designs.count() == 1
    stored = bundle.designs.get(document["design_id"])
    assert stored["design_parameters"]["wall_material"] == row["wall_material"]
    # A unique index guards against duplicate designs.
    assert any(
        "design_id" in str(index) for index in bundle.designs._collection.indexes
    )


def test_scenario_upsert_is_idempotent():
    bundle = make_fake_bundle()
    scenario, nasa_power = _first_scenario_entry()
    document = scenario_document(scenario, nasa_power)
    bundle.weather_scenarios.upsert_many([document])
    bundle.weather_scenarios.upsert_many([document])
    assert bundle.weather_scenarios.count() == 1


def test_seed_database_is_idempotent():
    bundle = make_fake_bundle()
    first = seed_database(bundle, DATASET_PATH, METADATA_PATH)
    assert first["designs"] == 300
    assert first["weather_scenarios"] == 10
    second = seed_database(bundle, DATASET_PATH, METADATA_PATH)
    # Running the seed twice changes nothing: no duplicates, same totals.
    assert second == first


def test_prediction_document_and_insert():
    document = prediction_document(
        {
            "design_id": "D0002",
            "weather_scenario_id": "S01_winter",
            "input_mode": "dataset_row",
            "primary_predictions": {
                "percent_time_comfortable": {
                    "value": 1.0,
                    "model": "gradient_boosting",
                    "in_bounds": True,
                    "display_value": 1.0,
                }
            },
            "raw_predictions": {},
            "out_of_bounds": {},
            "provenance": "test provenance",
            "surrogate_model_disclaimer": "test disclaimer",
            "artifact_info": {
                "primary_models": {"percent_time_comfortable": "gradient_boosting"}
            },
        }
    )
    # performance_score is not an ML target and is never stored.
    assert "performance_score" not in document
    assert document["primary_models"] == {
        "percent_time_comfortable": "gradient_boosting"
    }
    assert document["created_at_utc"] is not None

    bundle = make_fake_bundle()
    assert bundle.save_prediction(document).saved is True
    assert bundle.predictions._collection.count_documents({}) == 1


def test_recommendation_and_comparison_inserts():
    bundle = make_fake_bundle()
    recommendation = recommendation_document(
        {
            "scenario_id": "S01_winter",
            "count": 1,
            "objectives": [
                {
                    "target": "percent_time_comfortable",
                    "direction": "maximize",
                    "weight": 1.0,
                }
            ],
            "ranking": [
                {
                    "rank": 1,
                    "design_id": "D0002",
                    "recommendation_score": 88.0,
                    "components": {"percent_time_comfortable": 0.88},
                    "primary_predictions": {},
                }
            ],
            "provenance": "test",
            "surrogate_model_disclaimer": "test disclaimer",
        }
    )
    assert bundle.save_recommendation(recommendation).saved is True
    stored = bundle.recommendations._collection.documents[0]
    assert stored["ranking"][0]["design_id"] == "D0002"

    comparison = comparison_document(
        {
            "design_id": "D0002",
            "scenario_id": "S01_winter",
            "compared_targets": ["percent_time_comfortable"],
            "rows": [
                {
                    "target": "percent_time_comfortable",
                    "ml_prediction": 1.0,
                    "ml_model": "gradient_boosting",
                    "physics_result": 1.2,
                    "absolute_error": -0.2,
                    "relative_error": -0.166,
                }
            ],
            "provenance": "ML column = surrogate prediction",
            "surrogate_model_disclaimer": "test disclaimer",
        }
    )
    assert bundle.save_comparison(comparison).saved is True
    stored = bundle.comparisons._collection.documents[0]
    # The ML-vs-physics distinction travels with the stored document.
    assert "surrogate" in stored["provenance"]
    assert stored["rows"][0]["ml_model"] == "gradient_boosting"


def test_persistence_failure_is_reported_not_raised():
    from pymongo.errors import PyMongoError

    from building_hvac_twin.database.repositories import ComparisonRepository

    class FailingCollection(FakeCollection):
        def insert_one(self, document):
            raise PyMongoError("database unavailable")

    bundle = make_fake_bundle()
    bundle.comparisons = ComparisonRepository(FailingCollection("comparisons"))
    result = bundle.save_comparison({"design_id": "D0002"})
    assert result.saved is False
    assert "database unavailable" in result.detail


def test_fake_bundle_ping():
    bundle = make_fake_bundle()
    assert bundle.ping() is True
