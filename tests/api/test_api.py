"""Tests for the FastAPI layer using FastAPI's TestClient.

The tests run against the real trained artifacts, dataset and metadata in
the repository, exactly as the API would serve them.  No models are
retrained, no weather is fetched (the compare endpoint uses the NASA POWER
disk cache) and no designs are invented.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from building_hvac_twin.api.main import create_app
from building_hvac_twin.recommendation.schemas import (
    NASA_PROVENANCE_STATEMENT,
    PHYSICAL_TARGETS,
    SURROGATE_MODEL_DISCLAIMER,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["targets_loaded"] == len(PHYSICAL_TARGETS)
    assert body["models_loaded"] == 27


def test_scenarios(client):
    response = client.get("/scenarios")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 10
    ids = [scenario["scenario_id"] for scenario in body["scenarios"]]
    assert "S01_winter" in ids
    assert body["nasa_provenance_statement"] == NASA_PROVENANCE_STATEMENT
    assert body["location_name"]
    winter = next(s for s in body["scenarios"] if s["scenario_id"] == "S01_winter")
    assert winter["mean_outdoor_temperature_c"] is not None
    assert winter["weather_record_count"] == 24


def test_designs(client):
    response = client.get("/designs", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    ids = [design["design_id"] for design in body["designs"]]
    assert ids == sorted(ids)
    assert all(design_id.startswith("D") for design_id in ids)
    first = body["designs"][0]
    assert "wall_material" in first["design_parameters"]
    # Full catalog reports every existing design.
    full = client.get("/designs").json()
    assert full["count"] == 300


def test_predict_valid(client):
    response = client.post(
        "/predict",
        json={"design_id": "D0002", "scenario_id": "S01_winter"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["design_id"] == "D0002"
    assert body["scenario_id"] == "S01_winter"
    # Exactly the 9 existing physical ML targets, no invented targets.
    assert sorted(body["targets"]) == sorted(PHYSICAL_TARGETS)
    assert "performance_score" not in body["targets"]
    assert sorted(body["primary_predictions"]) == sorted(PHYSICAL_TARGETS)
    for prediction in body["primary_predictions"].values():
        assert prediction["model"] in {
            "linear_regression",
            "random_forest",
            "gradient_boosting",
        }
    assert body["surrogate_model_disclaimer"] == SURROGATE_MODEL_DISCLAIMER
    assert body["nasa_provenance_statement"] == NASA_PROVENANCE_STATEMENT


def test_predict_unknown_design(client):
    response = client.post(
        "/predict",
        json={"design_id": "D9999", "scenario_id": "S01_winter"},
    )
    assert response.status_code == 404
    assert "unknown design_id" in response.json()["detail"]


def test_predict_unknown_scenario(client):
    response = client.post(
        "/predict",
        json={"design_id": "D0002", "scenario_id": "S99_bogus"},
    )
    assert response.status_code == 404
    assert "unknown weather scenario" in response.json()["detail"]


def test_predict_validation_error(client):
    response = client.post("/predict", json={"design_id": "D0002"})
    assert response.status_code == 422


def test_recommend_valid(client):
    response = client.post(
        "/recommend",
        json={"scenario_id": "S01_winter", "count": 3, "seed": 42},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scenario_id"] == "S01_winter"
    assert body["count"] == 3
    ranks = [candidate["rank"] for candidate in body["ranking"]]
    assert ranks == [1, 2, 3]
    top = body["ranking"][0]
    assert 0.0 <= top["recommendation_score"] <= 100.0
    assert sorted(top["primary_predictions"]) == sorted(PHYSICAL_TARGETS)
    assert body["surrogate_model_disclaimer"] == SURROGATE_MODEL_DISCLAIMER
    # Deterministic: same seed gives the same ordering.
    again = client.post(
        "/recommend",
        json={"scenario_id": "S01_winter", "count": 3, "seed": 42},
    ).json()
    assert [c["design_id"] for c in again["ranking"]] == [
        c["design_id"] for c in body["ranking"]
    ]


def test_recommend_rejects_bad_count(client):
    response = client.post(
        "/recommend",
        json={"scenario_id": "S01_winter", "count": 0},
    )
    assert response.status_code == 422


def test_compare_valid(client):
    response = client.post(
        "/compare",
        json={"design_id": "D0002", "scenario_id": "S01_winter"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["design_id"] == "D0002"
    assert body["scenario_id"] == "S01_winter"
    assert len(body["rows"]) == len(PHYSICAL_TARGETS)
    for row in body["rows"]:
        assert row["target"] in PHYSICAL_TARGETS
        assert "ml_prediction" in row and "physics_result" in row
        assert "ml_model" in row
        assert "absolute_error" in row
    # The distinction between surrogate output and engine output is explicit.
    assert "surrogate" in body["provenance"]
    assert "neither is a measurement" in body["provenance"]
    assert body["surrogate_model_disclaimer"] == SURROGATE_MODEL_DISCLAIMER


def test_compare_unknown_design(client):
    response = client.post(
        "/compare",
        json={"design_id": "NOPE", "scenario_id": "S01_winter"},
    )
    assert response.status_code == 404
