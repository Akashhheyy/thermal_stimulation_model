"""Tests for the recommendation command-line interface.

These tests exercise the CLI plumbing (parser, argument handling, JSON
output shape, error paths) using a fake predictor bundle stub so no
trained artifacts are required.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from building_hvac_twin.recommendation.cli import main


class _FakeOutcome:
    """Minimal stand-in for PredictionOutcome returned by predict_design."""

    def __init__(self, design_id, scenario):
        self.design_id = design_id
        self.weather_scenario_id = scenario
        self.input_mode = "config"
        self.raw_predictions = {}
        self.primary_predictions = {
            "percent_time_comfortable": {"value": 50.0, "model": "random_forest", "in_bounds": True, "display_value": 50.0},
            "total_heat_loss_kwh": {"value": 100.0, "model": "random_forest", "in_bounds": True, "display_value": 100.0},
        }
        self.out_of_bounds = {}
        self.provenance = "test"
        self.artifact_info = None

    def to_dict(self):
        return {
            "design_id": self.design_id,
            "weather_scenario_id": self.weather_scenario_id,
            "input_mode": self.input_mode,
            "raw_predictions": self.raw_predictions,
            "primary_predictions": self.primary_predictions,
            "out_of_bounds": self.out_of_bounds,
            "provenance": self.provenance,
            "surrogate_model_disclaimer": "ML models are surrogate models.",
            "artifact_info": {},
        }


class _FakeBundle:
    """Minimal stand-in for PredictorBundle used only by the CLI tests."""

    targets = ("percent_time_comfortable", "total_heat_loss_kwh")
    model_names = ("random_forest",)
    primary_models = {"percent_time_comfortable": "random_forest", "total_heat_loss_kwh": "random_forest"}
    models = {}
    models_dir = "fake"
    artifact_info = None



def test_predict_prints_json(capsys):
    """predict subcommand prints a JSON object with provenance keys."""
    with patch("building_hvac_twin.recommendation.cli.predict_design") as mock_predict:
        mock_predict.return_value = _FakeOutcome("D0001", "S01_winter")
        code = main(["predict", "--scenario", "S01_winter", "--design-id", "D0001"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["design_id"] == "D0001"
    assert out["weather_scenario_id"] == "S01_winter"
    assert "surrogate_model_disclaimer" in out
    mock_predict.assert_called_once()


def test_recommend_prints_ranking(capsys):
    """recommend subcommand generates designs, predicts and ranks them."""
    designs = [{"design_id": "D0001"}, {"design_id": "D0002"}]
    outcomes = [_FakeOutcome(d["design_id"], "S01_winter") for d in designs]
    with patch("building_hvac_twin.recommendation.cli.load_predictors") as mock_load, \
         patch("building_hvac_twin.recommendation.cli.predict_design", return_value=outcomes[0]) as mock_pred, \
         patch("building_hvac_twin.shelter.ml_dataset.generate_designs", return_value=designs) as mock_gen, \
         patch("building_hvac_twin.recommendation.cli._design_to_config") as mock_config, \
         patch("building_hvac_twin.recommendation.cli.rank_designs") as mock_rank:
        mock_load.return_value = _FakeBundle()
        fake1 = MagicMock()
        fake1.design_id = "D0001"
        fake1.rank = 1
        fake1.recommendation_score = 85.0
        fake1.components = {"percent_time_comfortable": 0.9}
        fake1.primary_predictions = outcomes[0].primary_predictions
        fake2 = MagicMock()
        fake2.design_id = "D0002"
        fake2.rank = 2
        fake2.recommendation_score = 70.0
        fake2.components = {"percent_time_comfortable": 0.7}
        fake2.primary_predictions = outcomes[0].primary_predictions
        mock_rank.return_value = [fake1, fake2]
        code = main(["recommend", "--scenario", "S01_winter", "--count", "2"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scenario"] == "S01_winter"
    assert out["count"] == 2
    assert "ranking" in out
    assert "surrogate_model_disclaimer" in out


def test_compare_prints_json(capsys, tmp_path):
    """compare subcommand cross-checks ML against physics."""
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame([{"design_id": "D0001"}]).to_csv(dataset, index=False)
    weather = pd.DataFrame({"outdoor_temperature_c": [-5.0], "solar_radiation_w_m2": [0.0]})
    with patch("building_hvac_twin.recommendation.cli.load_predictors") as mock_load, \
         patch("building_hvac_twin.recommendation.cli.load_scenario_weather", return_value=weather), \
         patch("building_hvac_twin.shelter.ml_dataset.build_shelter_config") as mock_config, \
         patch("building_hvac_twin.recommendation.cli.compare_prediction_with_physics") as mock_compare:
        mock_load.return_value = _FakeBundle()
        mock_compare.return_value = {"design_id": "D0001", "rows": [], "provenance": "test"}
        code = main([
            "compare",
            "--scenario", "S01_winter",
            "--design-id", "D0001",
            "--dataset", str(dataset),
            "--metadata", str(tmp_path / "meta.json"),
        ])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert "design_id" in out


def test_missing_scenario_returns_error(capsys):
    """predict without --scenario fails."""
    code = main(["predict"])
    assert code == 2
    err = capsys.readouterr().err
    assert "scenario" in err.lower()


def test_unknown_scenario_returns_error(capsys):
    """predict with a bad scenario prints an error, not a traceback."""
    with patch("building_hvac_twin.recommendation.cli.load_predictors"), \
         patch("building_hvac_twin.recommendation.cli.predict_design", side_effect=ValueError("no such scenario")):
        code = main(["predict", "--scenario", "S999", "--design-id", "D0001"])
    assert code == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()
