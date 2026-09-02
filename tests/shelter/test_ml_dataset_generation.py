"""Offline tests for the ML dataset generation pipeline.

Every NASA POWER response in this module comes from a fake in-process
transport, so no test here touches the network.  The live retrieval path is
exercised separately by the generator script, never by the test suite.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from building_hvac_twin.shelter import (
    DEFAULT_SCENARIOS,
    DEFAULT_DESIGN_COUNT,
    clear_weather_cache,
    build_geometry,
    validate_shelter_config,
)
from building_hvac_twin.shelter.ml_dataset import (
    DATASET_COLUMNS,
    DESIGN_PARAMETER_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMNS as ML_TARGET_COLUMNS,
    WEATHER_SOURCE_LABEL,
    WeatherScenario,
    build_shelter_config,
    fetch_or_load_weather,
    generate_designs,
    generate_ml_dataset,
    write_dataset,
)

# Deterministic seasonal profile for the fake NASA transport (SYNTHETIC TEST
# PAYLOADS used only to exercise the pipeline offline; they never enter any
# generated dataset file).
MONTH_BASE_TEMP_C = {
    1: -10.0, 2: -8.0, 3: -3.0, 4: 2.0, 5: 6.0, 6: 10.0,
    7: 13.0, 8: 12.0, 9: 8.0, 10: 2.0, 11: -4.0, 12: -8.0,
}
MONTH_PEAK_SOLAR_W_M2 = {
    1: 450.0, 2: 520.0, 3: 650.0, 4: 750.0, 5: 850.0, 6: 900.0,
    7: 800.0, 8: 780.0, 9: 680.0, 10: 560.0, 11: 470.0, 12: 430.0,
}


def payload_for_date(nasa_date: str) -> dict:
    """Build a NASA POWER shaped payload for one UTC day (24 hourly keys)."""
    day = datetime.strptime(nasa_date, "%Y%m%d")
    base = MONTH_BASE_TEMP_C[day.month]
    peak = MONTH_PEAK_SOLAR_W_M2[day.month]
    temperatures, solar, wind, humidity = {}, {}, {}, {}
    for hour in range(24):
        key = f"{nasa_date}{hour:02d}"
        temperatures[key] = round(base + 5.0 * np.sin(2.0 * np.pi * (hour - 8) / 24.0), 2)
        solar[key] = round(max(0.0, peak * np.sin(np.pi * (hour - 7) / 12.0)), 1)
        wind[key] = 2.5
        humidity[key] = 41.0
    return {
        "properties": {
            "parameter": {
                "T2M": temperatures,
                "ALLSKY_SFC_SW_DWN": solar,
                "WS10M": wind,
                "RH2M": humidity,
            }
        }
    }


class FakeNasaTransport:
    """Fake transport returning deterministic payloads; records every URL."""

    def __init__(self, fail_dates=()):
        self.fail_dates = set(fail_dates)
        self.calls: list[str] = []

    def __call__(self, url: str, timeout_seconds: float) -> bytes:
        self.calls.append(url)
        start = url.split("start=")[1].split("&")[0]
        if start in self.fail_dates:
            raise RuntimeError("simulated NASA network failure")
        return json.dumps(payload_for_date(start)).encode()


@pytest.fixture(autouse=True)
def clean_cache():
    clear_weather_cache()
    yield
    clear_weather_cache()


@pytest.fixture
def small_run(tmp_path):
    """A small mocked dataset run: 10 designs x 3 scenarios = 30 rows."""
    transport = FakeNasaTransport()
    result = generate_ml_dataset(
        design_count=10,
        seed=7,
        scenarios=list(DEFAULT_SCENARIOS[:3]),
        cache_dir=tmp_path / "cache",
        transport=transport,
        timeout_seconds=5.0,
    )
    return result, transport


def test_design_generation_is_deterministic():
    first = generate_designs(25, seed=3)
    second = generate_designs(25, seed=3)
    assert first == second
    assert [d["design_id"] for d in first] == [d["design_id"] for d in second]
    assert generate_designs(25, seed=4) != first


def test_all_generated_designs_pass_shelter_validation():
    for design in generate_designs(25, seed=3):
        config = build_shelter_config(design)
        assert validate_shelter_config(config) == []


def test_generated_designs_are_unique():
    designs = generate_designs(60, seed=3)
    keys = [
        tuple(
            d[column]
            for column in DESIGN_PARAMETER_COLUMNS
            if column in d and column != "design_id"
        )
        for d in designs
    ]
    assert len(set(keys)) == len(designs) == 60


def test_opening_areas_fit_within_wall_area():
    for design in generate_designs(25, seed=3):
        config = build_shelter_config(design)
        geometry = build_geometry(config.geometry, config.openings)
        for direction in ("north", "east", "south", "west"):
            openings = (
                geometry.window_area_by_orientation_m2[direction]
                + geometry.door_area_by_orientation_m2[direction]
            )
            assert openings <= geometry.gross_wall_area_by_orientation_m2[direction]
            assert geometry.wall_area_by_orientation_m2[direction] >= 0.0


def test_row_count_equals_designs_times_scenarios(small_run):
    result, _ = small_run
    assert len(result.frame) == 10 * 3


def test_default_scale_targets_are_in_range():
    assert 200 <= DEFAULT_DESIGN_COUNT <= 500
    assert len(DEFAULT_SCENARIOS) == 10
    assert 2000 <= DEFAULT_DESIGN_COUNT * len(DEFAULT_SCENARIOS) <= 5000


def test_dataset_columns_exist_in_fixed_order(small_run):
    result, _ = small_run
    assert list(result.frame.columns) == list(DATASET_COLUMNS)


def test_no_nan_in_required_columns(small_run):
    result, _ = small_run
    for column in REQUIRED_COLUMNS:
        assert not result.frame[column].isna().any(), column


def test_numeric_columns_are_finite(small_run):
    result, _ = small_run
    numeric = result.frame.select_dtypes(include=[np.number]).drop(
        columns=list(OPTIONAL_COLUMNS)
    )
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_no_duplicate_design_scenario_rows(small_run):
    result, _ = small_run
    assert not result.frame.duplicated(subset=["design_id", "weather_scenario_id"]).any()


def test_design_ids_are_stable_across_scenarios(small_run):
    result, _ = small_run
    per_scenario = result.frame.groupby("weather_scenario_id")["design_id"].apply(
        lambda values: list(values)
    )
    frames = list(per_scenario)
    assert all(frame == frames[0] for frame in frames)
    assert len(set(frames[0])) == 10

def test_nasa_provenance_labels_are_correct(small_run):
    result, _ = small_run
    frame = result.frame
    assert set(frame["weather_source"].unique()) == {WEATHER_SOURCE_LABEL}
    assert set(frame["weather_data_category"].unique()) == {"nasa_power_satellite_reanalysis"}
    assert frame["weather_provenance"].str.startswith("NASA POWER data are").all()


def test_scenario_metadata_records_requested_dates(small_run):
    result, _ = small_run
    scenarios = result.metadata["weather_scenarios"]
    assert scenarios["count_used"] == 3
    assert scenarios["count_failed"] == 0
    used_ids = [info["scenario_id"] for info in scenarios["used"]]
    assert used_ids == ["S01_winter", "S02_late_winter", "S03_spring"]
    assert scenarios["used"][0]["requested_date"] == "2024-01-15"
    assert scenarios["used"][0]["effective_date"] == "2024-01-15"
    assert not scenarios["used"][0]["date_was_replaced"]


def test_dataset_is_deterministic_for_same_seed_and_weather(tmp_path):
    first = generate_ml_dataset(
        design_count=6,
        seed=11,
        scenarios=list(DEFAULT_SCENARIOS[:2]),
        cache_dir=tmp_path / "cache_one",
        transport=FakeNasaTransport(),
        timeout_seconds=5.0,
    )
    second = generate_ml_dataset(
        design_count=6,
        seed=11,
        scenarios=list(DEFAULT_SCENARIOS[:2]),
        cache_dir=tmp_path / "cache_two",
        transport=FakeNasaTransport(),
        timeout_seconds=5.0,
    )
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_disk_cache_prevents_refetch_and_preserves_data(tmp_path):
    transport = FakeNasaTransport()
    scenario = DEFAULT_SCENARIOS[0]
    first_frame, first_info = fetch_or_load_weather(
        scenario, tmp_path / "cache", transport=transport
    )
    assert len(transport.calls) == 1
    failing = FakeNasaTransport(fail_dates={"20240115", "20240122"})
    second_frame, second_info = fetch_or_load_weather(
        scenario, tmp_path / "cache", transport=failing
    )
    assert failing.calls == [], "cached scenario must never re-contact NASA"
    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert second_info["retrieval_status"] == "disk_cache_from_injected_transport"


def test_fallback_date_used_only_when_requested_date_fails(tmp_path):
    transport = FakeNasaTransport(fail_dates={"20240115"})
    scenario = DEFAULT_SCENARIOS[0]
    _, info = fetch_or_load_weather(
        scenario, tmp_path / "cache", transport=transport
    )
    assert info["date_was_replaced"] is True
    assert info["effective_date"] == "2024-01-22"
    assert len(transport.calls) == 2


def test_failing_scenario_is_skipped_without_synthetic_weather(tmp_path):
    transport = FakeNasaTransport(fail_dates={"20240115", "20240122"})
    result = generate_ml_dataset(
        design_count=5,
        seed=5,
        scenarios=list(DEFAULT_SCENARIOS[:2]),
        cache_dir=tmp_path / "cache",
        transport=transport,
        timeout_seconds=5.0,
    )
    assert result.metadata["weather_scenarios"]["count_failed"] == 1
    assert result.metadata["weather_scenarios"]["failed"][0]["scenario_id"] == "S01_winter"
    assert len(result.frame) == 5 * 1
    assert set(result.frame["weather_scenario_id"].unique()) == {"S02_late_winter"}


def test_csv_round_trip(tmp_path, small_run):
    result, _ = small_run
    path = write_dataset(result.frame, tmp_path / "out" / "dataset.csv")
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == list(DATASET_COLUMNS)
    assert len(loaded) == len(result.frame)
    assert not loaded.duplicated(subset=["design_id", "weather_scenario_id"]).any()
    numeric = loaded[list(ML_TARGET_COLUMNS)].to_numpy(dtype=float)
    expected = result.frame[list(ML_TARGET_COLUMNS)].to_numpy(dtype=float)
    assert numeric == pytest.approx(expected)


def test_performance_score_is_not_a_dataset_column_or_target(small_run):
    result, _ = small_run
    assert "performance_score" not in result.frame.columns
    assert "performance_score" not in ML_TARGET_COLUMNS
    assert "performance_score" in result.metadata["dataset"]["excluded_columns"]


def test_no_auxiliary_energy_columns_are_fabricated(small_run):
    result, _ = small_run
    forbidden = (
        "auxiliary_heating_kwh",
        "auxiliary_cooling_kwh",
        "heating_energy_kwh",
        "cooling_energy_kwh",
        "hvac_electricity_kwh",
        "heating_load_kw",
        "cooling_load_kw",
    )
    present = [column for column in forbidden if column in result.frame.columns]
    assert not present, present


