"""Numerical tests for the user-configurable comfort analysis."""
import pandas as pd
import pytest

from building_hvac_twin.shelter import (
    ComfortRange,
    comfort_metrics,
    comfort_summary,
    simulate_shelter,
)

BAND = ComfortRange(minimum_comfort_temperature_c=18.0, maximum_comfort_temperature_c=24.0)


def test_all_temperatures_inside_range_are_fully_comfortable():
    metrics = comfort_metrics([19.0, 20.0, 23.0], BAND, 1.0)
    assert metrics.percent_time_comfortable == pytest.approx(100.0)
    assert metrics.percent_time_below_comfort == pytest.approx(0.0)
    assert metrics.percent_time_above_comfort == pytest.approx(0.0)
    assert metrics.degree_hours_below_comfort == pytest.approx(0.0)
    assert metrics.degree_hours_above_comfort == pytest.approx(0.0)
    assert metrics.maximum_violation_c == pytest.approx(0.0)


def test_temperatures_below_range_produce_below_comfort_metrics():
    metrics = comfort_metrics([10.0, 15.0], BAND, 2.0)
    # (18-10)*2 + (18-15)*2 = 16 + 6 = 22 degree-hours below.
    assert metrics.degree_hours_below_comfort == pytest.approx(22.0)
    assert metrics.degree_hours_above_comfort == pytest.approx(0.0)
    assert metrics.percent_time_below_comfort == pytest.approx(100.0)
    assert metrics.percent_time_comfortable == pytest.approx(0.0)
    assert metrics.maximum_violation_c == pytest.approx(8.0)
    assert metrics.hours_below_comfort == pytest.approx(4.0)


def test_temperatures_above_range_produce_above_comfort_metrics():
    metrics = comfort_metrics([26.0, 30.0], BAND, 1.0)
    # (26-24)*1 + (30-24)*1 = 2 + 6 = 8 degree-hours above.
    assert metrics.degree_hours_above_comfort == pytest.approx(8.0)
    assert metrics.degree_hours_below_comfort == pytest.approx(0.0)
    assert metrics.percent_time_above_comfort == pytest.approx(100.0)


def test_mixed_series_gives_exact_comfort_percentage():
    # 3 of 4 one-hour steps are comfortable -> 75 percent.
    metrics = comfort_metrics([20.0, 22.0, 16.0, 21.0], BAND, 1.0)
    assert metrics.percent_time_comfortable == pytest.approx(75.0)
    assert metrics.percent_time_below_comfort == pytest.approx(25.0)
    assert metrics.degree_hours_below_comfort == pytest.approx(2.0)
    assert metrics.hours_comfortable == pytest.approx(3.0)


def test_summary_statistics_are_exact():
    metrics = comfort_metrics([10.0, 20.0, 30.0], BAND, 1.0)
    assert metrics.minimum_indoor_temperature_c == pytest.approx(10.0)
    assert metrics.maximum_indoor_temperature_c == pytest.approx(30.0)
    assert metrics.mean_indoor_temperature_c == pytest.approx(20.0)
    assert metrics.indoor_temperature_range_c == pytest.approx(20.0)


def test_variable_timesteps_integrate_degree_hours_correctly():
    # Violations of 8 K over 1 h and 3 K over 3 h -> 8 + 9 = 17 degree-hours.
    metrics = comfort_metrics([10.0, 15.0], BAND, [1.0, 3.0])
    assert metrics.degree_hours_below_comfort == pytest.approx(17.0)


def test_comfort_limits_are_user_configurable():
    narrow = ComfortRange(minimum_comfort_temperature_c=20.0, maximum_comfort_temperature_c=22.0)
    default = comfort_metrics([19.0, 21.0, 23.0], BAND, 1.0)
    strict = comfort_metrics([19.0, 21.0, 23.0], narrow, 1.0)
    assert default.percent_time_comfortable == pytest.approx(100.0)
    assert strict.percent_time_comfortable == pytest.approx(100.0 / 3.0)
    assert strict.degree_hours_below_comfort == pytest.approx(1.0)
    assert strict.degree_hours_above_comfort == pytest.approx(1.0)


def test_comfort_summary_works_on_simulation_records(reference_config, synthetic_weather):
    result = simulate_shelter(reference_config, synthetic_weather)
    summary = comfort_summary(result.records)
    indoor = result.records["indoor_temperature_c"]
    assert summary["minimum_indoor_temperature_c"] == pytest.approx(indoor.min())
    assert summary["maximum_indoor_temperature_c"] == pytest.approx(indoor.max())
    assert summary["mean_indoor_temperature_c"] == pytest.approx(indoor.mean())
    assert summary["percent_time_below_comfort"] == pytest.approx(
        100.0 - summary["percent_time_comfortable"] - summary["percent_time_above_comfort"]
    )
    assert result.summary["degree_hours_below_comfort"] == pytest.approx(
        summary["degree_hours_below_comfort"]
    )


def test_comfort_summary_rejects_bad_input():
    with pytest.raises(ValueError):
        comfort_metrics([], BAND, 1.0)
    with pytest.raises(ValueError):
        comfort_metrics([20.0], BAND, 0.0)
    with pytest.raises(ValueError):
        comfort_metrics([20.0, 21.0], BAND, [1.0])
    with pytest.raises(ValueError):
        comfort_summary(pd.DataFrame({"other": [1.0]}))
    with pytest.raises(ValueError):
        comfort_summary(pd.DataFrame({"indoor_temperature_c": []}))
