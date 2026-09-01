"""Physical sanity tests for the lumped-capacitance time-step simulation."""
from dataclasses import replace

import pandas as pd
import pytest

from building_hvac_twin.shelter import (
    EnvelopeAssembly,
    Layer,
    SimulationResult,
    ShelterConfig,
    simulate_shelter,
)

REQUIRED_FIELDS = (
    "timestamp",
    "outdoor_temperature_c",
    "solar_radiation_w_m2",
    "indoor_temperature_c",
    "wall_heat_transfer_w",
    "roof_heat_transfer_w",
    "floor_heat_transfer_w",
    "window_heat_transfer_w",
    "door_heat_transfer_w",
    "solar_heat_gain_w",
    "internal_heat_gain_w",
    "thermal_mass_heat_flow_w",
    "total_heat_gain_w",
    "total_heat_loss_w",
    "net_heat_balance_w",
)


def insulated_copy(config: ShelterConfig, extra_layer: Layer) -> ShelterConfig:
    wall = EnvelopeAssembly(
        config.wall_assembly.name,
        [*config.wall_assembly.layers, extra_layer],
        config.wall_assembly.surface_resistance_inner_m2k_w,
        config.wall_assembly.surface_resistance_outer_m2k_w,
    )
    return replace(config, wall_assembly=wall, name=config.name + "-insulated")


def test_one_output_row_per_weather_timestep(reference_config, synthetic_weather):
    result = simulate_shelter(reference_config, synthetic_weather)
    assert isinstance(result, SimulationResult)
    assert len(result.records) == len(synthetic_weather) == 24


def test_output_contains_all_required_fields(reference_config, synthetic_weather):
    result = simulate_shelter(reference_config, synthetic_weather)
    missing = [field for field in REQUIRED_FIELDS if field not in result.records.columns]
    assert not missing, f"missing output fields: {missing}"


def test_indoor_temperature_changes_over_time(reference_config, synthetic_weather):
    result = simulate_shelter(reference_config, synthetic_weather)
    indoor = result.records["indoor_temperature_c"]
    assert indoor.nunique() > 1
    assert indoor.std(ddof=0) > 0.1


def test_colder_outdoor_increases_heat_loss(reference_config, constant_weather):
    cold = simulate_shelter(reference_config, constant_weather)
    milder = simulate_shelter(reference_config, constant_weather.assign(outdoor_temperature_c=-5.0))
    assert cold.records["total_heat_loss_w"].mean() > milder.records["total_heat_loss_w"].mean()
    assert cold.records["indoor_temperature_c"].iloc[-1] < milder.records["indoor_temperature_c"].iloc[-1]


def test_higher_insulation_reduces_conductive_heat_loss(reference_config, constant_weather):
    extra = Layer(0.1, 0.035, 30.0, 1400.0, "insulation")
    base = simulate_shelter(reference_config, constant_weather)
    improved = simulate_shelter(insulated_copy(reference_config, extra), constant_weather)
    assert improved.records["total_heat_loss_w"].mean() < base.records["total_heat_loss_w"].mean()
    assert improved.records["indoor_temperature_c"].iloc[-1] > base.records["indoor_temperature_c"].iloc[-1]


def test_higher_solar_radiation_increases_solar_gain(reference_config, constant_weather):
    sunny = simulate_shelter(reference_config, constant_weather.assign(solar_radiation_w_m2=600.0))
    shaded = simulate_shelter(reference_config, constant_weather)
    assert sunny.records["solar_heat_gain_w"].max() > 0.0
    assert shaded.records["solar_heat_gain_w"].max() == pytest.approx(0.0)
    assert sunny.records["indoor_temperature_c"].mean() > shaded.records["indoor_temperature_c"].mean()


def test_zero_solar_radiation_produces_zero_direct_solar_gain(reference_config, constant_weather):
    result = simulate_shelter(reference_config, constant_weather)
    assert (result.records["solar_heat_gain_w"] == 0.0).all()


def test_higher_thermal_mass_slows_indoor_response(reference_config, synthetic_weather):
    light = simulate_shelter(replace(reference_config, thermal_mass=None), synthetic_weather)
    heavy = simulate_shelter(reference_config, synthetic_weather)
    light_steps = light.records["indoor_temperature_c"].diff().abs().max()
    heavy_steps = heavy.records["indoor_temperature_c"].diff().abs().max()
    assert heavy_steps < light_steps, "more thermal mass must damp the step-to-step swings"
    assert heavy.records["thermal_mass_heat_flow_w"].abs().max() > 0.0


def test_thermal_mass_absorbs_by_day_and_releases_by_night(reference_config, synthetic_weather):
    result = simulate_shelter(reference_config, synthetic_weather)
    daytime = result.records.loc[result.records["solar_heat_gain_w"] > 0.0, "thermal_mass_heat_flow_w"]
    nighttime = result.records.loc[result.records["solar_heat_gain_w"] == 0.0, "thermal_mass_heat_flow_w"]
    assert (daytime > 0.0).any(), "mass should absorb excess heat while the sun shines"
    assert (nighttime < 0.0).any(), "mass should release stored heat after dark"


def test_variable_timesteps_are_honoured(reference_config):
    timestamps = pd.to_datetime(["2025-01-15 00:00", "2025-01-15 01:00", "2025-01-15 03:00"])
    weather = pd.DataFrame(
        {
            "timestamp": timestamps,
            "outdoor_temperature_c": [-15.0, -15.0, -15.0],
            "solar_radiation_w_m2": [0.0, 0.0, 0.0],
        }
    )
    result = simulate_shelter(reference_config, weather)
    # First interval reuses the median of the others; the rest are exact.
    assert result.records["timestep_hours"].tolist() == [1.5, 1.0, 2.0]


def test_longer_timestep_moves_temperature_further(reference_config):
    def first_step_change(hours: float) -> float:
        second = pd.Timestamp("2025-01-15 00:00") + pd.Timedelta(hours=hours)
        weather = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-01-15 00:00", second]),
                "outdoor_temperature_c": [-15.0, -15.0],
                "solar_radiation_w_m2": [0.0, 0.0],
            }
        )
        records = simulate_shelter(reference_config, weather).records
        start = reference_config.initial_indoor_temperature_c
        return abs(records["indoor_temperature_c"].iloc[0] - start)

    one_hour = first_step_change(1)
    two_hours = first_step_change(2)
    assert two_hours == pytest.approx(2.0 * one_hour, rel=1e-6)


def test_initial_temperature_can_be_overridden(reference_config, constant_weather):
    default = simulate_shelter(reference_config, constant_weather)
    override = simulate_shelter(
        reference_config, constant_weather, initial_indoor_temperature_c=25.0
    )
    assert (
        override.records["indoor_temperature_c"].iloc[0]
        != default.records["indoor_temperature_c"].iloc[0]
    )
    # A warmer start loses more heat in the same first step.
    assert override.records["total_heat_loss_w"].iloc[0] > default.records["total_heat_loss_w"].iloc[0]


def test_energy_balance_identity_holds_every_step(reference_config, synthetic_weather):
    result = simulate_shelter(reference_config, synthetic_weather)
    residual = (
        result.records["net_heat_balance_w"]
        - (result.records["total_heat_gain_w"] - result.records["total_heat_loss_w"])
    ).abs()
    assert (residual < 1e-6).all()


def test_invalid_weather_inputs_are_rejected(reference_config):
    good = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-15", periods=3, freq="h"),
            "outdoor_temperature_c": [-15.0, -15.0, -15.0],
            "solar_radiation_w_m2": [0.0, 0.0, 0.0],
        }
    )
    with pytest.raises(ValueError):
        simulate_shelter(reference_config, good.drop(columns=["solar_radiation_w_m2"]))
    with pytest.raises(ValueError):
        simulate_shelter(reference_config, good.iloc[::-1])  # not strictly increasing
    with pytest.raises(ValueError):
        simulate_shelter(reference_config, good.assign(solar_radiation_w_m2=[0.0, None, 0.0]))
    with pytest.raises(ValueError):
        simulate_shelter(reference_config, good.iloc[:0])  # empty

