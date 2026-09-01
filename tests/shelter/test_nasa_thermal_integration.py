"""End-to-end test: NASA POWER weather (mocked) into the thermal engine.

No live internet access is used.  The NASA response is a deterministic
72-hour payload delivered through a fake transport, exactly the shape the
real API returns.
"""
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from building_hvac_twin.shelter import (
    clear_weather_cache,
    get_nasa_weather_data,
    simulate_shelter,
)

HOURS = 72
START = datetime(2024, 1, 1)

REQUIRED_SIMULATION_COLUMNS = (
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


def leh_series(hour_offset: int) -> tuple[float, float]:
    """Deterministic diurnal outdoor temperature and solar radiation."""
    hour_of_day = hour_offset % 24
    outdoor = -12.0 + 6.0 * np.sin(2.0 * np.pi * (hour_of_day - 8) / 24.0)
    solar = max(0.0, 500.0 * np.sin(np.pi * (hour_of_day - 7) / 12.0))
    return round(outdoor, 2), round(solar, 2)


def nasa_payload() -> dict:
    def key(offset: int) -> str:
        return (START + timedelta(hours=offset)).strftime("%Y%m%d%H")

    temperatures, solar_values, wind, humidity = {}, {}, {}, {}
    for offset in range(HOURS):
        outdoor, solar = leh_series(offset)
        temperatures[key(offset)] = outdoor
        solar_values[key(offset)] = solar
        wind[key(offset)] = 1.5
        humidity[key(offset)] = 45.0
    return {
        "properties": {
            "parameter": {
                "T2M": temperatures,
                "ALLSKY_SFC_SW_DWN": solar_values,
                "WS10M": wind,
                "RH2M": humidity,
            }
        }
    }


def equivalent_manual_frame() -> pd.DataFrame:
    """The same weather built by hand (no NASA code involved at all)."""
    stamps = pd.date_range(START, periods=HOURS, freq="h", tz="UTC")
    pairs = [leh_series(offset) for offset in range(HOURS)]
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "outdoor_temperature_c": [pair[0] for pair in pairs],
            "solar_radiation_w_m2": [pair[1] for pair in pairs],
        }
    )


class FakeNasaTransport:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def __call__(self, url: str, timeout_seconds: float) -> bytes:
        self.calls += 1
        return json.dumps(self.payload).encode()


@pytest.fixture(autouse=True)
def clean_cache():
    clear_weather_cache()
    yield
    clear_weather_cache()


@pytest.fixture
def nasa_weather_frame() -> pd.DataFrame:
    transport = FakeNasaTransport(nasa_payload())
    frame = get_nasa_weather_data(
        34.1645, 77.5789, "2024-01-01", "2024-01-03", transport=transport
    )
    assert transport.calls == 1
    return frame


def test_nasa_fields_are_converted_correctly(nasa_weather_frame):
    assert list(nasa_weather_frame.columns[:3]) == [
        "timestamp",
        "outdoor_temperature_c",
        "solar_radiation_w_m2",
    ]
    assert len(nasa_weather_frame) == HOURS
    assert nasa_weather_frame["solar_radiation_w_m2"].min() >= 0.0
    expected_first, _ = leh_series(0)
    assert nasa_weather_frame["outdoor_temperature_c"].iloc[0] == pytest.approx(expected_first)


def test_nasa_frame_drives_the_thermal_simulation(nasa_weather_frame, reference_config):
    result = simulate_shelter(reference_config, nasa_weather_frame)
    records = result.records
    assert len(records) == len(nasa_weather_frame) == HOURS
    missing = [column for column in REQUIRED_SIMULATION_COLUMNS if column not in records.columns]
    assert not missing, f"missing simulation columns: {missing}"
    assert records["outdoor_temperature_c"].tolist() == pytest.approx(
        nasa_weather_frame["outdoor_temperature_c"].tolist()
    )


def test_all_thermal_outputs_are_finite(nasa_weather_frame, reference_config):
    records = simulate_shelter(reference_config, nasa_weather_frame).records
    for column in (
        "indoor_temperature_c",
        "total_heat_gain_w",
        "total_heat_loss_w",
        "net_heat_balance_w",
        "solar_heat_gain_w",
        "thermal_mass_heat_flow_w",
    ):
        values = records[column].to_numpy()
        assert np.isfinite(values).all(), f"{column} contains non-finite values"
    assert (records["total_heat_loss_w"] >= 0.0).all()
    assert (records["solar_heat_gain_w"] >= 0.0).all()


def test_integration_does_not_alter_thermal_model_behaviour(
    nasa_weather_frame, reference_config
):
    """NASA path and hand-built path must produce identical simulations."""
    from_nasa = simulate_shelter(reference_config, nasa_weather_frame)
    by_hand = simulate_shelter(reference_config, equivalent_manual_frame())
    pd.testing.assert_frame_equal(from_nasa.records, by_hand.records)
