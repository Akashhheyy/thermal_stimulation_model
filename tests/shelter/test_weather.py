"""Offline tests for the NASA POWER integration (no live internet access)."""
import json
from datetime import datetime

import pandas as pd
import pytest

from building_hvac_twin.shelter import (
    NasaWeatherError,
    clear_weather_cache,
    fetch_nasa_power_hourly,
    get_nasa_weather_data,
    parse_nasa_power_hourly,
    simulate_shelter,
)


def nasa_payload(
    hours: int = 3,
    start_day: str = "2024-01-01",
    temperature: list[float] | None = None,
    solar: list[float] | None = None,
    wind: list[float] | None = None,
    humidity: list[float] | None = None,
) -> dict:
    """Build a NASA POWER shaped payload for offline tests."""
    day = datetime.strptime(start_day, "%Y-%m-%d")
    temperature = temperature if temperature is not None else [-8.5] * hours
    solar = solar if solar is not None else [0.0, 120.5, 240.0][:hours] + [0.0] * max(0, hours - 3)
    wind = wind if wind is not None else [3.1] * hours
    humidity = humidity if humidity is not None else [41.0] * hours

    def key(offset: int) -> str:
        stamp = day.replace(hour=offset)
        return stamp.strftime("%Y%m%d%H")

    def series(values: list[float]) -> dict:
        return {key(index): value for index, value in enumerate(values)}

    return {
        "properties": {
            "parameter": {
                "T2M": series(temperature),
                "ALLSKY_SFC_SW_DWN": series(solar),
                "WS10M": series(wind),
                "RH2M": series(humidity),
            }
        }
    }


class RecordingTransport:
    """Fake transport returning a canned payload and counting requests."""

    def __init__(self, payload: dict, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout_seconds: float) -> bytes:
        self.calls.append((url, timeout_seconds))
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload).encode()


@pytest.fixture(autouse=True)
def clean_cache():
    clear_weather_cache()
    yield
    clear_weather_cache()


def test_successful_parsing_and_field_conversion():
    frame = parse_nasa_power_hourly(nasa_payload())
    assert list(frame.columns) == [
        "timestamp",
        "outdoor_temperature_c",
        "solar_radiation_w_m2",
        "wind_speed_m_s",
        "relative_humidity_percent",
    ]
    assert len(frame) == 3
    assert frame["outdoor_temperature_c"].tolist() == pytest.approx([-8.5, -8.5, -8.5])
    assert frame["solar_radiation_w_m2"].tolist() == pytest.approx([0.0, 120.5, 240.0])
    assert frame["timestamp"].is_monotonic_increasing
    stamps = frame["timestamp"].tolist()
    assert stamps[0].tz is not None  # NASA hourly is UTC
    assert (stamps[1] - stamps[0]).total_seconds() == 3600.0


def test_missing_required_values_are_dropped_and_counted():
    payload = nasa_payload(hours=3, temperature=[-8.5, -999.0, -7.0], solar=[0.0, 100.0, -999.0])
    frame = parse_nasa_power_hourly(payload)
    # Hour 1 lost temperature and hour 2 lost irradiance: both rows dropped.
    assert len(frame) == 1
    assert frame.attrs["skipped_missing_required_records"] == 2


def test_missing_optional_values_become_nan_not_dropped():
    payload = nasa_payload(hours=2, wind=[3.1, -999.0], humidity=[41.0, -999.0])
    frame = parse_nasa_power_hourly(payload)
    assert len(frame) == 2
    assert frame["wind_speed_m_s"].isna().tolist() == [False, True]
    assert frame["relative_humidity_percent"].isna().tolist() == [False, True]


def test_optional_parameters_not_requested_are_absent():
    frame = parse_nasa_power_hourly(nasa_payload(), parameters=("T2M", "ALLSKY_SFC_SW_DWN"))
    assert "wind_speed_m_s" not in frame.columns
    assert "relative_humidity_percent" not in frame.columns


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError):
        get_nasa_weather_data(95.0, 77.0, "2024-01-01", "2024-01-02")
    with pytest.raises(ValueError):
        get_nasa_weather_data(34.0, -181.0, "2024-01-01", "2024-01-02")
    with pytest.raises(ValueError):
        get_nasa_weather_data("north", 77.0, "2024-01-01", "2024-01-02")


def test_invalid_date_ranges_are_rejected():
    with pytest.raises(ValueError):
        get_nasa_weather_data(34.0, 77.0, "2024-01-05", "2024-01-01")
    with pytest.raises(ValueError):
        get_nasa_weather_data(34.0, 77.0, "January 2024", "2024-01-02")
    with pytest.raises(ValueError):
        fetch_nasa_power_hourly(34.0, 77.0, "2024-13-01", "2024-13-02")


def test_api_failure_is_wrapped_in_nasa_weather_error():
    transport = RecordingTransport({}, error=OSError("network unreachable"))
    with pytest.raises(NasaWeatherError):
        get_nasa_weather_data(34.0, 77.0, "2024-01-01", "2024-01-02", transport=transport)
    assert len(transport.calls) == 1


def test_network_timeout_is_wrapped_in_nasa_weather_error():
    transport = RecordingTransport({}, error=TimeoutError("timed out"))
    with pytest.raises(NasaWeatherError):
        get_nasa_weather_data(34.0, 77.0, "2024-01-01", "2024-01-02", transport=transport)


def test_empty_and_malformed_responses_are_rejected():
    with pytest.raises(NasaWeatherError):
        parse_nasa_power_hourly({})
    with pytest.raises(NasaWeatherError):
        parse_nasa_power_hourly({"properties": {"parameter": {}}})
    missing_required = nasa_payload()
    del missing_required["properties"]["parameter"]["T2M"]
    with pytest.raises(NasaWeatherError):
        parse_nasa_power_hourly(missing_required)
    transport = RecordingTransport({"header": {}})
    with pytest.raises(NasaWeatherError):
        get_nasa_weather_data(34.0, 77.0, "2024-01-01", "2024-01-02", transport=transport)


def test_caching_prevents_repeated_nasa_calls():
    transport = RecordingTransport(nasa_payload())
    first = get_nasa_weather_data(34.16, 77.58, "2024-01-01", "2024-01-01", transport=transport)
    second = get_nasa_weather_data(34.16, 77.58, "2024-01-01", "2024-01-01", transport=transport)
    assert len(transport.calls) == 1, "identical request must be served from cache"
    pd.testing.assert_frame_equal(first, second)
    # Mutating the returned frame must not poison the cache.
    first.loc[0, "outdoor_temperature_c"] = 999.0
    third = get_nasa_weather_data(34.16, 77.58, "2024-01-01", "2024-01-01", transport=transport)
    assert third["outdoor_temperature_c"].iloc[0] != 999.0


def test_cache_distinguishes_location_and_dates():
    transport = RecordingTransport(nasa_payload())
    get_nasa_weather_data(34.16, 77.58, "2024-01-01", "2024-01-01", transport=transport)
    get_nasa_weather_data(28.61, 77.21, "2024-01-01", "2024-01-01", transport=transport)
    get_nasa_weather_data(34.16, 77.58, "2024-01-02", "2024-01-02", transport=transport)
    assert len(transport.calls) == 3
    clear_weather_cache()
    get_nasa_weather_data(34.16, 77.58, "2024-01-01", "2024-01-01", transport=transport)
    assert len(transport.calls) == 4, "cleared cache must trigger a fresh request"


def test_use_cache_false_forces_a_fresh_request():
    transport = RecordingTransport(nasa_payload())
    get_nasa_weather_data(34.16, 77.58, "2024-01-01", "2024-01-01", transport=transport)
    get_nasa_weather_data(
        34.16, 77.58, "2024-01-01", "2024-01-01", transport=transport, use_cache=False
    )
    assert len(transport.calls) == 2


def test_request_url_contains_location_dates_and_parameters():
    transport = RecordingTransport(nasa_payload())
    fetch_nasa_power_hourly(
        34.16,
        77.58,
        "2024-01-01",
        "2024-01-03",
        parameters=("T2M", "ALLSKY_SFC_SW_DWN"),
        transport=transport,
    )
    url, timeout = transport.calls[0]
    assert url.startswith("https://power.larc.nasa.gov/api/temporal/hourly/point")
    assert "latitude=34.16" in url
    assert "longitude=77.58" in url
    assert "start=20240101" in url
    assert "end=20240103" in url
    assert "T2M" in url and "ALLSKY_SFC_SW_DWN" in url
    assert "time-standard=UTC" in url
    assert timeout > 0


def test_converted_data_drives_the_thermal_simulation(reference_config):
    frame = parse_nasa_power_hourly(nasa_payload(hours=6))
    result = simulate_shelter(reference_config, frame)
    assert len(result.records) == 6
    assert result.records["outdoor_temperature_c"].tolist() == pytest.approx([-8.5] * 6)
    assert result.records["indoor_temperature_c"].nunique() >= 1
    # The engine never needed to know anything about NASA POWER.
    assert "T2M" not in result.records.columns


def test_non_json_content_is_rejected():
    def binary_transport(url: str, timeout_seconds: float) -> bytes:
        return b"<html>gateway error</html>"

    with pytest.raises(NasaWeatherError):
        get_nasa_weather_data(34.0, 77.0, "2024-01-01", "2024-01-02", transport=binary_transport)

