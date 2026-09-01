"""NASA POWER historical weather integration layer.

ISOLATION GUARANTEE: the thermal engine never imports this module.  The only
contract between the two is the DataFrame produced here, which carries the
columns the simulation already accepts (``timestamp``,
``outdoor_temperature_c``, ``solar_radiation_w_m2``, and optionally
``wind_speed_m_s`` and ``relative_humidity_percent``).  The thermal model
therefore remains fully independent of the external API, and every test for
this layer runs offline against a mocked transport.

External source: NASA POWER (Prediction Of Worldwide Energy Resources),
hourly point API: https://power.larc.nasa.gov/ .  Parameters used:

    T2M               outdoor air temperature at 2 m (C)
    ALLSKY_SFC_SW_DWN all-sky surface shortwave downward irradiance (W/m2)
    WS10M             wind speed at 10 m (m/s, optional for the model)
    RH2M              relative humidity at 2 m (percent, optional)

Field mapping into the internal format:

    T2M               -> outdoor_temperature_c
    ALLSKY_SFC_SW_DWN -> solar_radiation_w_m2
    WS10M             -> wind_speed_m_s
    RH2M              -> relative_humidity_percent

Missing-value policy: NASA reports missing points as -999.  Rows whose
REQUIRED values (temperature or irradiance) are missing or nonfinite are
excluded and counted; missing OPTIONAL values become NaN so the thermal
simulation (which only uses the required columns) still runs.  No values
are silently invented.  Hours are returned in UTC because the request pins
``time-standard=UTC``.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Callable, Iterable, Sequence

import pandas as pd

__all__ = [
    "NasaWeatherError",
    "NASA_BASE_URL",
    "DEFAULT_NASA_PARAMETERS",
    "REQUIRED_NASA_PARAMETERS",
    "PARAMETER_MAP",
    "NASA_MISSING_SENTINEL",
    "fetch_nasa_power_hourly",
    "parse_nasa_power_hourly",
    "get_nasa_weather_data",
    "clear_weather_cache",
    "weather_cache_info",
]

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
DEFAULT_NASA_PARAMETERS = ("T2M", "ALLSKY_SFC_SW_DWN", "WS10M", "RH2M")
REQUIRED_NASA_PARAMETERS = ("T2M", "ALLSKY_SFC_SW_DWN")
PARAMETER_MAP = {
    "T2M": "outdoor_temperature_c",
    "ALLSKY_SFC_SW_DWN": "solar_radiation_w_m2",
    "WS10M": "wind_speed_m_s",
    "RH2M": "relative_humidity_percent",
}
NASA_MISSING_SENTINEL = -999.0

# Transport contract: callable(url: str, timeout_seconds: float) -> bytes body.
Transport = Callable[[str, float], bytes]


class NasaWeatherError(RuntimeError):
    """Raised when the NASA POWER request, response, or parsing fails."""


def _default_transport(url: str, timeout_seconds: float) -> bytes:
    """Stdlib HTTP transport; replaceable in tests (no live calls there)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.read()
    except NasaWeatherError:
        raise
    except Exception as error:  # urllib.error.HTTPError, URLError, timeout, ...
        raise NasaWeatherError(f"NASA POWER request failed: {error}") from error


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as error:
        raise ValueError(f"latitude and longitude must be numbers, got {latitude!r}, {longitude!r}") from error
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude must be within [-90, 90], got {lat}")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude must be within [-180, 180], got {lon}")
    return lat, lon


def _to_nasa_date(value: str | date | datetime, label: str) -> str:
    """Accept YYYY-MM-DD strings (or date/datetime) and return YYYYMMDD."""
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y%m%d")
            except ValueError:
                continue
        raise ValueError(f"{label} must look like YYYY-MM-DD, got {value!r}")
    raise ValueError(f"{label} must be a date or YYYY-MM-DD string, got {value!r}")


def _validate_date_range(start_date, end_date) -> tuple[str, str]:
    start = _to_nasa_date(start_date, "start_date")
    end = _to_nasa_date(end_date, "end_date")
    if start > end:
        raise ValueError(f"start_date ({start}) must not be after end_date ({end})")
    return start, end


def _validate_parameters(parameters: Sequence[str] | None) -> tuple[str, ...]:
    chosen = tuple(parameters) if parameters else DEFAULT_NASA_PARAMETERS
    if not chosen:
        raise ValueError("parameters must not be empty")
    unknown = [name for name in chosen if name not in PARAMETER_MAP]
    if unknown:
        raise ValueError(f"unsupported NASA parameters: {unknown}; supported: {sorted(PARAMETER_MAP)}")
    for required in REQUIRED_NASA_PARAMETERS:
        if required not in chosen:
            raise ValueError(
                f"parameter {required} is required by the thermal model and must be requested"
            )
    return chosen


def _nasa_timestamp(key: str):
    """NASA hourly keys look like YYYYMMDDHH and are UTC."""
    text = str(key).strip()
    try:
        return pd.Timestamp(datetime.strptime(text, "%Y%m%d%H"), tz="UTC")
    except ValueError as error:
        raise NasaWeatherError(f"unexpected NASA hourly timestamp {key!r}") from error


def fetch_nasa_power_hourly(
    latitude: float,
    longitude: float,
    start_date,
    end_date,
    parameters: Sequence[str] | None = None,
    community: str = "RE",
    timeout_seconds: float = 30.0,
    transport: Transport | None = None,
) -> dict:
    """Request hourly NASA POWER data and return the raw JSON payload.

    ``transport`` is injectable for offline testing; the default performs a
    real HTTP GET against the NASA POWER hourly point endpoint.
    """
    lat, lon = _validate_coordinates(latitude, longitude)
    start, end = _validate_date_range(start_date, end_date)
    chosen = _validate_parameters(parameters)
    if community not in ("RE", "AG", "SB"):
        raise ValueError(f"community must be one of RE, AG, SB, got {community!r}")

    query = urllib.parse.urlencode(
        {
            "parameters": ",".join(chosen),
            "community": community,
            "latitude": lat,
            "longitude": lon,
            "start": start,
            "end": end,
            "time-standard": "UTC",
        }
    )
    url = f"{NASA_BASE_URL}?{query}"
    moved: Transport = transport if transport is not None else _default_transport
    try:
        body = moved(url, timeout_seconds)
    except NasaWeatherError:
        raise
    except Exception as error:
        # Covers HTTPError, URLError, timeouts, and injected failing transports.
        raise NasaWeatherError(f"NASA POWER request failed: {error}") from error
    try:
        payload = json.loads(body)
    except (TypeError, ValueError) as error:
        raise NasaWeatherError(f"NASA POWER returned non-JSON content: {error}") from error
    if not isinstance(payload, dict):
        raise NasaWeatherError("NASA POWER response is not a JSON object")
    return payload



def parse_nasa_power_hourly(payload: dict, parameters: Sequence[str] | None = None) -> pd.DataFrame:
    """Convert a NASA POWER payload into the internal weather DataFrame.

    Required-value rows (temperature or irradiance) that are missing or
    nonfinite are dropped and counted; optional-value gaps become NaN.
    """
    chosen = _validate_parameters(parameters)
    if not isinstance(payload, dict):
        raise NasaWeatherError("NASA POWER payload must be a JSON object")
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        raise NasaWeatherError("NASA POWER response has no 'properties' section (empty response?)")
    parameter_block = properties.get("parameter")
    if not isinstance(parameter_block, dict) or not parameter_block:
        raise NasaWeatherError("NASA POWER response contains no parameter data (empty response?)")

    for name in REQUIRED_NASA_PARAMETERS:
        if name not in parameter_block:
            raise NasaWeatherError(f"NASA POWER response is missing required parameter {name}")

    series: dict[str, dict] = {}
    for name in chosen:
        raw = parameter_block.get(name)
        if raw is None:
            if name in REQUIRED_NASA_PARAMETERS:
                raise NasaWeatherError(f"NASA POWER response is missing required parameter {name}")
            series[name] = {}
            continue
        if not isinstance(raw, dict):
            raise NasaWeatherError(f"NASA POWER parameter {name} has unexpected shape")
        series[name] = raw

    all_keys: set[str] = set()
    for name in chosen:
        all_keys.update(series[name].keys())

    rows: list[dict] = []
    skipped_missing_required = 0
    for key in sorted(all_keys):
        stamp = _nasa_timestamp(key)
        values: dict[str, float] = {"timestamp": stamp}
        required_missing = False
        for name in chosen:
            number = pd.to_numeric(series[name].get(key), errors="coerce")
            column = PARAMETER_MAP[name]
            if pd.isna(number) or float(number) <= NASA_MISSING_SENTINEL + 1e-9:
                if name in REQUIRED_NASA_PARAMETERS:
                    required_missing = True
                else:
                    values[column] = float("nan")
                continue
            values[column] = float(number)
        if required_missing or values["outdoor_temperature_c"] <= -273.15:
            skipped_missing_required += 1
            continue
        rows.append(values)

    if not rows:
        raise NasaWeatherError("NASA POWER returned no usable records (empty response?)")

    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    columns = ["timestamp"] + [PARAMETER_MAP[name] for name in chosen]
    frame = frame[columns]
    frame.attrs["skipped_missing_required_records"] = skipped_missing_required
    return frame


_CACHE: dict[tuple, pd.DataFrame] = {}


def clear_weather_cache() -> None:
    """Drop every cached NASA POWER response."""
    _CACHE.clear()


def weather_cache_info() -> dict:
    """Number of cached responses and their keys (for diagnostics)."""
    return {"entries": len(_CACHE), "keys": sorted(str(key) for key in _CACHE)}


def get_nasa_weather_data(
    latitude: float,
    longitude: float,
    start_date,
    end_date,
    parameters: Sequence[str] | None = None,
    community: str = "RE",
    timeout_seconds: float = 30.0,
    transport: Transport | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Validated, cached, converted NASA POWER weather for one location.

    Repeated calls with exactly the same location, dates, parameters, and
    community reuse the cached DataFrame and do not call NASA again.  The
    returned DataFrame is a copy so callers cannot mutate the cache.
    """
    lat, lon = _validate_coordinates(latitude, longitude)
    start, end = _validate_date_range(start_date, end_date)
    chosen = _validate_parameters(parameters)

    cache_key = (lat, lon, start, end, chosen, community)
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    payload = fetch_nasa_power_hourly(
        lat,
        lon,
        start,
        end,
        parameters=chosen,
        community=community,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    frame = parse_nasa_power_hourly(payload, chosen)
    if use_cache:
        _CACHE[cache_key] = frame.copy()
    return frame

