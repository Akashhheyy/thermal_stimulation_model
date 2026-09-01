"""NASA POWER historical weather demo.

This demo performs a LIVE request to the NASA POWER hourly API for Leh,
Ladakh (coordinates passed as arguments here; the library function itself
never hard-codes any location).  It prints record counts, first and last
records, and temperature and solar statistics.

Everything printed below is NASA POWER HISTORICAL WEATHER DATA, not
synthetic data and not a local measurement campaign.

Run from the repository root (internet access required):

    python examples/shelter/nasa_weather_demo.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from building_hvac_twin.shelter import (  # noqa: E402
    NasaWeatherError,
    clear_weather_cache,
    get_nasa_weather_data,
)


def main() -> int:
    latitude = 34.1645   # Leh, Ladakh (demo coordinates only)
    longitude = 77.5789
    start_date = "2024-01-01"
    end_date = "2024-01-03"  # 3 days, hourly -> up to 72 records

    print("=" * 78)
    print("NASA POWER HISTORICAL WEATHER DATA")
    print(f"Location: lat {latitude}, lon {longitude} (user-supplied, any location works)")
    print(f"Period: {start_date} to {end_date} (hourly, UTC)")
    print("Parameters: T2M, ALLSKY_SFC_SW_DWN, WS10M, RH2M")
    print("=" * 78)

    clear_weather_cache()
    try:
        weather = get_nasa_weather_data(latitude, longitude, start_date, end_date)
    except NasaWeatherError as error:
        print(f"NASA POWER request failed: {error}")
        print("Check internet access and try again; the thermal model itself")
        print("does not depend on this API and keeps working with other inputs.")
        return 1

    skipped = weather.attrs.get("skipped_missing_required_records", 0)
    temperatures = weather["outdoor_temperature_c"]
    solar = weather["solar_radiation_w_m2"]

    print(f"\nNumber of records: {len(weather)}"
          f" (skipped for missing NASA values: {skipped})")
    print("\nFirst 5 records:")
    print(weather.head(5).to_string(index=False))
    print("\nLast 5 records:")
    print(weather.tail(5).to_string(index=False))
    print("\nTemperature summary (T2M, degrees C):")
    print(f"  minimum: {temperatures.min():.2f}")
    print(f"  maximum: {temperatures.max():.2f}")
    print(f"  average: {temperatures.mean():.2f}")
    print("\nSolar summary (ALLSKY_SFC_SW_DWN, W/m2):")
    print(f"  total (sum of hourly values): {solar.sum():.1f}")
    print(f"  average: {solar.mean():.2f}")
    print(f"  maximum hourly: {solar.max():.1f}")
    print("\nThis DataFrame feeds building_hvac_twin.shelter.simulate_shelter")
    print("directly; the thermal model stays independent of the NASA API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
