# NASA POWER Weather Data Integration

## What NASA POWER is here

NASA POWER (Prediction Of Worldwide Energy Resources) is the external
historical weather data source for the shelter model.  It provides global,
satellite-derived hourly weather through a public HTTP API
(<https://power.larc.nasa.gov/>).  The integration lives in
`src/building_hvac_twin/shelter/weather.py` and is the ONLY module that
talks to NASA.

## Parameters we request

| NASA parameter | Meaning | Unit |
|---|---|---|
| `T2M` | outdoor air temperature at 2 m | degrees C |
| `ALLSKY_SFC_SW_DWN` | all-sky surface shortwave downward irradiance | W/m2 |
| `WS10M` | wind speed at 10 m | m/s |
| `RH2M` | relative humidity at 2 m | percent |

`T2M` and `ALLSKY_SFC_SW_DWN` are required by the thermal model; `WS10M`
and `RH2M` are carried for future work (infiltration, comfort).

## Mapping into the thermal model

| NASA field | Internal column |
|---|---|
| `T2M` | `outdoor_temperature_c` |
| `ALLSKY_SFC_SW_DWN` | `solar_radiation_w_m2` |
| `WS10M` | `wind_speed_m_s` |
| `RH2M` | `relative_humidity_percent` |

The returned pandas DataFrame has a `timestamp` column (hourly, UTC,
because the request pins `time-standard=UTC`) and plugs directly into
`building_hvac_twin.shelter.simulate_shelter`, which already accepts
DataFrames with `timestamp`, `outdoor_temperature_c`, and
`solar_radiation_w_m2`.

## Isolation guarantee

The thermal engine never imports `weather.py` and contains no NASA, HTTP,
or URL logic.  The only contract between the two is the DataFrame columns
above.  Tests for the weather layer run fully offline against a mocked
transport function, so no test depends on live internet access.

## Behaviour details

- Coordinates: any latitude in [-90, 90] and longitude in [-180, 180];
  nothing is hard-coded to Leh or any other place.
- Dates: `YYYY-MM-DD` strings (or `date`/`datetime` objects); `start_date`
  after `end_date` is rejected.
- Missing values: NASA marks missing points with -999.  Records whose
  temperature or irradiance are missing are dropped and counted
  (see `frame.attrs["skipped_missing_required_records"]`); missing optional
  values become NaN.  Nothing is silently invented.
- Failures: API errors, network timeouts, non-JSON content, empty
  responses, and unexpected shapes raise `NasaWeatherError`.  Invalid
  coordinates, dates, or parameters raise `ValueError`.
- Caching: identical requests (same rounded coordinates, dates, parameters,
  community) are served from an in-memory cache; `clear_weather_cache()`
  resets it and `use_cache=False` bypasses it.

## Example

See `examples/shelter/nasa_weather_demo.py` for a live request for Leh
(3 days, hourly) printing record counts and temperature and solar
statistics.

## Limitations

- NASA POWER values are satellite/reanalysis estimates, not ground-station
  measurements; treat them with corresponding caution.
- The in-memory cache does not persist across processes.
- Hourly NASA data can lag recent dates; very recent periods may be empty.
- The thermal model currently uses only the required columns; wind and
  humidity are stored but unused.
