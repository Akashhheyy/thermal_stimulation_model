"""Transparent lumped thermal-capacitance time-step simulation.

Governing equation (single zone, fully mixed air, passive only):

    C * dT_in/dt = Q_wall + Q_roof + Q_floor + Q_window + Q_door
                   + Q_solar + Q_internal

with the effective capacitance

    C = C_air + C_mass
    C_air = air_capacity_per_volume_j_m3k * shelter volume (m3)
    C_mass = thermal_mass_kg * specific_heat_j_kgk (0 when no mass is given)

Discretisation is explicit (forward) Euler using the previous timestep's
indoor temperature:

    T_in[i] = T_in[i-1] + dt_s * Q_net[i] / C

Numerical stability: explicit Euler is only monotonic when the interval is
shorter than the zone time constant (C_total / UA_total).  Lightweight
shelters with little thermal mass can violate this at hourly weather steps,
which would produce nonphysical ringing instead of temperatures.  The
simulation therefore divides any offending interval into equal internal
substeps (see MAX_STABILITY_FRACTION); already-stable intervals run exactly
one substep and are numerically identical to the plain update above.  The
reported heat flows are the instantaneous rates at the start of each
weather interval.

Sign convention: every heat flow is positive when it adds heat to the
shelter air and negative when it removes heat.

Thermal mass is fully coupled to the zone air.  During a step the mass
stores ``C_mass * (T_in[i] - T_in[i-1])`` joules, and
``thermal_mass_heat_flow_w`` reports that storage rate: positive while the
mass absorbs excess heat, negative while it releases stored heat.  A larger
mass therefore slows the indoor response; it never acts as a fixed offset.

Stated assumptions: no ventilation or infiltration, no long-wave exchange,
no latent effects, window solar gain uses the given SHGC with the supplied
radiation already incident on the opening plane, and no HVAC.  Timestep
intervals come from the weather timestamps, so variable intervals work; the
first record reuses the median interval of the series (1 hour if there is
only one record), matching the original Vicena reference model.
"""
from __future__ import annotations

from math import ceil
from typing import Iterable

import pandas as pd

from .geometry import build_geometry
from .models import DataCategory, SimulationResult, WeatherRecord
from .solar import solar_heat_gain_w
from .thermal import component_transfers, total_conductive_u_a_w_k
from .thermal_mass import ThermalMassState

__all__ = ["simulate_shelter", "weather_frame", "MAX_STABILITY_FRACTION"]

# Numerical stability guard for the explicit update.  When the weather
# interval is longer than MAX_STABILITY_FRACTION of the zone time constant
# (C_total / UA_total), the interval is internally divided into equal
# substeps so each substep satisfies
#     dt_sub * UA_total / C_total <= MAX_STABILITY_FRACTION.
# Intervals that already satisfy the guard (the common case) run exactly one
# substep, so their results are bit-for-bit identical to the plain update.
MAX_STABILITY_FRACTION = 0.5

REQUIRED_WEATHER_COLUMNS = ("timestamp", "outdoor_temperature_c", "solar_radiation_w_m2")

OUTPUT_COLUMNS = (
    "timestamp",
    "timestep_hours",
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
    "thermal_mass_temperature_c",
    "total_heat_gain_w",
    "total_heat_loss_w",
    "net_heat_balance_w",
)


def weather_frame(weather: Iterable[WeatherRecord] | pd.DataFrame) -> pd.DataFrame:
    """Normalise weather input to a validated DataFrame.

    Accepts a sequence of :class:`WeatherRecord` or a DataFrame with at least
    ``timestamp``, ``outdoor_temperature_c``, and ``solar_radiation_w_m2``.
    Raises ``ValueError`` on missing columns, non-numeric values, or
    timestamps that are not strictly increasing.
    """
    if isinstance(weather, pd.DataFrame):
        frame = weather.copy().reset_index(drop=True)
    else:
        rows = []
        for record in weather:
            if not isinstance(record, WeatherRecord):
                raise ValueError("weather must contain WeatherRecord items or a DataFrame")
            rows.append(
                {
                    "timestamp": record.timestamp,
                    "outdoor_temperature_c": record.outdoor_temperature_c,
                    "solar_radiation_w_m2": record.solar_radiation_w_m2,
                    "wind_speed_m_s": record.wind_speed_m_s,
                    "relative_humidity_percent": record.relative_humidity_percent,
                }
            )
        frame = pd.DataFrame(rows)

    missing = [column for column in REQUIRED_WEATHER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Weather input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Weather input contains no records")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    for column in ("outdoor_temperature_c", "solar_radiation_w_m2"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Weather column {column!r} contains non-numeric values")
    if (frame["solar_radiation_w_m2"] < 0).any():
        raise ValueError("solar_radiation_w_m2 must be nonnegative")

    intervals = frame["timestamp"].diff().dt.total_seconds()
    if len(frame) > 1 and (intervals.iloc[1:] <= 0).any():
        raise ValueError("weather timestamps must be strictly increasing")
    return frame


def simulate_shelter(
    config,
    weather: Iterable[WeatherRecord] | pd.DataFrame,
    initial_indoor_temperature_c: float | None = None,
    weather_category: DataCategory = DataCategory.SYNTHETIC,
) -> SimulationResult:
    """Run the passive lumped-capacitance simulation over a weather series.

    ``initial_indoor_temperature_c`` overrides the value on the config when
    given; otherwise the config value is used (its documented default is
    10 C when the caller leaves that at the default too).
    """
    from .models import ShelterConfig

    if not isinstance(config, ShelterConfig):
        raise ValueError("simulate_shelter expects a ShelterConfig")

    frame = weather_frame(weather)
    built = build_geometry(config.geometry, config.openings)

    air_capacity_j_k = config.air_capacity_per_volume_j_m3k * built.volume_m3
    mass_state = (
        ThermalMassState.from_thermal_mass(config.thermal_mass)
        if config.thermal_mass is not None
        else None
    )
    mass_capacity_j_k = mass_state.heat_capacity_j_k if mass_state else 0.0
    total_capacity_j_k = air_capacity_j_k + mass_capacity_j_k
    conductance_w_k = total_conductive_u_a_w_k(
        built,
        config.wall_assembly,
        config.roof_assembly,
        config.floor_assembly,
        config.openings,
    )

    start_temperature = (
        float(initial_indoor_temperature_c)
        if initial_indoor_temperature_c is not None
        else float(config.initial_indoor_temperature_c)
    )

    intervals_hours = frame["timestamp"].diff().dt.total_seconds().div(3600)
    if len(frame) > 1:
        intervals_hours.iloc[0] = float(intervals_hours.iloc[1:].median())
    else:
        intervals_hours.iloc[0] = 1.0

    indoor_temperature_c = start_temperature
    rows: list[dict[str, float]] = []
    for index in range(len(frame)):
        outdoor = float(frame["outdoor_temperature_c"].iloc[index])
        radiation = float(frame["solar_radiation_w_m2"].iloc[index])
        dt_hours = float(intervals_hours.iloc[index])
        dt_seconds = dt_hours * 3600.0

        flows = component_transfers(
            built,
            config.wall_assembly,
            config.roof_assembly,
            config.floor_assembly,
            config.openings,
            outdoor,
            indoor_temperature_c,
            config.ground_temperature_c,
        )
        solar = solar_heat_gain_w(
            radiation,
            config.openings.window_area_m2,
            config.openings.window_solar_heat_gain_coefficient,
        )
        internal = config.internal_heat_sources.total_heat_w
        net = flows.total_w + solar + internal

        # Stability guard: subdivide the interval when the zone time constant
        # is shorter than the weather interval (see MAX_STABILITY_FRACTION).
        # The reported flows are the instantaneous rates at the interval start.
        substep_count = max(
            1,
            ceil(dt_seconds * conductance_w_k / (MAX_STABILITY_FRACTION * total_capacity_j_k)),
        )
        substep_seconds = dt_seconds / substep_count
        interval_start_temperature = indoor_temperature_c
        for _ in range(substep_count):
            if _ > 0:
                flows = component_transfers(
                    built,
                    config.wall_assembly,
                    config.roof_assembly,
                    config.floor_assembly,
                    config.openings,
                    outdoor,
                    indoor_temperature_c,
                    config.ground_temperature_c,
                )
                net = flows.total_w + solar + internal
            indoor_temperature_c += net * substep_seconds / total_capacity_j_k
        next_temperature = indoor_temperature_c

        if mass_state is not None:
            mass_flow = mass_state.storage_rate_w(
                interval_start_temperature, next_temperature, dt_seconds
            )
            mass_state.temperature_c = next_temperature
            mass_temperature = mass_state.temperature_c
        else:
            mass_flow = 0.0
            mass_temperature = next_temperature

        component_values = (
            flows.wall_w,
            flows.roof_w,
            flows.floor_w,
            flows.window_w,
            flows.door_w,
        )
        total_gain = sum(v for v in component_values if v > 0.0) + solar + internal
        total_loss = sum(abs(v) for v in component_values if v < 0.0)

        rows.append(
            {
                "timestamp": frame["timestamp"].iloc[index],
                "timestep_hours": dt_hours,
                "outdoor_temperature_c": outdoor,
                "solar_radiation_w_m2": radiation,
                "indoor_temperature_c": next_temperature,
                "wall_heat_transfer_w": flows.wall_w,
                "roof_heat_transfer_w": flows.roof_w,
                "floor_heat_transfer_w": flows.floor_w,
                "window_heat_transfer_w": flows.window_w,
                "door_heat_transfer_w": flows.door_w,
                "solar_heat_gain_w": solar,
                "internal_heat_gain_w": internal,
                "thermal_mass_heat_flow_w": mass_flow,
                "thermal_mass_temperature_c": mass_temperature,
                "total_heat_gain_w": total_gain,
                "total_heat_loss_w": total_loss,
                "net_heat_balance_w": net,
            }
        )
        indoor_temperature_c = next_temperature

    records = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    median_hours = float(intervals_hours.median())
    return SimulationResult(
        config_name=config.name,
        records=records,
        timestep_hours=median_hours,
        weather_category=weather_category,
    )

