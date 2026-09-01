"""User-configurable comfort analysis for simulated shelter temperatures.

This is a prototype comfort indicator built directly from the simulated
indoor temperatures and the limits the user supplies.  It is NOT a certified
human-comfort model (no humidity, air speed, clothing, or metabolic rate).
Nothing here assumes an official comfort standard: the limits always come
from the caller, defaulting to the repository's :class:`ComfortRange`
(18 to 24 C) only when the caller passes nothing.

Degree-hours use the actual timestep of every record, so variable-interval
simulations integrate correctly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import pandas as pd

from .models import ComfortRange

__all__ = ["ComfortSummary", "comfort_metrics", "comfort_summary"]


@dataclass
class ComfortSummary:
    """Aggregate comfort indicators for one simulation."""

    minimum_indoor_temperature_c: float
    maximum_indoor_temperature_c: float
    mean_indoor_temperature_c: float
    indoor_temperature_range_c: float
    hours_comfortable: float
    hours_below_comfort: float
    hours_above_comfort: float
    percent_time_comfortable: float
    percent_time_below_comfort: float
    percent_time_above_comfort: float
    degree_hours_below_comfort: float
    degree_hours_above_comfort: float
    maximum_violation_c: float
    minimum_comfort_temperature_c: float
    maximum_comfort_temperature_c: float

    @property
    def total_degree_hours_outside_comfort(self) -> float:
        return self.degree_hours_below_comfort + self.degree_hours_above_comfort

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _timestep_hours(records: pd.DataFrame) -> pd.Series:
    """Per-record duration in hours: explicit column, timestamps, or 1 h."""
    if "timestep_hours" in records.columns:
        hours = pd.to_numeric(records["timestep_hours"], errors="coerce")
        if hours.isna().any() or (hours <= 0).any():
            raise ValueError("timestep_hours must contain positive numbers")
        return hours
    if "timestamp" in records.columns:
        stamps = pd.to_datetime(records["timestamp"])
        hours = stamps.diff().dt.total_seconds().div(3600)
        if len(records) > 1:
            hours.iloc[0] = float(hours.iloc[1:].median())
            if (hours <= 0).any():
                raise ValueError("timestamps must be strictly increasing")
        else:
            hours.iloc[0] = 1.0
        return hours
    # Documented fallback: a bare temperature series is treated as hourly.
    return pd.Series(1.0, index=records.index)


def comfort_metrics(
    indoor_temperatures_c: Sequence[float],
    comfort_range: ComfortRange,
    timestep_hours: float | Sequence[float],
) -> ComfortSummary:
    """Comfort indicators from explicit temperatures, limits, and durations.

    ``timestep_hours`` is either one number applied to every record or a
    per-record sequence; degree-hours integrate ``violation * duration``.
    """
    temperatures = [float(value) for value in indoor_temperatures_c]
    if not temperatures:
        raise ValueError("indoor_temperatures_c must contain at least one value")
    if isinstance(timestep_hours, (int, float)):
        hours = [float(timestep_hours)] * len(temperatures)
    else:
        hours = [float(value) for value in timestep_hours]
    if len(hours) != len(temperatures):
        raise ValueError("timestep_hours must match the number of temperatures")
    if any(value <= 0.0 for value in hours):
        raise ValueError("every timestep must be positive")

    minimum = min(temperatures)
    maximum = max(temperatures)
    mean = sum(temperatures) / len(temperatures)

    hours_below = 0.0
    hours_above = 0.0
    degree_hours_below = 0.0
    degree_hours_above = 0.0
    maximum_violation = 0.0
    for temperature, duration in zip(temperatures, hours):
        low = max(comfort_range.minimum_comfort_temperature_c - temperature, 0.0)
        high = max(temperature - comfort_range.maximum_comfort_temperature_c, 0.0)
        violation = low + high
        hours_below += duration if low > 0.0 else 0.0
        hours_above += duration if high > 0.0 else 0.0
        degree_hours_below += low * duration
        degree_hours_above += high * duration
        maximum_violation = max(maximum_violation, violation)

    total_hours = sum(hours)
    percent = 100.0 / max(total_hours, 1e-12)
    return ComfortSummary(
        minimum_indoor_temperature_c=minimum,
        maximum_indoor_temperature_c=maximum,
        mean_indoor_temperature_c=mean,
        indoor_temperature_range_c=maximum - minimum,
        hours_comfortable=total_hours - hours_below - hours_above,
        hours_below_comfort=hours_below,
        hours_above_comfort=hours_above,
        percent_time_comfortable=100.0 - (hours_below + hours_above) * percent,
        percent_time_below_comfort=hours_below * percent,
        percent_time_above_comfort=hours_above * percent,
        degree_hours_below_comfort=degree_hours_below,
        degree_hours_above_comfort=degree_hours_above,
        maximum_violation_c=maximum_violation,
        minimum_comfort_temperature_c=comfort_range.minimum_comfort_temperature_c,
        maximum_comfort_temperature_c=comfort_range.maximum_comfort_temperature_c,
    )


def comfort_summary(
    records: pd.DataFrame,
    comfort_range: ComfortRange | None = None,
) -> dict[str, float]:
    """Comfort indicators from a simulation records DataFrame.

    Requires an ``indoor_temperature_c`` column.  Durations come from a
    ``timestep_hours`` column when present, else from ``timestamp`` diffs,
    else the documented 1 h fallback.  ``comfort_range`` defaults to the
    repository default band (18 to 24 C) when the caller passes nothing.
    """
    if not isinstance(records, pd.DataFrame):
        raise ValueError("comfort_summary expects a pandas DataFrame")
    if "indoor_temperature_c" not in records.columns:
        raise ValueError("records must contain an 'indoor_temperature_c' column")
    if records.empty:
        raise ValueError("records must contain at least one row")
    range_used = comfort_range if comfort_range is not None else ComfortRange()
    hours = _timestep_hours(records)
    metrics = comfort_metrics(
        records["indoor_temperature_c"].tolist(),
        range_used,
        hours.tolist(),
    )
    summary = metrics.to_dict()
    summary["total_degree_hours_outside_comfort"] = metrics.total_degree_hours_outside_comfort
    return summary

