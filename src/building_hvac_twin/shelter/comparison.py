"""Same-weather design comparison with a transparent performance score.

Every design in a comparison is simulated against EXACTLY the same weather
records, so differences in the metrics come from the designs alone.

Performance score (higher is better, range 0 to 100), fully configurable and
documented; there is no hidden model:

    comfort_component_i    = percent_time_comfortable_i / 100
                             (absolute, not scaled across designs)
    retention_component_i  = 1 - minmax(total_heat_loss_kwh_i)
                             (lower heat loss is better)
    solar_component_i      = minmax(total_solar_gain_kwh_i)
                             (more captured solar gain is better)
    stability_component_i  = 1 - minmax(indoor_temperature_range_c_i)
                             (smaller indoor temperature swings are better)

    minmax(x)_i = (x_i - min(x)) / (max(x) - min(x))
    minmax with max == min (one design, or tied values) is defined as 1.0,
    which makes that component neutral rather than zero.

    performance_score_i = 100 * (w_c*comfort + w_r*retention + w_s*solar
                                 + w_t*stability) / (w_c + w_r + w_s + w_t)

A weight of 0 removes that component from the score entirely.  Because the
retention, solar, and stability components are min-max scaled across the
compared designs, they express relative standing within the set; the comfort
component is absolute.  This is a prototype ranking aid, not a certified
rating.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Iterable, Sequence

import pandas as pd

from .comfort import comfort_summary
from .models import ComfortRange, DataCategory, SimulationResult, ShelterConfig
from .simulation import simulate_shelter

__all__ = [
    "PerformanceWeights",
    "ComparisonReport",
    "simulate_designs",
    "design_metrics",
    "score_components",
    "performance_scores",
    "compare_designs",
]


@dataclass
class PerformanceWeights:
    """Weights for the four documented score components (all nonnegative)."""

    comfort: float = 1.0
    heat_retention: float = 1.0
    solar_utilization: float = 1.0
    thermal_stability: float = 1.0

    def __post_init__(self) -> None:
        for name in ("comfort", "heat_retention", "solar_utilization", "thermal_stability"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"PerformanceWeights.{name} must be nonnegative")
        if sum(float(getattr(self, name)) for name in (
            "comfort", "heat_retention", "solar_utilization", "thermal_stability"
        )) <= 0.0:
            raise ValueError("at least one PerformanceWeights entry must be positive")

    @property
    def total(self) -> float:
        return (
            float(self.comfort)
            + float(self.heat_retention)
            + float(self.solar_utilization)
            + float(self.thermal_stability)
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _minmax(values: Sequence[float]) -> list[float]:
    """Min-max scale to [0, 1]; constant series maps to all 1.0 (neutral)."""
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    span = high - low
    return [(value - low) / span for value in values]


def score_components(metrics_rows: Sequence[dict]) -> dict[str, list[float]]:
    """Normalized score components for a set of design metric rows."""
    if not metrics_rows:
        raise ValueError("score_components needs at least one metric row")
    losses = [float(row["total_heat_loss_kwh"]) for row in metrics_rows]
    gains = [float(row["total_solar_gain_kwh"]) for row in metrics_rows]
    ranges = [float(row["indoor_temperature_range_c"]) for row in metrics_rows]
    return {
        "comfort": [float(row["percent_time_comfortable"]) / 100.0 for row in metrics_rows],
        "heat_retention": [1.0 - scaled for scaled in _minmax(losses)],
        "solar_utilization": _minmax(gains),
        "thermal_stability": [1.0 - scaled for scaled in _minmax(ranges)],
    }


def performance_scores(
    metrics_rows: Sequence[dict],
    weights: PerformanceWeights | None = None,
) -> list[float]:
    """Weighted performance score per design row; higher is better."""
    used = weights if weights is not None else PerformanceWeights()
    components = score_components(metrics_rows)
    weighted = {
        "comfort": float(used.comfort),
        "heat_retention": float(used.heat_retention),
        "solar_utilization": float(used.solar_utilization),
        "thermal_stability": float(used.thermal_stability),
    }
    scores = []
    for index in range(len(metrics_rows)):
        total = sum(weight * components[name][index] for name, weight in weighted.items())
        scores.append(100.0 * total / used.total)
    return scores


def simulate_designs(
    configs: Sequence[ShelterConfig],
    weather,
    initial_indoor_temperature_c: float | None = None,
    weather_category: DataCategory = DataCategory.SYNTHETIC,
) -> list[SimulationResult]:
    """Simulate every design against EXACTLY the same weather records.

    The same weather object is reused for every call, and every design sees
    identical outdoor temperatures and solar radiation by construction.
    """
    if not configs:
        raise ValueError("simulate_designs needs at least one ShelterConfig")
    return [
        simulate_shelter(
            config,
            weather,
            initial_indoor_temperature_c=initial_indoor_temperature_c,
            weather_category=weather_category,
        )
        for config in configs
    ]


def design_metrics(
    result: SimulationResult,
    comfort_range: ComfortRange | None = None,
) -> dict[str, float]:
    """Measurable metrics for one design result (energy columns in kWh)."""
    records = result.records
    range_used = comfort_range if comfort_range is not None else ComfortRange()
    comfort = comfort_summary(records, range_used)
    hours = records["timestep_hours"] if "timestep_hours" in records.columns else 1.0
    heat_loss_kwh = float((records["total_heat_loss_w"] * hours).sum() / 1000.0)
    solar_kwh = float((records["solar_heat_gain_w"] * hours).sum() / 1000.0)
    mass_flow = records.get("thermal_mass_heat_flow_w")
    mass_absorbed = float((mass_flow.clip(lower=0.0) * hours).sum() / 1000.0) if mass_flow is not None else 0.0
    mass_released = float((mass_flow.clip(upper=0.0).abs() * hours).sum() / 1000.0) if mass_flow is not None else 0.0
    return {
        "design": result.config_name,
        "percent_time_comfortable": comfort["percent_time_comfortable"],
        "percent_time_below_comfort": comfort["percent_time_below_comfort"],
        "percent_time_above_comfort": comfort["percent_time_above_comfort"],
        "minimum_indoor_temperature_c": comfort["minimum_indoor_temperature_c"],
        "maximum_indoor_temperature_c": comfort["maximum_indoor_temperature_c"],
        "mean_indoor_temperature_c": comfort["mean_indoor_temperature_c"],
        "indoor_temperature_range_c": comfort["indoor_temperature_range_c"],
        "degree_hours_below_comfort": comfort["degree_hours_below_comfort"],
        "degree_hours_above_comfort": comfort["degree_hours_above_comfort"],
        "total_heat_loss_kwh": heat_loss_kwh,
        "total_solar_gain_kwh": solar_kwh,
        "thermal_mass_absorbed_kwh": mass_absorbed,
        "thermal_mass_released_kwh": mass_released,
        "thermal_mass_net_kwh": mass_absorbed - mass_released,
    }


@dataclass
class ComparisonReport:
    """Results of comparing several designs under identical weather."""

    weights: PerformanceWeights
    comfort_range: ComfortRange
    table: pd.DataFrame
    results: dict[str, SimulationResult] = field(default_factory=dict)

    @property
    def ranking(self) -> list[str]:
        """Design names from best to worst; ties break alphabetically."""
        ordered = self.table.sort_values(
            ["performance_score", "design"], ascending=[False, True]
        )
        return ordered["design"].tolist()

    @property
    def best_design(self) -> str:
        return self.ranking[0]

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.to_dict(),
            "comfort_range": self.comfort_range.to_dict(),
            "ranking": self.ranking,
            "table": self.table.to_dict(orient="records"),
        }


def compare_designs(
    configs: Sequence[ShelterConfig],
    weather,
    weights: PerformanceWeights | None = None,
    comfort_range: ComfortRange | None = None,
    initial_indoor_temperature_c: float | None = None,
) -> ComparisonReport:
    """Simulate every design on identical weather and rank them.

    ``comfort_range`` applies the SAME limits to every design so the
    comparison stays fair; the per-config comfort ranges are not used here.
    """
    used_weights = weights if weights is not None else PerformanceWeights()
    used_range = comfort_range if comfort_range is not None else ComfortRange()
    results = simulate_designs(
        configs, weather, initial_indoor_temperature_c=initial_indoor_temperature_c
    )
    rows = [design_metrics(result, used_range) for result in results]
    scores = performance_scores(rows, used_weights)
    for row, score in zip(rows, scores):
        row["performance_score"] = score
    table = pd.DataFrame(rows)
    return ComparisonReport(
        weights=used_weights,
        comfort_range=used_range,
        table=table,
        results={result.config_name: result for result in results},
    )

