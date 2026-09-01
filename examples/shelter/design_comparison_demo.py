"""Design comparison demo: three passive shelters on identical weather.

The weather below is SYNTHETIC DEMONSTRATION DATA, not measured Ladakh data.
Material and construction values are REFERENCE / DEMONSTRATION VALUES.

Three designs are simulated against EXACTLY the same 24-hour weather series:

    Design A: baseline construction
    Design B: improved insulation
    Design C: improved insulation + thermal mass

Run from the repository root:

    python examples/shelter/design_comparison_demo.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from building_hvac_twin.shelter import (  # noqa: E402
    ComfortRange,
    EnvelopeAssembly,
    InternalHeatSources,
    Layer,
    Openings,
    PerformanceWeights,
    ShelterConfig,
    ShelterGeometry,
    ThermalMass,
    compare_designs,
)


def synthetic_weather(hours: int = 24) -> pd.DataFrame:
    """SYNTHETIC DEMONSTRATION WEATHER, not measured Ladakh data."""
    timestamps = pd.date_range("2025-01-15", periods=hours, freq="h")
    hour = timestamps.hour.to_numpy()
    outdoor = -6.0 + 5.0 * np.sin(np.pi * (hour - 4) / 12.0)
    solar = np.maximum(0.0, 650.0 * np.sin(np.pi * (hour - 7) / 12.0))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "outdoor_temperature_c": outdoor,
            "solar_radiation_w_m2": solar,
        }
    )


def baseline_wall() -> EnvelopeAssembly:
    """REFERENCE / DEMONSTRATION baseline: single masonry leaf."""
    return EnvelopeAssembly("wall-baseline", [Layer(0.35, 0.7, 1800.0, 840.0, "brick_masonry")])


def insulated_wall() -> EnvelopeAssembly:
    """REFERENCE / DEMONSTRATION improved wall: masonry plus insulation."""
    return EnvelopeAssembly(
        "wall-insulated",
        [
            Layer(0.35, 0.7, 1800.0, 840.0, "brick_masonry"),
            Layer(0.1, 0.035, 30.0, 1400.0, "insulation"),
        ],
    )


def baseline_roof() -> EnvelopeAssembly:
    return EnvelopeAssembly("roof-baseline", [Layer(0.15, 1.7, 2300.0, 880.0, "concrete")])


def insulated_roof() -> EnvelopeAssembly:
    return EnvelopeAssembly(
        "roof-insulated",
        [
            Layer(0.15, 0.035, 30.0, 1400.0, "insulation"),
            Layer(0.05, 1.7, 2300.0, 880.0, "concrete"),
        ],
    )


def baseline_floor() -> EnvelopeAssembly:
    return EnvelopeAssembly("floor-baseline", [Layer(0.15, 1.7, 2300.0, 880.0, "concrete")])


def insulated_floor() -> EnvelopeAssembly:
    return EnvelopeAssembly("floor-insulated", [Layer(0.1, 0.035, 30.0, 1400.0, "insulation")])


def design(
    name: str,
    wall: EnvelopeAssembly,
    roof: EnvelopeAssembly,
    floor: EnvelopeAssembly,
    thermal_mass: ThermalMass | None,
) -> ShelterConfig:
    return ShelterConfig(
        name=name,
        geometry=ShelterGeometry(length_m=6.0, width_m=4.0, height_m=3.0, orientation_deg=180.0),
        wall_assembly=wall,
        roof_assembly=roof,
        floor_assembly=floor,
        openings=Openings(
            window_area_m2=2.0,
            door_area_m2=2.0,
            window_solar_heat_gain_coefficient=0.7,
            window_wall_orientation="south",
            door_wall_orientation="south",
        ),
        thermal_mass=thermal_mass,
        internal_heat_sources=InternalHeatSources(
            occupant_count=2,
            sensible_heat_per_person_w=75.0,
            equipment_heat_w=50.0,
            lighting_heat_w=20.0,
        ),
    )


def build_designs() -> list[ShelterConfig]:
    return [
        design("A-baseline", baseline_wall(), baseline_roof(), baseline_floor(), None),
        design(
            "B-improved-insulation",
            insulated_wall(),
            insulated_roof(),
            insulated_floor(),
            None,
        ),
        design(
            "C-insulation-plus-mass",
            insulated_wall(),
            insulated_roof(),
            insulated_floor(),
            ThermalMass(mass_kg=800.0, specific_heat_j_kgk=4186.0, initial_temperature_c=8.0),
        ),
    ]


def main() -> None:
    weather = synthetic_weather()
    designs = build_designs()
    weights = PerformanceWeights(
        comfort=1.0, heat_retention=1.0, solar_utilization=1.0, thermal_stability=1.0
    )
    comfort_band = ComfortRange(minimum_comfort_temperature_c=18.0, maximum_comfort_temperature_c=24.0)
    report = compare_designs(designs, weather, weights=weights, comfort_range=comfort_band)

    print("=" * 96)
    print("PASSIVE SHELTER DESIGN COMPARISON DEMO")
    print("Weather: SYNTHETIC DEMONSTRATION DATA (not measured Ladakh data)")
    print("Materials: REFERENCE / DEMONSTRATION VALUES")
    print("All three designs were simulated against EXACTLY the same 24-hour weather series.")
    print("=" * 96)
    print(f"Score weights (formula documented in comparison.py): {weights.to_dict()}")
    print(f"Comfort band applied to every design: {comfort_band.to_dict()}")
    print()

    header = (
        f"{'Design':<24}{'Comfort %':>10}{'Min C':>8}{'Max C':>8}{'Range C':>9}"
        f"{'Loss kWh':>10}{'Solar kWh':>11}{'Mass kWh':>10}{'Score':>8}"
    )
    print(header)
    print("-" * 96)
    table = report.table.set_index("design")
    for name in report.ranking:
        row = table.loc[name]
        print(
            f"{name:<24}{row['percent_time_comfortable']:>10.1f}"
            f"{row['minimum_indoor_temperature_c']:>8.2f}"
            f"{row['maximum_indoor_temperature_c']:>8.2f}"
            f"{row['indoor_temperature_range_c']:>9.2f}"
            f"{row['total_heat_loss_kwh']:>10.2f}"
            f"{row['total_solar_gain_kwh']:>11.2f}"
            f"{row['thermal_mass_net_kwh']:>10.2f}"
            f"{row['performance_score']:>8.1f}"
        )
    print("-" * 96)
    print(f"Ranking (best first): {', '.join(report.ranking)}")
    print("Mass kWh column: net thermal-mass effect (absorbed minus released);")
    print("negative means the mass gave back more stored heat than it absorbed.")
    print()
    print("Supporting detail:")
    for row in report.table.itertuples(index=False):
        print(
            f"  {row.design:<24} below: {row.degree_hours_below_comfort:>7.1f} Kh"
            f"  above: {row.degree_hours_above_comfort:>7.1f} Kh"
            f"  mean indoor: {row.mean_indoor_temperature_c:>6.2f} C"
        )


if __name__ == "__main__":
    main()

