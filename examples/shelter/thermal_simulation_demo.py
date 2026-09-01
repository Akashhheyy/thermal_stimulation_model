"""Reference passive shelter thermal simulation demo.

The weather below is SYNTHETIC DEMONSTRATION DATA, not measured Ladakh data.
Material and construction values are REFERENCE / DEMONSTRATION VALUES.

Run from the repository root:

    python examples/shelter/thermal_simulation_demo.py

(works once the package is installed with `python -m pip install -e .`, or
with PYTHONPATH=src set).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from building_hvac_twin.shelter import (  # noqa: E402
    EnvelopeAssembly,
    InternalHeatSources,
    Layer,
    Openings,
    ShelterConfig,
    ShelterGeometry,
    ThermalMass,
    simulate_shelter,
)


def synthetic_weather(hours: int = 24) -> pd.DataFrame:
    """SYNTHETIC DEMONSTRATION WEATHER, not measured Ladakh data."""
    timestamps = pd.date_range("2025-01-15", periods=hours, freq="h")
    hour = timestamps.hour.to_numpy()
    outdoor = -12.0 + 6.0 * np.sin(np.pi * (hour - 4) / 12.0)
    solar = np.maximum(0.0, 700.0 * np.sin(np.pi * (hour - 7) / 12.0))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "outdoor_temperature_c": outdoor,
            "solar_radiation_w_m2": solar,
        }
    )


def reference_shelter() -> ShelterConfig:
    """REFERENCE / DEMONSTRATION shelter, not a measured Ladakh design."""
    wall = EnvelopeAssembly(
        "wall",
        [
            Layer(0.35, 0.7, 1800.0, 840.0, "brick_masonry"),
            Layer(0.1, 0.035, 30.0, 1400.0, "insulation"),
        ],
    )
    roof = EnvelopeAssembly(
        "roof",
        [
            Layer(0.15, 0.035, 30.0, 1400.0, "insulation"),
            Layer(0.05, 1.7, 2300.0, 880.0, "concrete"),
        ],
    )
    floor = EnvelopeAssembly("floor", [Layer(0.1, 0.035, 30.0, 1400.0, "insulation")])
    return ShelterConfig(
        name="reference-demo-shelter",
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
        thermal_mass=ThermalMass(mass_kg=800.0, specific_heat_j_kgk=900.0, initial_temperature_c=5.0),
        internal_heat_sources=InternalHeatSources(
            occupant_count=2,
            sensible_heat_per_person_w=75.0,
            equipment_heat_w=50.0,
            lighting_heat_w=20.0,
        ),
    )


def main() -> None:
    weather = synthetic_weather()
    config = reference_shelter()
    result = simulate_shelter(config, weather)
    records = result.records

    print("=" * 78)
    print("PASSIVE SHELTER THERMAL SIMULATION DEMO")
    print("Weather: SYNTHETIC DEMONSTRATION DATA (not measured Ladakh data)")
    print("Materials: REFERENCE / DEMONSTRATION VALUES")
    print("=" * 78)
    print(f"Shelter: {config.geometry.length_m} x {config.geometry.width_m} x "
          f"{config.geometry.height_m} m, {config.name}")
    print()

    print("Hourly indoor temperature (SYNTHETIC DEMONSTRATION DATA):")
    header = f"{'time':<18}{'outdoor C':>10}{'solar W/m2':>12}{'indoor C':>10}"
    print(header)
    for row in records.itertuples(index=False):
        stamp = row.timestamp.strftime("%Y-%m-%d %H:%M")
        print(
            f"{stamp:<18}{row.outdoor_temperature_c:>10.1f}"
            f"{row.solar_radiation_w_m2:>12.1f}{row.indoor_temperature_c:>10.2f}"
        )
    print()

    total_loss_kwh = records["total_heat_loss_w"].sum() * 1.0 / 1000.0
    total_gain_kwh = records["total_heat_gain_w"].sum() * 1.0 / 1000.0
    solar_kwh = records["solar_heat_gain_w"].sum() * 1.0 / 1000.0
    internal_kwh = records["internal_heat_gain_w"].sum() * 1.0 / 1000.0
    stored_kwh = records["thermal_mass_heat_flow_w"].clip(lower=0).sum() * 1.0 / 1000.0
    released_kwh = records["thermal_mass_heat_flow_w"].clip(upper=0).abs().sum() * 1.0 / 1000.0

    print("24-hour totals (1 hour per timestep):")
    print(f"  total heat gain through envelope + sun + internal: {total_gain_kwh:8.2f} kWh")
    print(f"  total conductive heat loss:                        {total_loss_kwh:8.2f} kWh")
    print(f"  solar heat gain:                                   {solar_kwh:8.2f} kWh")
    print(f"  internal heat gain:                                {internal_kwh:8.2f} kWh")
    print(f"  thermal mass absorbed:                             {stored_kwh:8.2f} kWh")
    print(f"  thermal mass released:                             {released_kwh:8.2f} kWh")
    print()
    print("Per-component average heat flows (W, positive into the shelter):")
    for column in (
        "wall_heat_transfer_w",
        "roof_heat_transfer_w",
        "floor_heat_transfer_w",
        "window_heat_transfer_w",
        "door_heat_transfer_w",
    ):
        print(f"  {column:<26}{records[column].mean():>10.1f}")
    print()
    print(
        "Indoor temperature range: "
        f"{records['indoor_temperature_c'].min():.2f} to "
        f"{records['indoor_temperature_c'].max():.2f} C"
    )


if __name__ == "__main__":
    main()
