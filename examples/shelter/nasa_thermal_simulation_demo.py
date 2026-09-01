"""End-to-end demo: NASA POWER historical weather into the thermal engine.

Pipeline demonstrated (no ML, no frontend, no database):

    NASA POWER API
        -> shelter.weather (request, parse, convert)
        -> pandas DataFrame (timestamp, outdoor_temperature_c, ...)
        -> shelter.simulation.simulate_shelter (existing engine, unchanged)
        -> thermal performance results

The weather below is NASA POWER HISTORICAL WEATHER DATA.  The shelter
configuration uses REFERENCE / DEMONSTRATION material values.

Run from the repository root (internet access required):

    python examples/shelter/nasa_thermal_simulation_demo.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from building_hvac_twin.shelter import (  # noqa: E402
    EnvelopeAssembly,
    InternalHeatSources,
    Layer,
    Openings,
    ShelterConfig,
    ShelterGeometry,
    ThermalMass,
    comfort_summary,
    get_nasa_weather_data,
    simulate_shelter,
)

LATITUDE = 34.1645
LONGITUDE = 77.5789
START_DATE = "2024-01-01"
END_DATE = "2024-01-03"
CSV_PATH = Path("outputs/shelter/nasa_thermal_simulation_leh.csv")

# The first-10-row table below uses the ACTUAL simulation output columns.
ROW_COLUMNS = [
    "timestamp",
    "outdoor_temperature_c",
    "solar_radiation_w_m2",
    "indoor_temperature_c",
    "solar_heat_gain_w",
    "thermal_mass_heat_flow_w",
    "total_heat_loss_w",
    "net_heat_balance_w",
]
ROW_LEGEND = (
    "Column legend (actual engine names): solar_gain -> solar_heat_gain_w; "
    "thermal_mass_net -> thermal_mass_heat_flow_w (+ absorbed, - released); "
    "conductive_loss -> total_heat_loss_w; net_balance -> net_heat_balance_w"
)


def reference_shelter() -> ShelterConfig:
    """REFERENCE / DEMONSTRATION shelter using the existing model API."""
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
        name="leh-nasa-reference",
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
        thermal_mass=ThermalMass(mass_kg=800.0, specific_heat_j_kgk=4186.0, initial_temperature_c=5.0),
        internal_heat_sources=InternalHeatSources(
            occupant_count=2,
            sensible_heat_per_person_w=75.0,
            equipment_heat_w=50.0,
            lighting_heat_w=20.0,
        ),
    )


def to_kwh(records: pd.DataFrame, column: str) -> float:
    """Convert a watt column to kWh using each row's timestep."""
    hours = records["timestep_hours"] if "timestep_hours" in records.columns else 1.0
    return float((records[column] * hours).sum() / 1000.0)


def main() -> int:
    print("=" * 64)
    print("NASA POWER -> PASSIVE SHELTER THERMAL SIMULATION")
    print("=" * 64)
    print("Location:        Leh, Ladakh")
    print(f"Latitude:        {LATITUDE}")
    print(f"Longitude:       {LONGITUDE}")
    print(f"Weather period:  {START_DATE} -> {END_DATE}")
    print("Resolution:      Hourly UTC")
    print("Parameters:      T2M, ALLSKY_SFC_SW_DWN, WS10M, RH2M")

    try:
        weather = get_nasa_weather_data(LATITUDE, LONGITUDE, START_DATE, END_DATE)
    except Exception as error:  # NasaWeatherError and network variants
        print(f"\nNASA POWER request failed: {error}")
        print("The thermal engine itself does not depend on the NASA API;")
        print("rerun with internet access, or feed any compatible DataFrame.")
        return 1

    skipped = weather.attrs.get("skipped_missing_required_records", 0)
    print(f"Weather records: {len(weather)}"
          f" (skipped for missing NASA values: {skipped})")

    config = reference_shelter()
    result = simulate_shelter(config, weather)
    records = result.records
    comfort = comfort_summary(records)

    mass_absorbed_kwh = to_kwh(records.assign(_m=records["thermal_mass_heat_flow_w"].clip(lower=0.0)), "_m")
    mass_released_kwh = to_kwh(records.assign(_m=records["thermal_mass_heat_flow_w"].clip(upper=0.0).abs()), "_m")

    print()
    print("THERMAL RESULTS")
    print(f"Minimum indoor temperature:  {records['indoor_temperature_c'].min():.2f} C")
    print(f"Maximum indoor temperature:  {records['indoor_temperature_c'].max():.2f} C")
    print(f"Average indoor temperature:  {records['indoor_temperature_c'].mean():.2f} C")
    print(f"Indoor temperature range:    {comfort['indoor_temperature_range_c']:.2f} C")
    print(f"Total solar gain:            {to_kwh(records, 'solar_heat_gain_w'):.2f} kWh")
    print(f"Total conductive loss:       {to_kwh(records, 'total_heat_loss_w'):.2f} kWh")
    print(f"Thermal mass absorbed:       {mass_absorbed_kwh:.2f} kWh")
    print(f"Thermal mass released:       {mass_released_kwh:.2f} kWh")
    print(f"Comfort degree-hours:        "
          f"{comfort['degree_hours_below_comfort'] + comfort['degree_hours_above_comfort']:.1f} C.h")
    print("=" * 64)

    print()
    print("First 10 simulation rows (actual engine columns):")
    print(ROW_LEGEND)
    print(records[ROW_COLUMNS].head(10).to_string(index=False))

    try:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(CSV_PATH, index=False)
        print(f"\nCSV written: {CSV_PATH} ({len(records)} hourly records)")
    except OSError as error:
        print(f"\nCSV output skipped: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

