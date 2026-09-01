"""Shared fixtures for the shelter tests."""
import numpy as np
import pandas as pd
import pytest

from building_hvac_twin.shelter import (
    EnvelopeAssembly,
    InternalHeatSources,
    Layer,
    Openings,
    ShelterConfig,
    ShelterGeometry,
    ThermalMass,
)


@pytest.fixture
def synthetic_weather() -> pd.DataFrame:
    """SYNTHETIC DEMONSTRATION WEATHER, not measured Ladakh data."""
    timestamps = pd.date_range("2025-01-15", periods=24, freq="h")
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


@pytest.fixture
def constant_weather() -> pd.DataFrame:
    """24 hours of constant -15 C, zero sun. SYNTHETIC DEMONSTRATION WEATHER."""
    timestamps = pd.date_range("2025-01-15", periods=24, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "outdoor_temperature_c": -15.0,
            "solar_radiation_w_m2": 0.0,
        }
    )


@pytest.fixture
def wall_assembly() -> EnvelopeAssembly:
    return EnvelopeAssembly(
        "wall",
        [
            Layer(0.35, 0.7, 1800.0, 840.0, "brick_masonry"),
            Layer(0.1, 0.035, 30.0, 1400.0, "insulation"),
        ],
    )


@pytest.fixture
def roof_assembly() -> EnvelopeAssembly:
    return EnvelopeAssembly(
        "roof",
        [
            Layer(0.15, 0.035, 30.0, 1400.0, "insulation"),
            Layer(0.05, 1.7, 2300.0, 880.0, "concrete"),
        ],
    )


@pytest.fixture
def floor_assembly() -> EnvelopeAssembly:
    return EnvelopeAssembly("floor", [Layer(0.1, 0.035, 30.0, 1400.0, "insulation")])


@pytest.fixture
def reference_config(wall_assembly, roof_assembly, floor_assembly) -> ShelterConfig:
    """REFERENCE / DEMONSTRATION shelter, not a measured Ladakh design."""
    return ShelterConfig(
        name="reference-demo-shelter",
        geometry=ShelterGeometry(length_m=6.0, width_m=4.0, height_m=3.0, orientation_deg=180.0),
        wall_assembly=wall_assembly,
        roof_assembly=roof_assembly,
        floor_assembly=floor_assembly,
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
