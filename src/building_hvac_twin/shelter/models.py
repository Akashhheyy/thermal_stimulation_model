"""Typed dataclasses for the passive shelter thermal model.

Every physical quantity carries its unit in the field name so that callers
cannot silently mix incompatible units.  Validation runs in ``__post_init__``
so a physically impossible value raises immediately at construction time
instead of deep inside the simulation loop.

This module replaces an earlier truncated, syntactically invalid version.
The original Vicena reference model (``building_hvac_twin.model`` and
related modules) is untouched and remains the baseline for this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from .validation import (
    reject_negative,
    reject_nonpositive,
    reject_out_of_range,
    require_field,
)

__all__ = [
    "DataCategory",
    "Location",
    "WeatherRecord",
    "Material",
    "Layer",
    "EnvelopeAssembly",
    "Openings",
    "ThermalMass",
    "InternalHeatSources",
    "ComfortRange",
    "ShelterGeometry",
    "ShelterConfig",
    "SimulationResult",
]


class DataCategory(str, Enum):
    """Provenance label for every bundled value.

    The repository never claims bundled numbers are measured site data.
    ``REFERENCE`` means a clearly labelled demonstration value that must be
    replaced by verified measurements before any real design decision.
    """

    SYNTHETIC = "synthetic"
    REFERENCE = "reference"
    MEASURED = "measured"


@dataclass
class Location:
    """Named study site.  Coordinates and altitude are metadata only."""

    name: str
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    altitude_m: float | None = None
    data_category: DataCategory = DataCategory.REFERENCE

    def __post_init__(self) -> None:
        require_field(self.name, "Location.name")
        reject_out_of_range(self.latitude_deg or 0.0, "Location.latitude_deg", -90.0, 90.0)
        reject_out_of_range(self.longitude_deg or 0.0, "Location.longitude_deg", -180.0, 180.0)
        reject_negative(self.altitude_m if self.altitude_m is not None else 0.0, "Location.altitude_m")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeatherRecord:
    """One outdoor weather interval for the shelter site."""

    timestamp: datetime
    outdoor_temperature_c: float
    solar_radiation_w_m2: float
    wind_speed_m_s: float = 0.0
    relative_humidity_percent: float = 0.0

    def __post_init__(self) -> None:
        require_field(self.timestamp, "WeatherRecord.timestamp")
        reject_negative(
            float(self.outdoor_temperature_c) + 273.15,
            "WeatherRecord.outdoor_temperature_c (below -273.15 C is unphysical)",
        )
        reject_negative(self.solar_radiation_w_m2, "WeatherRecord.solar_radiation_w_m2")
        reject_negative(self.wind_speed_m_s, "WeatherRecord.wind_speed_m_s")
        reject_out_of_range(self.relative_humidity_percent, "WeatherRecord.relative_humidity_percent", 0.0, 100.0)


@dataclass
class Material:
    """A construction material with thermal properties.

    Bundled values in this repository are REFERENCE / DEMONSTRATION values,
    not measured site measurements.  ``source`` must say where the numbers
    came from so they can be replaced later with verified data.
    """

    name: str
    thermal_conductivity_w_mk: float
    density_kg_m3: float
    specific_heat_j_kgk: float
    data_category: DataCategory = DataCategory.REFERENCE
    source: str = "reference demonstration value, not a measured site measurement"

    def __post_init__(self) -> None:
        require_field(self.name, "Material.name")
        reject_nonpositive(self.thermal_conductivity_w_mk, "Material.thermal_conductivity_w_mk")
        reject_nonpositive(self.density_kg_m3, "Material.density_kg_m3")
        reject_nonpositive(self.specific_heat_j_kgk, "Material.specific_heat_j_kgk")



@dataclass
class Layer:
    """One construction layer inside an envelope assembly.

    Resistance of the layer alone is ``thickness_m / thermal_conductivity_w_mk``
    (units m2K/W).  Surface film resistances live on the assembly, not here.
    """

    thickness_m: float
    thermal_conductivity_w_mk: float
    density_kg_m3: float
    specific_heat_j_kgk: float
    material_name: str = "unspecified"

    def __post_init__(self) -> None:
        reject_nonpositive(self.thickness_m, "Layer.thickness_m")
        reject_nonpositive(self.thermal_conductivity_w_mk, "Layer.thermal_conductivity_w_mk")
        reject_nonpositive(self.density_kg_m3, "Layer.density_kg_m3")
        reject_nonpositive(self.specific_heat_j_kgk, "Layer.specific_heat_j_kgk")
        require_field(self.material_name, "Layer.material_name")

    @property
    def conductive_resistance_m2k_w(self) -> float:
        """Layer resistance only: R = thickness / conductivity."""
        return self.thickness_m / self.thermal_conductivity_w_mk

    @property
    def areal_heat_capacity_j_m2k(self) -> float:
        """Heat stored per square metre: thickness * density * specific heat."""
        return self.thickness_m * self.density_kg_m3 * self.specific_heat_j_kgk

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvelopeAssembly:
    """A multi-layer construction such as a wall, roof, or floor.

    ``surface_resistance_inner_m2k_w`` and ``surface_resistance_outer_m2k_w``
    are explicit and configurable on purpose: they are not hidden inside
    unexplained constants.  Set either to 0.0 to exclude that film.

    ``u_value_w_m2k`` is the whole-assembly U-value including both films:
    U = 1 / (R_inner + sum(layer R) + R_outer).
    """

    name: str
    layers: list[Layer] = field(default_factory=list)
    surface_resistance_inner_m2k_w: float = 0.13
    surface_resistance_outer_m2k_w: float = 0.04

    def __post_init__(self) -> None:
        require_field(self.name, "EnvelopeAssembly.name")
        reject_negative(self.surface_resistance_inner_m2k_w, "EnvelopeAssembly.surface_resistance_inner_m2k_w")
        reject_negative(self.surface_resistance_outer_m2k_w, "EnvelopeAssembly.surface_resistance_outer_m2k_w")
        if not self.layers:
            raise ValueError(f"EnvelopeAssembly {self.name!r} must contain at least one Layer")
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, Layer):
                raise ValueError(f"EnvelopeAssembly {self.name!r} layer {index} is not a Layer")

    @property
    def total_resistance_m2k_w(self) -> float:
        """R_total = R_inner + sum of layer resistances + R_outer."""
        return (
            self.surface_resistance_inner_m2k_w
            + sum(layer.conductive_resistance_m2k_w for layer in self.layers)
            + self.surface_resistance_outer_m2k_w
        )

    @property
    def u_value_w_m2k(self) -> float:
        """U = 1 / R_total."""
        return 1.0 / self.total_resistance_m2k_w

    @property
    def areal_heat_capacity_j_m2k(self) -> float:
        """Sum of layer areal heat capacities."""
        return sum(layer.areal_heat_capacity_j_m2k for layer in self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "u_value_w_m2k": self.u_value_w_m2k,
            "total_resistance_m2k_w": self.total_resistance_m2k_w,
            "surface_resistance_inner_m2k_w": self.surface_resistance_inner_m2k_w,
            "surface_resistance_outer_m2k_w": self.surface_resistance_outer_m2k_w,
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass
class Openings:
    """Windows and doors cut into the shelter envelope.

    Areas are subtracted from the wall they belong to (see ``geometry.py``).
    U-values are whole-opening values including frames, given directly so the
    user controls them instead of the model inventing them.
    """

    window_area_m2: float = 0.0
    door_area_m2: float = 0.0
    number_of_windows: int = 0
    number_of_doors: int = 0
    window_u_value_w_m2k: float = 5.0
    door_u_value_w_m2k: float = 2.0
    window_solar_heat_gain_coefficient: float = 0.7
    window_wall_orientation: str = "south"
    door_wall_orientation: str = "south"

    def __post_init__(self) -> None:
        reject_negative(self.window_area_m2, "Openings.window_area_m2")
        reject_negative(self.door_area_m2, "Openings.door_area_m2")
        reject_negative(self.number_of_windows, "Openings.number_of_windows")
        reject_negative(self.number_of_doors, "Openings.number_of_doors")
        reject_negative(self.window_u_value_w_m2k, "Openings.window_u_value_w_m2k")
        reject_negative(self.door_u_value_w_m2k, "Openings.door_u_value_w_m2k")
        reject_out_of_range(
            self.window_solar_heat_gain_coefficient,
            "Openings.window_solar_heat_gain_coefficient",
            0.0,
            1.0,
        )
        require_field(self.window_wall_orientation, "Openings.window_wall_orientation")
        require_field(self.door_wall_orientation, "Openings.door_wall_orientation")

    @property
    def total_opening_area_m2(self) -> float:
        return self.window_area_m2 + self.door_area_m2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThermalMass:
    """Internal thermal storage mass (water, stone, adobe core, and so on).

    The mass participates dynamically in the time-step simulation through
    ``Q = m * cp * dT``.  It is never applied as a constant temperature offset.
    """

    mass_kg: float
    specific_heat_j_kgk: float
    initial_temperature_c: float
    material_name: str = "unspecified"

    def __post_init__(self) -> None:
        reject_nonpositive(self.mass_kg, "ThermalMass.mass_kg")
        reject_nonpositive(self.specific_heat_j_kgk, "ThermalMass.specific_heat_j_kgk")
        reject_negative(
            float(self.initial_temperature_c) + 273.15,
            "ThermalMass.initial_temperature_c (below -273.15 C is unphysical)",
        )

    @property
    def heat_capacity_j_k(self) -> float:
        """C_mass = mass * specific heat."""
        return self.mass_kg * self.specific_heat_j_kgk

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InternalHeatSources:
    """Internal heat gains.  Every per-person value is configurable."""

    occupant_count: int = 0
    sensible_heat_per_person_w: float = 75.0
    equipment_heat_w: float = 0.0
    lighting_heat_w: float = 0.0

    def __post_init__(self) -> None:
        reject_negative(self.occupant_count, "InternalHeatSources.occupant_count")
        reject_negative(self.sensible_heat_per_person_w, "InternalHeatSources.sensible_heat_per_person_w")
        reject_negative(self.equipment_heat_w, "InternalHeatSources.equipment_heat_w")
        reject_negative(self.lighting_heat_w, "InternalHeatSources.lighting_heat_w")

    @property
    def total_heat_w(self) -> float:
        """Total internal gain in watts."""
        return self.occupant_count * self.sensible_heat_per_person_w + self.equipment_heat_w + self.lighting_heat_w

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComfortRange:
    """Prototype comfort band.  This is not a certified human-comfort model."""

    minimum_comfort_temperature_c: float = 18.0
    maximum_comfort_temperature_c: float = 24.0

    def __post_init__(self) -> None:
        reject_negative(
            float(self.minimum_comfort_temperature_c) + 273.15,
            "ComfortRange.minimum_comfort_temperature_c (below -273.15 C is unphysical)",
        )
        reject_negative(
            float(self.maximum_comfort_temperature_c) + 273.15,
            "ComfortRange.maximum_comfort_temperature_c (below -273.15 C is unphysical)",
        )
        if self.minimum_comfort_temperature_c >= self.maximum_comfort_temperature_c:
            raise ValueError(
                "ComfortRange.minimum_comfort_temperature_c must be below "
                "ComfortRange.maximum_comfort_temperature_c, got "
                f"{self.minimum_comfort_temperature_c} and {self.maximum_comfort_temperature_c}"
            )

    def violation_c(self, indoor_temperature_c: float) -> float:
        """Degrees outside the band at one instant (0 when inside the band)."""
        low = max(self.minimum_comfort_temperature_c - indoor_temperature_c, 0.0)
        high = max(indoor_temperature_c - self.maximum_comfort_temperature_c, 0.0)
        return low + high

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShelterGeometry:
    """Rectangular shelter outer dimensions and orientation.

    ``orientation_deg`` is a compass bearing in degrees clockwise from north
    (0 to 360) for the wall that faces the short ``width_m`` side.  Detailed
    area derivation lives in ``geometry.py``.
    """

    length_m: float
    width_m: float
    height_m: float
    orientation_deg: float = 0.0

    def __post_init__(self) -> None:
        reject_nonpositive(self.length_m, "ShelterGeometry.length_m")
        reject_nonpositive(self.width_m, "ShelterGeometry.width_m")
        reject_nonpositive(self.height_m, "ShelterGeometry.height_m")
        reject_out_of_range(self.orientation_deg, "ShelterGeometry.orientation_deg", 0.0, 360.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass
class ShelterConfig:
    """Complete description of one passive shelter design.

    The auxiliary fields describe how much heating or cooling power would be
    needed if the passive balance alone leaves the shelter outside the comfort
    band.  HVAC is never the primary mechanism here; the purpose is to make
    the link "better passive design means less auxiliary energy" measurable.
    """

    name: str
    geometry: ShelterGeometry
    wall_assembly: EnvelopeAssembly
    roof_assembly: EnvelopeAssembly
    floor_assembly: EnvelopeAssembly
    openings: Openings = field(default_factory=Openings)
    thermal_mass: ThermalMass | None = None
    internal_heat_sources: InternalHeatSources = field(default_factory=InternalHeatSources)
    comfort_range: ComfortRange = field(default_factory=ComfortRange)
    initial_indoor_temperature_c: float = 10.0
    ground_temperature_c: float | None = None
    auxiliary_heating_allowed: bool = True
    auxiliary_cooling_allowed: bool = True
    air_capacity_per_volume_j_m3k: float = 1200.0

    def __post_init__(self) -> None:
        require_field(self.name, "ShelterConfig.name")
        reject_negative(float(self.initial_indoor_temperature_c) + 273.15, "ShelterConfig.initial_indoor_temperature_c")
        if self.ground_temperature_c is not None:
            reject_negative(float(self.ground_temperature_c) + 273.15, "ShelterConfig.ground_temperature_c")
        reject_nonpositive(self.air_capacity_per_volume_j_m3k, "ShelterConfig.air_capacity_per_volume_j_m3k")
        if self.thermal_mass is not None and not isinstance(self.thermal_mass, ThermalMass):
            raise ValueError("ShelterConfig.thermal_mass must be a ThermalMass or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "geometry": self.geometry.to_dict(),
            "wall_assembly": self.wall_assembly.to_dict(),
            "roof_assembly": self.roof_assembly.to_dict(),
            "floor_assembly": self.floor_assembly.to_dict(),
            "openings": self.openings.to_dict(),
            "thermal_mass": self.thermal_mass.to_dict() if self.thermal_mass else None,
            "internal_heat_sources": self.internal_heat_sources.to_dict(),
            "comfort_range": self.comfort_range.to_dict(),
            "initial_indoor_temperature_c": self.initial_indoor_temperature_c,
            "ground_temperature_c": self.ground_temperature_c,
        }


@dataclass
class SimulationResult:
    """One shelter simulation over a weather series.

    ``records`` holds one row per timestep with the documented output columns.
    ``summary`` holds aggregate metrics.  The result object is intentionally a
    thin wrapper over a pandas DataFrame so callers can inspect every number.
    """

    config_name: str
    records: pd.DataFrame
    timestep_hours: float
    weather_category: DataCategory = DataCategory.SYNTHETIC

    def __post_init__(self) -> None:
        require_field(self.config_name, "SimulationResult.config_name")
        reject_nonpositive(self.timestep_hours, "SimulationResult.timestep_hours")
        if not isinstance(self.records, pd.DataFrame):
            raise ValueError("SimulationResult.records must be a pandas DataFrame")

    @property
    def summary(self) -> dict[str, float]:
        from .comfort import comfort_summary

        return comfort_summary(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "timestep_hours": self.timestep_hours,
            "weather_category": self.weather_category.value,
            "summary": self.summary,
        }

