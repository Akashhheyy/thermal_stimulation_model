"""Dynamic thermal mass behaviour based on ``Q = m * cp * dT``.

The thermal mass is a real heat-storage node, not a fixed temperature
offset.  When the shelter air is warmer than the mass, the mass absorbs
heat and warms; when the air is cooler, stored heat flows back out and the
mass cools.  In the lumped simulation the mass is fully coupled to the zone
air, so its capacitance slows the indoor response (see ``simulation.py`` for
the exact discretisation and the reported ``thermal_mass_heat_flow_w``).
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import ThermalMass
from .validation import reject_negative, reject_nonpositive

__all__ = [
    "effective_capacitance_j_k",
    "stored_energy_j",
    "temperature_change_c",
    "heat_flow_w",
    "heat_absorbed_j",
    "heat_released_j",
    "ThermalMassState",
]


def effective_capacitance_j_k(mass_kg: float, specific_heat_j_kgk: float) -> float:
    """Heat capacity of the mass in J/K: ``mass * specific heat``."""
    reject_nonpositive(mass_kg, "mass_kg")
    reject_nonpositive(specific_heat_j_kgk, "specific_heat_j_kgk")
    return mass_kg * specific_heat_j_kgk


def stored_energy_j(
    mass_kg: float,
    specific_heat_j_kgk: float,
    temperature_c: float,
    reference_temperature_c: float = 0.0,
) -> float:
    """Sensible heat stored relative to a reference temperature, in joules.

    ``Q = m * cp * (T - T_reference)``.  The reference is 0 C by default so
    the value is a consistent inventory number rather than an absolute
    internal energy.
    """
    reject_nonpositive(mass_kg, "mass_kg")
    reject_nonpositive(specific_heat_j_kgk, "specific_heat_j_kgk")
    return mass_kg * specific_heat_j_kgk * (temperature_c - reference_temperature_c)


def temperature_change_c(
    energy_j: float,
    mass_kg: float,
    specific_heat_j_kgk: float,
) -> float:
    """Temperature change caused by an energy input: ``dT = Q / (m * cp)``."""
    reject_nonpositive(mass_kg, "mass_kg")
    reject_nonpositive(specific_heat_j_kgk, "specific_heat_j_kgk")
    return energy_j / (mass_kg * specific_heat_j_kgk)


def heat_flow_w(
    mass_kg: float,
    specific_heat_j_kgk: float,
    temperature_change_c: float,
    dt_seconds: float,
) -> float:
    """Storage rate in watts over one timestep.

    Positive means the mass is absorbing heat from the air (it is warming);
    negative means it is releasing stored heat back to the air (it is
    cooling).  This is the quantity reported as
    ``thermal_mass_heat_flow_w`` in the simulation output.
    """
    reject_nonpositive(mass_kg, "mass_kg")
    reject_nonpositive(specific_heat_j_kgk, "specific_heat_j_kgk")
    reject_nonpositive(dt_seconds, "dt_seconds")
    return mass_kg * specific_heat_j_kgk * temperature_change_c / dt_seconds


def heat_absorbed_j(
    mass_kg: float,
    specific_heat_j_kgk: float,
    from_temperature_c: float,
    to_temperature_c: float,
) -> float:
    """Energy taken in while warming from one temperature to another (J >= 0)."""
    delta = to_temperature_c - from_temperature_c
    return mass_kg * specific_heat_j_kgk * max(delta, 0.0)


def heat_released_j(
    mass_kg: float,
    specific_heat_j_kgk: float,
    from_temperature_c: float,
    to_temperature_c: float,
) -> float:
    """Energy given up while cooling from one temperature to another (J >= 0)."""
    delta = from_temperature_c - to_temperature_c
    return mass_kg * specific_heat_j_kgk * max(delta, 0.0)


@dataclass
class ThermalMassState:
    """Temperature state of one thermal mass carried across timesteps."""

    mass_kg: float
    specific_heat_j_kgk: float
    temperature_c: float

    def __post_init__(self) -> None:
        reject_nonpositive(self.mass_kg, "ThermalMassState.mass_kg")
        reject_nonpositive(self.specific_heat_j_kgk, "ThermalMassState.specific_heat_j_kgk")

    @classmethod
    def from_thermal_mass(cls, mass: ThermalMass) -> "ThermalMassState":
        """Build a mutable state from a validated :class:`ThermalMass`."""
        return cls(
            mass_kg=mass.mass_kg,
            specific_heat_j_kgk=mass.specific_heat_j_kgk,
            temperature_c=mass.initial_temperature_c,
        )

    @property
    def heat_capacity_j_k(self) -> float:
        return effective_capacitance_j_k(self.mass_kg, self.specific_heat_j_kgk)

    def absorb(self, energy_j: float) -> float:
        """Take in heat (J >= 0); the mass temperature rises. Returns the new temperature."""
        reject_negative(energy_j, "energy_j")
        self.temperature_c += temperature_change_c(energy_j, self.mass_kg, self.specific_heat_j_kgk)
        return self.temperature_c

    def release(self, energy_j: float) -> float:
        """Give up heat (J >= 0); the mass temperature falls. Returns the new temperature."""
        reject_negative(energy_j, "energy_j")
        self.temperature_c -= temperature_change_c(energy_j, self.mass_kg, self.specific_heat_j_kgk)
        return self.temperature_c

    def storage_rate_w(
        self,
        previous_temperature_c: float,
        current_temperature_c: float,
        dt_seconds: float,
    ) -> float:
        """Heat flow into the mass over one step (W); positive means absorbing."""
        return heat_flow_w(
            self.mass_kg,
            self.specific_heat_j_kgk,
            current_temperature_c - previous_temperature_c,
            dt_seconds,
        )

    def stored_energy_j(self, reference_temperature_c: float = 0.0) -> float:
        return stored_energy_j(
            self.mass_kg,
            self.specific_heat_j_kgk,
            self.temperature_c,
            reference_temperature_c,
        )
