"""Component-by-component conductive heat transfer.

Sign convention (used everywhere in this package):

    Positive heat flow means heat entering the shelter air.
    Negative heat flow means heat leaving the shelter air.

Every component therefore uses ``Q = U * A * (T_out - T_in)`` for the
envelope and ``Q = U * A * (T_ground - T_in)`` for the floor.  When the
outdoor air is colder than indoors the value is negative (a loss).  The wall,
roof, floor, window, and door paths are kept separate because design
comparison needs per-component numbers; the shelter is never collapsed into
one anonymous UA value.
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import RectangularGeometry
from .models import EnvelopeAssembly, Openings

__all__ = [
    "ComponentHeatFlows",
    "conduction_heat_w",
    "component_transfers",
    "total_conductive_u_a_w_k",
]


def conduction_heat_w(
    u_value_w_m2k: float,
    area_m2: float,
    outdoor_temperature_c: float,
    indoor_temperature_c: float,
) -> float:
    """Conductive heat flow in watts; positive into the shelter."""
    if area_m2 < 0.0:
        raise ValueError("area_m2 must be nonnegative")
    return u_value_w_m2k * area_m2 * (outdoor_temperature_c - indoor_temperature_c)


@dataclass
class ComponentHeatFlows:
    """Conductive heat flow per envelope path, all in watts.

    Positive values warm the shelter, negative values cool it.  ``total_w``
    is the simple sum, which is the net conductive exchange with the
    environment.
    """

    wall_w: float
    roof_w: float
    floor_w: float
    window_w: float
    door_w: float

    @property
    def total_w(self) -> float:
        return self.wall_w + self.roof_w + self.floor_w + self.window_w + self.door_w

    def as_dict(self) -> dict[str, float]:
        return {
            "wall_heat_transfer_w": self.wall_w,
            "roof_heat_transfer_w": self.roof_w,
            "floor_heat_transfer_w": self.floor_w,
            "window_heat_transfer_w": self.window_w,
            "door_heat_transfer_w": self.door_w,
            "total_conductive_heat_transfer_w": self.total_w,
        }


def component_transfers(
    geometry: RectangularGeometry,
    wall_assembly: EnvelopeAssembly,
    roof_assembly: EnvelopeAssembly,
    floor_assembly: EnvelopeAssembly,
    openings: Openings,
    outdoor_temperature_c: float,
    indoor_temperature_c: float,
    ground_temperature_c: float | None = None,
) -> ComponentHeatFlows:
    """Compute per-component conductive heat flows for one instant.

    ``ground_temperature_c`` defaults to the outdoor temperature when not
    given; pass a measured ground temperature for better floor behaviour.
    """
    ground = outdoor_temperature_c if ground_temperature_c is None else ground_temperature_c
    wall = conduction_heat_w(
        wall_assembly.u_value_w_m2k,
        geometry.net_wall_area_m2,
        outdoor_temperature_c,
        indoor_temperature_c,
    )
    roof = conduction_heat_w(
        roof_assembly.u_value_w_m2k,
        geometry.roof_area_m2,
        outdoor_temperature_c,
        indoor_temperature_c,
    )
    floor = conduction_heat_w(
        floor_assembly.u_value_w_m2k,
        geometry.floor_area_m2,
        ground,
        indoor_temperature_c,
    )
    window = conduction_heat_w(
        openings.window_u_value_w_m2k,
        openings.window_area_m2,
        outdoor_temperature_c,
        indoor_temperature_c,
    )
    door = conduction_heat_w(
        openings.door_u_value_w_m2k,
        openings.door_area_m2,
        outdoor_temperature_c,
        indoor_temperature_c,
    )
    return ComponentHeatFlows(wall_w=wall, roof_w=roof, floor_w=floor, window_w=window, door_w=door)


def total_conductive_u_a_w_k(
    geometry: RectangularGeometry,
    wall_assembly: EnvelopeAssembly,
    roof_assembly: EnvelopeAssembly,
    floor_assembly: EnvelopeAssembly,
    openings: Openings,
) -> float:
    """Sum of U * A over every envelope path (W/K), for diagnostics only.

    The simulation itself never uses this aggregate; it exists so callers can
    sanity-check a design at a glance.
    """
    return (
        wall_assembly.u_value_w_m2k * geometry.net_wall_area_m2
        + roof_assembly.u_value_w_m2k * geometry.roof_area_m2
        + floor_assembly.u_value_w_m2k * geometry.floor_area_m2
        + openings.window_u_value_w_m2k * openings.window_area_m2
        + openings.door_u_value_w_m2k * openings.door_area_m2
    )
