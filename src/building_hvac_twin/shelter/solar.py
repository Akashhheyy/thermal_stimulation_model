"""Solar heat gain through shelter openings.

FIRST-VERSION ASSUMPTIONS (deliberately simple, and stated plainly):

1. ``solar_radiation_w_m2`` is treated as the radiation already incident on
   the opening plane.  No solar-position, incidence-angle, or sky-diffuse
   split is modelled yet.
2. The gain is ``radiation * opening area * solar_heat_gain_coefficient``
   times an orientation factor between 0 and 1.  The default orientation
   factor is 1.0, meaning "use the given radiation as-is".
3. Roof and opaque-wall solar absorption are not converted to indoor heat in
   this version; they are listed as future work in docs/shelter_model.

The module keeps a ``SolarGainModel`` seam so solar position, shading, and
per-surface exposure can be added later without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import CARDINAL_DIRECTIONS

__all__ = [
    "SolarGainModel",
    "solar_heat_gain_w",
    "orientation_factor",
]


def solar_heat_gain_w(
    solar_radiation_w_m2: float,
    opening_area_m2: float,
    solar_heat_gain_coefficient: float,
    orientation_factor: float = 1.0,
) -> float:
    """Direct solar heat gain through one opening, in watts.

    ``solar_heat_gain_coefficient`` is the fraction of incident radiation that
    ends up as indoor heat.  The result is never negative.
    """
    if solar_radiation_w_m2 < 0.0:
        raise ValueError("solar_radiation_w_m2 must be nonnegative")
    if opening_area_m2 < 0.0:
        raise ValueError("opening_area_m2 must be nonnegative")
    if not 0.0 <= solar_heat_gain_coefficient <= 1.0:
        raise ValueError("solar_heat_gain_coefficient must be within [0, 1]")
    if not 0.0 <= orientation_factor <= 1.0:
        raise ValueError("orientation_factor must be within [0, 1]")
    return max(
        solar_radiation_w_m2 * opening_area_m2 * solar_heat_gain_coefficient * orientation_factor,
        0.0,
    )


def orientation_factor(
    opening_orientation: str,
    solar_azimuth_deg: float | None = None,
) -> float:
    """Weighting for how directly an opening faces the sun.

    Without a solar azimuth (the current default path) every orientation gets
    1.0, which matches assumption 2 above.  When a future caller supplies an
    azimuth, openings facing away from the sun are damped toward 0.
    """
    direction = str(opening_orientation).strip().lower()
    if direction not in CARDINAL_DIRECTIONS:
        raise ValueError(f"opening_orientation must be one of {CARDINAL_DIRECTIONS}, got {opening_orientation!r}")
    if solar_azimuth_deg is None:
        return 1.0
    compass = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}[direction]
    difference = abs((float(solar_azimuth_deg) - compass + 180.0) % 360.0 - 180.0)
    # 0 degrees apart -> fully exposed, 180 degrees apart -> shaded.
    return max(0.0, 1.0 - difference / 180.0)


@dataclass
class SolarGainModel:
    """Configurable solar-gain seam for future solar-position models."""

    include_roof_gain: bool = False
    roof_absorptivity: float = 0.0
    shading_factor: float = 1.0
    ground_reflectivity: float = 0.0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.shading_factor <= 1.0:
            raise ValueError("shading_factor must be within [0, 1]")
        if not 0.0 <= self.roof_absorptivity <= 1.0:
            raise ValueError("roof_absorptivity must be within [0, 1]")
        if not 0.0 <= self.ground_reflectivity <= 1.0:
            raise ValueError("ground_reflectivity must be within [0, 1]")

    def opening_gain_w(
        self,
        solar_radiation_w_m2: float,
        opening_area_m2: float,
        solar_heat_gain_coefficient: float,
        opening_orientation: str = "south",
        solar_azimuth_deg: float | None = None,
    ) -> float:
        """Solar gain through one opening including the model's shading factor."""
        factor = orientation_factor(opening_orientation, solar_azimuth_deg) * self.shading_factor
        return solar_heat_gain_w(
            solar_radiation_w_m2,
            opening_area_m2,
            solar_heat_gain_coefficient,
            orientation_factor=factor,
        )
