"""Deterministic rectangular-shelter geometry.

The shelter is a rectangular prism.  ``orientation_deg`` is the compass
bearing (degrees clockwise from north, 0 to 360) that the short ``width_m``
side faces.  Wall areas are reported per cardinal direction so openings can
be subtracted from the correct wall.

Only axis-aligned orientations are resolved to a single cardinal wall in this
first version.  Diagonal orientations are snapped to the nearest cardinal
direction and that assumption is documented in docs/shelter_model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Openings, ShelterGeometry

__all__ = [
    "CARDINAL_DIRECTIONS",
    "RectangularGeometry",
    "cardinal_direction",
    "build_geometry",
    "derived_areas",
]

CARDINAL_DIRECTIONS = ("north", "east", "south", "west")


def cardinal_direction(angle_deg: float) -> str:
    """Snap a compass bearing to the nearest cardinal direction."""
    normalized = angle_deg % 360.0
    if normalized < 45.0 or normalized >= 315.0:
        return "north"
    if normalized < 135.0:
        return "east"
    if normalized < 225.0:
        return "south"
    return "west"


@dataclass
class RectangularGeometry:
    """Derived areas and volumes for one rectangular shelter.

    ``wall_area_by_orientation_m2`` values are NET areas: the gross wall area
    minus any window or door area assigned to that wall.  Conductive heat
    transfer uses these net areas so openings are never double counted.
    """

    length_m: float
    width_m: float
    height_m: float
    orientation_deg: float
    floor_area_m2: float
    roof_area_m2: float
    gross_wall_area_m2: float
    volume_m3: float
    gross_wall_area_by_orientation_m2: dict[str, float]
    window_area_by_orientation_m2: dict[str, float]
    door_area_by_orientation_m2: dict[str, float]
    wall_area_by_orientation_m2: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for direction in CARDINAL_DIRECTIONS:
            self.wall_area_by_orientation_m2.setdefault(direction, 0.0)

    @property
    def net_wall_area_m2(self) -> float:
        """Total wall area available for conduction after openings."""
        return sum(self.wall_area_by_orientation_m2.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "orientation_deg": self.orientation_deg,
            "floor_area_m2": self.floor_area_m2,
            "roof_area_m2": self.roof_area_m2,
            "gross_wall_area_m2": self.gross_wall_area_m2,
            "volume_m3": self.volume_m3,
            "gross_wall_area_by_orientation_m2": dict(self.gross_wall_area_by_orientation_m2),
            "window_area_by_orientation_m2": dict(self.window_area_by_orientation_m2),
            "door_area_by_orientation_m2": dict(self.door_area_by_orientation_m2),
            "wall_area_by_orientation_m2": dict(self.wall_area_by_orientation_m2),
            "net_wall_area_m2": self.net_wall_area_m2,
        }


def build_geometry(geometry: ShelterGeometry, openings: Openings | None = None) -> RectangularGeometry:
    """Derive areas from a :class:`ShelterGeometry`, subtracting openings.

    Raises ``ValueError`` when openings exceed the wall area they belong to.
    """
    if not isinstance(geometry, ShelterGeometry):
        raise ValueError("build_geometry expects a ShelterGeometry")
    openings = openings or Openings()

    floor = geometry.length_m * geometry.width_m
    roof = floor
    volume = floor * geometry.height_m
    short_wall = geometry.width_m * geometry.height_m
    long_wall = geometry.length_m * geometry.height_m
    gross_total = 2.0 * short_wall + 2.0 * long_wall

    # The wall facing orientation_deg carries the short side.
    front = cardinal_direction(geometry.orientation_deg)
    back = cardinal_direction(geometry.orientation_deg + 180.0)
    left = cardinal_direction(geometry.orientation_deg + 90.0)
    right = cardinal_direction(geometry.orientation_deg + 270.0)

    gross: dict[str, float] = {d: 0.0 for d in CARDINAL_DIRECTIONS}
    gross[front] += short_wall
    gross[back] += short_wall
    gross[left] += long_wall
    gross[right] += long_wall

    windows: dict[str, float] = {d: 0.0 for d in CARDINAL_DIRECTIONS}
    doors: dict[str, float] = {d: 0.0 for d in CARDINAL_DIRECTIONS}
    for label, mapping, area, target in (
        ("window", windows, openings.window_area_m2, openings.window_wall_orientation),
        ("door", doors, openings.door_area_m2, openings.door_wall_orientation),
    ):
        if area <= 0.0:
            continue
        wall = str(target).strip().lower()
        if wall not in CARDINAL_DIRECTIONS:
            raise ValueError(
                f"Openings.{label}_wall_orientation must be one of "
                f"{CARDINAL_DIRECTIONS}, got {wall!r}"
            )
        mapping[wall] += area

    for direction in CARDINAL_DIRECTIONS:
        opening_total = windows[direction] + doors[direction]
        if opening_total > gross[direction] + 1e-9:
            raise ValueError(
                f"Openings on the {direction} wall ({opening_total:.4f} m2) exceed the "
                f"available gross wall area ({gross[direction]:.4f} m2)"
            )

    net = {d: gross[d] - windows[d] - doors[d] for d in CARDINAL_DIRECTIONS}
    return RectangularGeometry(
        length_m=geometry.length_m,
        width_m=geometry.width_m,
        height_m=geometry.height_m,
        orientation_deg=geometry.orientation_deg,
        floor_area_m2=floor,
        roof_area_m2=roof,
        gross_wall_area_m2=gross_total,
        volume_m3=volume,
        gross_wall_area_by_orientation_m2=gross,
        window_area_by_orientation_m2=windows,
        door_area_by_orientation_m2=doors,
        wall_area_by_orientation_m2=net,
    )


def derived_areas(geometry: ShelterGeometry, openings: Openings | None = None) -> dict[str, float]:
    """Convenience dict of the headline derived areas and volume."""
    built = build_geometry(geometry, openings)
    return {
        "floor_area_m2": built.floor_area_m2,
        "roof_area_m2": built.roof_area_m2,
        "gross_wall_area_m2": built.gross_wall_area_m2,
        "net_wall_area_m2": built.net_wall_area_m2,
        "volume_m3": built.volume_m3,
    }
