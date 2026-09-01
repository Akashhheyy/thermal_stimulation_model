"""Input validation helpers for the passive shelter model.

All validation raises a clear ValueError on invalid input so that callers
cannot silently pass wrong physical data into the thermal engine.
"""
from __future__ import annotations


def require_field(value, name: str) -> None:
    """Raise if a required field is missing (None or empty str)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing required input: {name!r}")


def reject_negative(value, name: str) -> None:
    """Raise if a numeric field must be nonnegative but is negative."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


def reject_nonpositive(value, name: str) -> None:
    """Raise if a numeric field must be positive but is zero or negative."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def reject_out_of_range(value, name: str, low: float, high: float) -> None:
    """Raise if a numeric field falls outside an inclusive physical range."""
    if not (low <= value <= high):
        raise ValueError(f"{name} must be within [{low}, {high}], got {value}")


def collect_errors(*partials) -> list[str]:
    """Combine several (name, error) tuples into a list of messages."""
    return [f"{name}: {message}" for name, message in partials if message]


def validate_shelter_config(config) -> list[str]:
    """Cross-field sanity checks that individual dataclasses cannot express.

    Returns a list of human-readable problems; an empty list means the config
    is internally consistent.  Construction-time dataclass validation already
    rejects bad field values, so this covers relationships between fields.
    """
    from .geometry import CARDINAL_DIRECTIONS, cardinal_direction

    problems: list[str] = []
    try:
        built = build_geometry(config.geometry, config.openings)
    except ValueError as error:
        problems.append(str(error))
        return problems

    orientation_cardinal = cardinal_direction(config.geometry.orientation_deg)
    if orientation_cardinal not in CARDINAL_DIRECTIONS:
        problems.append(f"orientation resolves to unknown wall {orientation_cardinal!r}")

    if config.openings.window_area_m2 > 0.0 and built.window_area_by_orientation_m2.get(
        config.openings.window_wall_orientation.strip().lower(), 0.0
    ) <= 0.0:
        problems.append("window area was not assigned to any wall")

    if config.openings.door_area_m2 > 0.0 and built.door_area_by_orientation_m2.get(
        config.openings.door_wall_orientation.strip().lower(), 0.0
    ) <= 0.0:
        problems.append("door area was not assigned to any wall")

    if built.net_wall_area_m2 <= 0.0:
        problems.append("openings consume the entire wall area; no conductive wall remains")

    if config.thermal_mass is not None and config.thermal_mass.heat_capacity_j_k <= 0.0:
        problems.append("thermal mass heat capacity must be positive")

    if config.initial_indoor_temperature_c < -100.0:
        problems.append("initial indoor temperature below -100 C is outside the model range")

    return problems
