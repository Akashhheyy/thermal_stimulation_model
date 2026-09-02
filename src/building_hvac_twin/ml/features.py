"""Deterministic ML feature preparation for the shelter dataset.

Feature selection follows the repository's established recommendations and is
mapped onto the exact column names present in ``data/shelter_ml_dataset.csv``.
Only two derived quantities are created, both from existing columns:

- ``orientation_cardinal``: the compass direction of ``orientation_deg``
  (reuses ``shelter.geometry.cardinal_direction``).
- ``thermal_mass_heat_capacity_kj_k``: ``thermal_mass_heat_capacity_j_k / 1000``.

Feature ordering is fixed by ``FEATURE_COLUMNS`` so training and prediction
always see identical column order.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..shelter.geometry import cardinal_direction

__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "FEATURE_COLUMNS",
    "ML_TARGETS",
    "DERIVED_COLUMN_SOURCE",
    "prepare_features",
]

# Categorical design features (exact dataset column names).
CATEGORICAL_FEATURES = (
    "wall_material",
    "window_wall_orientation",
    "orientation_cardinal",  # derived from orientation_deg
    "thermal_mass_material",
)

# Numeric design and weather-scenario features (exact dataset column names).
NUMERIC_FEATURES = (
    "length_m",
    "width_m",
    "height_m",
    "floor_area_m2",
    "volume_m3",
    "wall_insulation_thickness_m",
    "roof_insulation_thickness_m",
    "floor_insulation_thickness_m",
    "window_area_m2",
    "window_u_value_w_m2k",
    "door_area_m2",
    "window_solar_heat_gain_coefficient",
    "thermal_mass_heat_capacity_kj_k",  # derived from thermal_mass_heat_capacity_j_k
    "net_wall_area_m2",
    # Weather scenario features (NASA POWER scenario summaries).
    "mean_outdoor_temperature_c",
    "minimum_outdoor_temperature_c",
    "maximum_outdoor_temperature_c",
    "daily_solar_sum_wh_m2",
)

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# The physically meaningful regression targets actually present in the
# dataset.  performance_score is deliberately absent: it is a relative
# cohort-dependent ranking metric, not an absolute physical target.  No
# auxiliary heating/cooling target exists because the thermal engine is
# passive-only and computes none.
ML_TARGETS = (
    "percent_time_comfortable",
    "degree_hours_below_comfort",
    "degree_hours_above_comfort",
    "minimum_indoor_temperature_c",
    "mean_indoor_temperature_c",
    "indoor_temperature_range_c",
    "total_heat_loss_kwh",
    "total_solar_gain_kwh",
    "thermal_mass_net_kwh",
)

# Where each derived feature comes from (documented, not invented physics).
DERIVED_COLUMN_SOURCE = {
    "orientation_cardinal": "orientation_deg",
    "thermal_mass_heat_capacity_kj_k": "thermal_mass_heat_capacity_j_k",
}


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the ML feature matrix with fixed column order.

    Adds the two documented derived columns (when their source columns are
    present) and reorders to ``FEATURE_COLUMNS``.  The input frame is not
    mutated.  Callers may deliver the derived value directly when the source
    column is unavailable.
    """
    data = frame.copy()
    if "orientation_deg" in data.columns:
        data["orientation_cardinal"] = data["orientation_deg"].map(
            lambda value: cardinal_direction(float(value))
        )
    if "thermal_mass_heat_capacity_j_k" in data.columns:
        data["thermal_mass_heat_capacity_kj_k"] = (
            pd.to_numeric(data["thermal_mass_heat_capacity_j_k"], errors="raise") / 1000.0
        )
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"feature preparation is missing columns: {missing}")
    return data.loc[:, list(FEATURE_COLUMNS)]


def build_feature_row(config_features: dict[str, Any], weather_features: dict[str, Any]) -> pd.DataFrame:
    """Assemble one feature row from design and weather feature dicts.

    Unknown keys are rejected so callers cannot silently feed the wrong
    quantity under the right name.
    """
    row: dict[str, Any] = {}
    row.update(config_features)
    row.update(weather_features)
    known = set(FEATURE_COLUMNS) | set(DERIVED_COLUMN_SOURCE.values())
    unknown = set(row) - known
    if unknown:
        raise ValueError(f"unknown feature keys: {sorted(unknown)}")
    frame = pd.DataFrame([row])
    return prepare_features(frame)
